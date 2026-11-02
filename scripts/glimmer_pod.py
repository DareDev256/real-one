#!/usr/bin/env python3
"""Muse Glimmer 30B on a GPU we rent, for exactly as long as one command needs it.

    scripts/glimmer_pod.py run -- .venv/bin/python -m realone.director plan --brain glimmer --no-transcribe \
        --no-audio --song footage/test_song.wav --lyrics tests/fixtures/director/test_song_lyrics.txt --tag glimmer
    scripts/glimmer_pod.py status      # every realone-glimmer pod, its $/h and what it has cost so far
    scripts/glimmer_pod.py down        # delete every realone-glimmer pod, then prove none is left

`run` creates one RunPod pod (llama.cpp's CUDA server image, Meta's official Q4_K_M GGUF of Muse Glimmer 30B,
Apache-2.0), waits for the model to load, runs the command with GLIMMER_BASE_URL and GLIMMER_API_KEY in its
environment, and deletes the pod in a `finally`, so a crash or Ctrl-C still stops the bill. The server only answers
requests that carry a key made fresh for this run, because the pod's proxy URL is public.

Refuses, before creating anything: no RunPod key, a balance under US$1, or a realone-glimmer pod already running.
Key: ~/.secrets/runpod-api-key (never printed). Pod create fields: rest.runpod.io/v1/openapi.json, read 2026-11-01.
"""
import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

NAME = "realone-glimmer"
REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
HF_REPO = "meta-models/Muse-Glimmer-30B-GGUF"
HF_FILE = "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"   # 16.8 GB; the repo's 1.6 GB dflash file is a draft model
# 24 GB holds the 16.8 GB weights plus a 16k context; bigger cards are fallbacks, cheapest first by typical price
GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 5090",
        "NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA A40"]
MIN_BALANCE_USD = 1.0
LOAD_TIMEOUT_S = 40 * 60


class Refuse(Exception):
    pass


def key():
    p = Path.home() / ".secrets" / "runpod-api-key"
    k = os.environ.get("RUNPOD_API_KEY") or (p.read_text().strip() if p.exists() else "")
    if not k:
        raise Refuse("no RunPod key (RUNPOD_API_KEY or ~/.secrets/runpod-api-key)")
    return k


def call(method, path, body=None, base=REST):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json",
                                          "User-Agent": "realone-glimmer-pod/1.0"})  # urllib's default is blocked (CF 1010)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raise Refuse(f"RunPod {method} {path}: HTTP {e.code} {e.read()[:300].decode(errors='replace')}") from None


def balance():
    d = call("POST", "", {"query": "{ myself { clientBalance } }"}, base=GRAPHQL)
    return float(((d or {}).get("data") or {}).get("myself", {}).get("clientBalance") or 0.0)


def ours():
    return [p for p in (call("GET", "/pods") or []) if str(p.get("name", "")).startswith(NAME)]


def url_of(pod_id):
    return f"https://{pod_id}-8080.proxy.runpod.net"


def cost_so_far(p):
    t0 = p.get("lastStartedAt") or p.get("createdAt")
    try:
        import datetime as dt
        start = dt.datetime.fromisoformat(str(t0).replace("Z", "+00:00"))
        hrs = (dt.datetime.now(dt.timezone.utc) - start).total_seconds() / 3600
    except (ValueError, TypeError):
        return None, None
    rate = float(p.get("costPerHr") or 0)
    return round(hrs * 60, 1), round(hrs * rate, 3)


def up(api_key):
    if ours():
        raise Refuse(f"a {NAME} pod is already running ({', '.join(p['id'] for p in ours())}); `down` it first")
    b = balance()
    if b < MIN_BALANCE_USD:
        raise Refuse(f"RunPod balance is US${b:.2f}, under US${MIN_BALANCE_USD:.2f}; top it up first (nothing created)")
    cmd = ["--hf-repo", HF_REPO, "--hf-file", HF_FILE, "-ngl", "99", "-c", "16384", "--host", "0.0.0.0",
           "--port", "8080", "--jinja", "--api-key", api_key]
    last = None
    for cloud in ("COMMUNITY", "SECURE"):
        try:
            pod = call("POST", "/pods", {"name": NAME, "imageName": IMAGE, "gpuTypeIds": GPUS, "gpuCount": 1,
                                         "cloudType": cloud, "containerDiskInGb": 40, "ports": ["8080/http"],
                                         "dockerStartCmd": cmd})
            print(f"glimmer_pod: created {pod['id']} ({cloud}, {pod.get('costPerHr', '?')} US$/h)")
            return pod
        except Refuse as e:
            last = e
            print(f"glimmer_pod: {cloud} cloud refused: {str(e)[:200]}")
    raise Refuse(f"no GPU from the list in either cloud: {last}")


def wait_ready(pod_id):
    """llama.cpp answers /health with 503 while it downloads and loads the weights, then 200."""
    t0, said = time.time(), None
    while time.time() - t0 < LOAD_TIMEOUT_S:
        try:
            with urllib.request.urlopen(url_of(pod_id) + "/health", timeout=20) as r:
                if r.status == 200:
                    print(f"glimmer_pod: model loaded in {time.time() - t0:.0f} s")
                    return time.time() - t0
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            msg = getattr(e, "code", None) or type(e).__name__
            if msg != said:
                print(f"glimmer_pod: waiting ({msg}) ...")
                said = msg
        time.sleep(15)
    raise Refuse(f"the model did not load in {LOAD_TIMEOUT_S // 60} min")


def down():
    pods = ours()
    for p in pods:
        mins, usd = cost_so_far(p)
        call("DELETE", f"/pods/{p['id']}")
        print(f"glimmer_pod: deleted {p['id']} after {mins} min, about US${usd}")
    left = ours()
    if left:
        raise Refuse(f"pods still listed after delete: {', '.join(p['id'] for p in left)}")
    print("glimmer_pod: no realone-glimmer pod left")
    return len(pods)


def status():
    pods = ours()
    for p in pods:
        mins, usd = cost_so_far(p)
        print(f"{p['id']}  {p.get('desiredStatus')}  {p.get('costPerHr')} US$/h  up {mins} min  about US${usd}  "
              f"{(p.get('gpu') or {}).get('displayName', '')}")
    if not pods:
        print("no realone-glimmer pod")
    print(f"balance US${balance():.2f}")


def run(command):
    if not command:
        raise Refuse("run needs a command after --")
    api_key = secrets.token_urlsafe(24)
    pod = up(api_key)
    try:
        wait_ready(pod["id"])
        env = dict(os.environ, GLIMMER_BASE_URL=url_of(pod["id"]) + "/v1", GLIMMER_API_KEY=api_key)
        return subprocess.run(command, env=env).returncode
    finally:
        down()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("action", choices=["run", "status", "down"])
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    try:
        if a.action == "run":
            return run(cmd)
        if a.action == "status":
            status()
        else:
            down()
        return 0
    except Refuse as e:
        print(f"glimmer_pod: REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
