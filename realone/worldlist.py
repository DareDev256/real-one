"""Which worlds each section gets (0.2.0).

One render has ONE global world list (deduplicated, in first-seen order); every section gets a list of ids into it.
Sources, in the order a section looks them up:
  --worlds-json FILE   {"hook": [...], "hook:2": [...], "#3": [...], "verse": [...], "default": [...]}
                       each item a Muse Image prompt (a string, or {"prompt": "..."}) or {"image": "path"}
  --section-world KEY=PROMPT   repeatable; KEY is the same as a json key (hook, hook:2, #3, default)
  --world / --worlds-file / --world-image   the 0.1.0 flags: the list for every section with nothing more specific
Lookup per section: "#<index>", "<role>:<occurrence>", "<role>", "default", then the 0.1.0 flags.
--local-plate (TEST ONLY) uses the three procedural plates; each role starts the rotation on a different plate
so sections read as different places.
"""
import json
from pathlib import Path

from realone.common import Refuse

PLATE_ROT = {"hook": 0, "verse": 1, "intro": 2, "bridge": 2, "outro": 1}


def _entry(x, where):
    if isinstance(x, str) and x.strip():
        return ("prompt", x.strip())
    if isinstance(x, dict) and isinstance(x.get("prompt"), str) and x["prompt"].strip():
        return ("prompt", x["prompt"].strip())
    if isinstance(x, dict) and isinstance(x.get("image"), str):
        if not Path(x["image"]).exists():
            raise Refuse(f"{where}: world image not found: {x['image']}")
        return ("image", x["image"])
    raise Refuse(f"{where}: each world is a prompt string, {{\"prompt\": ...}} or {{\"image\": path}} (got {x!r})")


def load_config(a, default_prompts):
    """-> (default entries, {key: [entries]}). Validates everything that can be validated before any work."""
    default = [("image", p) for p in (a.world_image or [])] + [("prompt", p) for p in default_prompts]
    per = {}
    if getattr(a, "worlds_json", None):
        p = Path(a.worlds_json)
        if not p.exists():
            raise Refuse(f"--worlds-json not found: {p}")
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise Refuse(f"--worlds-json is not valid JSON: {e}") from None
        if not isinstance(data, dict):
            raise Refuse("--worlds-json must be an object: {\"hook\": [...], \"verse\": [...], \"default\": [...]}")
        for k, items in data.items():
            items = items if isinstance(items, list) else [items]
            if not items:
                raise Refuse(f"--worlds-json {k}: empty list")
            per[str(k).strip()] = [_entry(x, f"--worlds-json {k}") for x in items]
    for s in getattr(a, "section_world", None) or []:
        k, sep, v = s.partition("=")
        if not sep or not k.strip() or not v.strip():
            raise Refuse(f"--section-world wants KEY=PROMPT, e.g. hook=\"rooftop at dusk\" (got {s!r})")
        per.setdefault(k.strip(), []).append(("prompt", v.strip()))
    return default, per


def prompts_in(default, per):
    return [v for kind, v in default + [e for es in per.values() for e in es] if kind == "prompt"]


def assign(sections, default, per, local_plate):
    """-> (entries: the global world list, per_section: [[ids]], sources: [which key each section used])."""
    entries, ids, per_section, sources = [], {}, [], []

    def idx(e):
        if e not in ids:
            ids[e] = len(entries)
            entries.append(e)
        return ids[e]

    for s in sections:
        keys = [f"#{s.get('index', -1)}", f"{s['role']}:{s.get('occurrence', 1)}", s["role"], "default"]
        hit = next((k for k in keys if k in per), None)
        if hit is not None:
            lst, src = per[hit], hit
        elif default and not local_plate:
            lst, src = default, "--world flags"
        elif local_plate:
            r = PLATE_ROT.get(s["role"], 0)
            lst, src = [("plate", (r + j) % 3) for j in range(3)], "local plates (TEST ONLY)"
        else:
            raise Refuse(f"no world for {s['role']} #{s.get('occurrence', 1)} (bars {s['start_bar']}-{s['end_bar']}):"
                         f" add it to --worlds-json, or give a \"default\" list or --world")
        per_section.append([idx(e) for e in lst])
        sources.append(src)
    return entries, per_section, sources
