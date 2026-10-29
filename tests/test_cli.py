"""The keyed path refuses cleanly without MODEL_API_KEY, before any work, and never prints a key."""
import os
import subprocess
import sys
import time

import pytest

from conftest import REPO


def run(args, env_extra=None, cwd=REPO):
    env = {k: v for k, v in os.environ.items() if k != "MODEL_API_KEY"}
    env.update(env_extra or {})
    t = time.time()
    p = subprocess.run([sys.executable, "-m", "realone"] + args, cwd=cwd, env=env, capture_output=True, text=True)
    return p, time.time() - t


@pytest.fixture
def junk(tmp_path):
    c, s = tmp_path / "take.mov", tmp_path / "song.wav"
    c.write_bytes(b"not a video")
    s.write_bytes(b"not audio")
    return str(c), str(s)


@pytest.mark.parametrize("extra", [["--mask", "sam", "--world-image", "x"], ["--mask", "sam", "--local-plate"],
                                   ["--mask", "local", "--world", "rain-soaked alley"]])
def test_keyed_paths_refuse_without_key(junk, extra, tmp_path):
    if "--world-image" in extra:
        img = tmp_path / "w.png"
        img.write_bytes(b"x")
        extra = [x if x != "x" else str(img) for x in extra]
    p, secs = run(["render", "--clip", junk[0], "--song", junk[1]] + extra)
    assert p.returncode == 2, p.stderr
    assert "REFUSED" in p.stderr
    if "--local-plate" in extra:
        assert "test-only" in p.stderr
    else:
        assert "MODEL_API_KEY is not set" in p.stderr and "Nothing was called" in p.stderr
    assert secs < 10  # refused at the gate: no audio analysis, no ffmpeg, no network


def test_no_worlds_refuses(junk):
    p, _ = run(["render", "--clip", junk[0], "--song", junk[1], "--mask", "local"])
    assert p.returncode == 2 and "no worlds" in p.stderr


def test_a_key_is_never_echoed(junk):
    secret = "sk-test-" + "Q7" * 16
    p, _ = run(["render", "--clip", junk[0], "--song", junk[1], "--mask", "sam", "--world", "x"],
               {"MODEL_API_KEY": secret})
    assert p.returncode == 2  # gets past the gate, then refuses on the junk song: still no network call
    assert secret not in p.stdout and secret not in p.stderr
    assert "present" in p.stdout


@pytest.mark.parametrize("extra,msg", [
    (["--worlds-json", "MISSING"], "--worlds-json not found"),
    (["--section-world", "hook"], "KEY=PROMPT"),
    (["--section-world", "hook=rooftop at dusk"], "MODEL_API_KEY is not set"),
    (["--worlds-json", "PROMPTS"], "MODEL_API_KEY is not set"),
    (["--local-plate", "--workers", "0"], "--workers must be at least 1"),
])
def test_song_flags_refuse_at_the_gate(junk, tmp_path, extra, msg):
    j = tmp_path / "w.json"
    j.write_text('{"hook": ["neon alley"], "default": ["rooftop"]}')
    extra = [str(tmp_path / "nope.json") if x == "MISSING" else str(j) if x == "PROMPTS" else x for x in extra]
    p, secs = run(["render", "--clip", junk[0], "--song", junk[1], "--mask", "local", "--section", "all"] + extra)
    assert p.returncode == 2 and msg in p.stderr, p.stderr
    assert secs < 10
