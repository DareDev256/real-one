"""The director CLI: refuses cleanly without a key (before any work), refuses a quote over the cap before any
client exists, never prints a key, and a --dry-run plan writes outputs labelled FIXTURE. Outputs go to tmp dirs."""
import json
import os
import subprocess
import sys
import time

import pytest

from conftest import REPO
import realone.director.__main__ as dm

LYRICS = str(REPO / "tests" / "fixtures" / "director" / "test_song_lyrics.txt")
SECRET = "sk-test-" + "Z9" * 16


def run(args, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k != "MODEL_API_KEY"}
    env.update(env_extra or {})
    t = time.time()
    p = subprocess.run([sys.executable, "-m", "realone.director"] + args, cwd=REPO, env=env, capture_output=True,
                       text=True)
    return p, time.time() - t


@pytest.fixture
def junk(tmp_path):
    c, s = tmp_path / "take.mov", tmp_path / "song.wav"
    c.write_bytes(b"not a video")
    s.write_bytes(b"not audio")
    return str(c), str(s)


@pytest.mark.parametrize("cmd", [["plan"], ["direct", "--clip", "CLIP"]])
def test_keyed_runs_refuse_without_a_key_before_any_work(junk, cmd):
    args = [x if x != "CLIP" else junk[0] for x in cmd] + ["--song", junk[1]]
    p, secs = run(args)
    assert p.returncode == 2, p.stderr
    assert "MODEL_API_KEY is not set" in p.stderr and "Nothing was called" in p.stderr
    assert secs < 10                            # refused at the gate: no audio analysis, no ffmpeg, no network


def test_over_the_cap_refuses_before_any_client_exists(song, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dm, "OUT", tmp_path)
    monkeypatch.setenv("MODEL_API_KEY", SECRET)

    def no_client(*a, **k):
        raise AssertionError("a Meta client was built although the quote was over the cap")
    monkeypatch.setattr(dm, "MetaBackend", no_client)
    rc = dm.main(["plan", "--song", str(song[0]), "--cap-usd", "0.05"])
    out = capsys.readouterr()
    assert rc == 2 and "over the cap" in out.err and "nothing was called" in out.err
    assert "TOTAL ceiling" in out.out
    assert SECRET not in out.out and SECRET not in out.err


def test_dry_run_plan_writes_fixture_labelled_outputs(song, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dm, "OUT", tmp_path)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    rc = dm.main(["plan", "--song", str(song[0]), "--lyrics", LYRICS, "--dry-run", "--tag", "t"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "DRY RUN spend: US$0.00" in out and "[FIXTURE]" in out
    rec = json.loads((tmp_path / "TEST_LOCAL_director_t_treatment.json").read_text())
    assert rec["mode"] == "DRY RUN" and "not model output" in rec["FIXTURE"]
    assert all(a["source"].startswith("FIXTURE") for a in rec["attempts"])
    assert [a["purpose"] for a in rec["attempts"]] == ["treatment", "treatment_repair"]
    assert rec["guard_reask"]["applied"] == [1, 2] and rec["still_literal"] == []
    hook = rec["sections"][2]
    assert hook["lyric_source"] == "artist" and hook["guard"]["verdict"] == "pass"
    assert hook["guard_first_pass"]["verdict"] == "reject" and "key" in hook["guard_first_pass"]["concrete"]
    assert [s["freeze"] for s in rec["sections"]] == [False, False, False, False, True]
    lyr = json.loads((tmp_path / "TEST_LOCAL_director_t_lyrics.json").read_text())
    assert lyr["sources"]["transcript"].startswith("FIXTURE")


def test_review_never_uploads_a_file_outside_out(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dm, "OUT", tmp_path / "out")
    (tmp_path / "out").mkdir()
    t = tmp_path / "t.json"
    t.write_text("{}")
    rc = dm.main(["review", "--render", "/etc/hosts", "--treatment", str(t), "--section", "2", "--dry-run"])
    assert rc == 2 and "only uploads renders under out/" in capsys.readouterr().err
