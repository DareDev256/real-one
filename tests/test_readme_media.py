"""scripts/readme_media.py: the README can never show a test render as real, and footage/own/ never leaves the Mac."""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("readme_media", REPO / "scripts" / "readme_media.py")
rm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rm)


@pytest.mark.parametrize("tag, run, test", [
    ("TEST_LOCAL_web_sample", {"mode": "keyed", "engine": "SAM 3.1"}, True),           # the tag alone stamps it
    ("REAL_FIRST", {"mode": "local test", "engine": "SAM 3.1"}, True),                  # local mask mode stamps it
    ("REAL_FIRST", {"mode": "keyed", "engine": "Apple Vision (...), TEST ONLY"}, True),  # a test engine stamps it
    ("REAL_FIRST", {"mode": "keyed", "engine": "SAM 3.1"}, False),
])
def test_only_a_keyed_sam_render_goes_unstamped(tag, run, test):
    assert rm.is_test(tag, run) is test


def test_footage_own_is_refused(tmp_path, monkeypatch):
    own = tmp_path / "footage" / "own"
    own.mkdir(parents=True)
    monkeypatch.setattr(rm, "OUT", own)
    for suffix in ("_run.json", "_collage.mp4", "_receipt.mp4"):
        (own / f"X_hook{suffix}").write_text("{}")
    with pytest.raises(rm.Refuse, match="footage/own"):
        rm.inputs("X", "hook", "collage")


def test_a_missing_receipt_look_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "OUT", tmp_path)
    for suffix in ("_run.json", "_collage.mp4"):
        (tmp_path / f"X_hook{suffix}").write_text("{}")
    with pytest.raises(rm.Refuse, match="missing receipt"):
        rm.inputs("X", "hook", "collage")


def test_a_render_without_a_freeze_is_refused():
    with pytest.raises(rm.Refuse, match="no freeze"):
        rm.freeze_start_s({"freeze": {"freezes": []}})


def test_a_keyed_render_of_the_test_song_needs_stand_in_and_is_stamped(tmp_path):
    """A keyed SAM render of the synthetic test song (Nov 1: a stock clip through the real models) is not the
    artist's film: without --stand-in it is refused (caught by content, so a renamed copy counts); with it, the
    images carry the stand-in band. A real master goes unstamped."""
    song = tmp_path / "renamed.wav"
    song.write_bytes(rm.TEST_SONG.read_bytes())
    run = {"mode": "demo", "engine": "SAM 3.1 'the person'", "song": str(song)}
    with pytest.raises(rm.Refuse, match="--stand-in"):
        rm.stamp_text("showcase3", run)
    assert rm.stamp_text("showcase3", run, stand_in=True) == rm.STAND_IN_STAMP
    master = tmp_path / "master.wav"
    master.write_bytes(b"RIFF the artist's own master")
    assert rm.stamp_text("real", dict(run, song=str(master))) is None
    assert rm.stamp_text("TEST_LOCAL_x", run) == rm.TEST_STAMP


def test_the_gif_shows_the_landing_or_else_the_release():
    lands = {"freeze": {"freezes": [{"frames": "90-236"}]}}            # 3 s of take before the freeze
    t0, t1 = rm.clip_times(lands)
    assert t0 == pytest.approx(2.0) and 3.0 < t1 < 236 / 30
    opens = {"freeze": {"freezes": [{"frames": "0-146"}]}}             # the take starts frozen (Nov 1 showcase3)
    t0, t1 = rm.clip_times(opens)
    assert t0 == pytest.approx(146 / 30 - 1.6) and t0 < 146 / 30 < t0 + 3.2  # the release is inside the GIF
