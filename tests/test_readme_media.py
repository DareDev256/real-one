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
