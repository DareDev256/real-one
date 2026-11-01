"""Camera moves on the beat (realone/stage/moves.py): the grammar, the refusals, and each profile's shape."""
import numpy as np
import pytest

from realone.common import FPS, H, W, Refuse
from realone.stage import moves as MV

BAR_S = 2.0


def _tl():
    db = [i * BAR_S for i in range(16)]
    secs = [dict(index=0, role="verse", start_bar=0, end_bar=8, bars=8, end_s=16.0, occurrence=1),
            dict(index=1, role="hook", start_bar=8, end_bar=16, bars=8, end_s=32.0, occurrence=1)]
    return dict(downbeats=db, sections=secs), secs


def _f(bar):
    return int(round(bar * BAR_S * FPS))


def test_grammar_and_refusals():
    assert MV.parse("dolly:4@hook") == ("dolly", 4, "hook")
    assert MV.parse("crash@verse") == ("crash", None, "verse")
    for bad in ("dolly", "zoom@hook", "dolly:0@hook", "crash:2@hook", "dolly:x@hook"):
        with pytest.raises(Refuse):
            MV.parse(bad)


def test_resolve_places_moves_on_downbeats_and_refuses_what_it_cannot_do():
    tl, secs = _tl()
    ms = MV.resolve(tl, ["dolly:2@hook", "crash@verse"], secs, {})
    d = next(m for m in ms if m["kind"] == "dolly")
    assert (d["start_frame"], d["end_frame"]) == (_f(8), _f(10))
    c = next(m for m in ms if m["kind"] == "crash")
    assert c["downbeat_frames"] == [_f(b) for b in range(0, 8)]
    with pytest.raises(Refuse, match="runs past"):
        MV.resolve(tl, ["dolly:9@hook"], secs, {})
    with pytest.raises(Refuse, match="Hook Freeze"):
        MV.resolve(tl, ["swap@hook"], secs, {}, freezes=[dict(start_bar=8, end_bar=10)])


def test_dolly_eases_in_holds_to_the_next_cut_then_resets():
    tl, secs = _tl()
    ms = MV.resolve(tl, ["dolly:2@hook"], secs, {})
    n = _f(16)
    dz = MV.dolly_profile(n, ms, shot_starts=[0, _f(8), _f(12)], dz_m=3.0)
    assert dz[: _f(8)].max() == 0                               # nothing before the hook
    assert dz[_f(8)] < 0.01 and abs(dz[_f(10) - 1] - 3.0) < 0.05  # eases from 0 to the full 3 m over 2 bars
    assert np.all(np.diff(dz[_f(8):_f(10)]) >= -1e-9)           # never backs off mid-move
    assert abs(dz[_f(11)] - 3.0) < 1e-9                         # holds after the move
    assert dz[_f(12):].max() == 0                               # the next cut resets it


def test_crash_snaps_on_each_downbeat_and_resets_on_the_next():
    tl, secs = _tl()
    ms = MV.resolve(tl, ["crash@hook"], secs, {})
    n = _f(16)
    boxes = np.tile(np.array([400.0, 300, 700, 1800]), (n, 1))
    sc, cx, cy = MV.crash_profile(n, ms, boxes)
    d = _f(9)
    assert sc[d - 1] > 1.2 and sc[d] < sc[d - 1]                # the next downbeat starts a new snap from low
    assert abs(sc[d + MV.CRASH_FRAMES] - MV.CRASH_SCALE) < 1e-6   # fully in after CRASH_FRAMES
    assert sc[: _f(8)].max() == 1.0                             # nothing outside the section
    assert abs(cy[d] - (300 + 0.12 * 1500)) < 1                 # aimed at the face, not the frame centre


def test_apply_crash_fills_the_frame_and_keeps_its_size():
    f = np.full((H, W, 3), 200, np.uint8)
    out = MV.apply_crash(f, 1.3, 10.0, 10.0)                    # a centre near the corner is clamped
    assert out.shape == f.shape and out.min() == 200            # no empty border anywhere


def test_swap_cycles_the_sections_worlds_by_bar_and_refuses_one_world():
    tl, secs = _tl()
    ms = MV.resolve(tl, ["swap@hook"], secs, {})
    n = _f(16)
    fw = MV.swap_worlds(n, ms, np.zeros(n, int), lambda f: [2, 5])
    assert fw[_f(8)] == 2 and fw[_f(9)] == 5 and fw[_f(10)] == 2 and fw[_f(8) - 1] == 0
    with pytest.raises(Refuse, match="at least 2"):
        MV.swap_worlds(n, ms, np.zeros(n, int), lambda f: [3])


def test_designed_changes_cover_every_snap_and_swap():
    tl, secs = _tl()
    ms = MV.resolve(tl, ["crash@verse", "swap@hook"], secs, {})
    ch = MV.designed_changes(ms)
    assert all(d + k in ch for d in ms[0]["downbeat_frames"] for k in range(MV.CRASH_FRAMES + 1))
    assert {_f(b) for b in range(8, 16)} <= ch
