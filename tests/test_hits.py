"""Hits on the beat (realone/hits.py): the finished frame punches on downbeats, kicks smaller on backbeats, hits
harder with a jolt on the hook, and every frame a hit changes is reported to the cut check as designed."""
import numpy as np

from realone import hits as HT

FPS = 30


def _tl(hook=None):
    # 120 BPM, 4/4: a beat every 0.5 s, a downbeat every 2 s, 8 s long
    beats = [round(0.5 * k, 3) for k in range(16)]
    downs = [b for k, b in enumerate(beats) if k % 4 == 0]
    secs = [{"start_s": 0.0, "end_s": 4.0, "is_hook": False}, {"start_s": 4.0, "end_s": 8.0, "is_hook": bool(hook)}]
    return {"beats": beats, "downbeats": downs, "beats_per_bar": 4, "sections": secs}


def test_downbeat_punches_and_flashes():
    p = HT.profile(_tl(), 0.0, 8 * FPS, FPS)
    assert abs(p["punch"][0] - (1 + HT.PUNCH_DOWN)) < 1e-9
    assert abs(p["flash"][0] - HT.FLASH_DOWN) < 1e-9
    assert p["punch"][2 * FPS] > 1.04  # the next downbeat


def test_backbeat_smaller_and_no_flash():
    p = HT.profile(_tl(), 0.0, 8 * FPS, FPS)
    f2 = int(0.5 * FPS)  # beat 2 of bar 1
    assert abs(p["punch"][f2] - (1 + HT.PUNCH_BACK)) < 1e-9
    assert p["flash"][f2] == 0.0
    f3 = int(1.0 * FPS)  # beat 3 is not a backbeat
    assert p["punch"][f3] == 1.0


def test_punch_eases_back_to_rest():
    p = HT.profile(_tl(), 0.0, 8 * FPS, FPS)
    seg = p["punch"][0:HT.PUNCH_FRAMES + 1]
    assert all(seg[k] >= seg[k + 1] for k in range(len(seg) - 1))
    assert seg[-1] == 1.0 and p["punch"][HT.PUNCH_FRAMES + 2] == 1.0


def test_hook_hits_harder_and_jolts():
    plain = HT.profile(_tl(hook=False), 0.0, 8 * FPS, FPS)
    hook = HT.profile(_tl(hook=True), 0.0, 8 * FPS, FPS)
    f = 4 * FPS  # downbeat inside the second section
    assert hook["punch"][f] > plain["punch"][f]
    assert abs(hook["punch"][f] - (1 + HT.PUNCH_DOWN * HT.HOOK_GAIN)) < 1e-9
    assert np.hypot(hook["dx"][f], hook["dy"][f]) > 5
    assert np.hypot(plain["dx"][f], plain["dy"][f]) == 0
    assert not hook["dx"][:4 * FPS].any()  # no jolt outside the hook


def test_designed_frames_cover_each_hit():
    p = HT.profile(_tl(), 0.0, 8 * FPS, FPS)
    assert {0, 1, int(0.5 * FPS), 2 * FPS} <= p["designed"]
    assert int(1.0 * FPS) not in p["designed"]


def test_window_offset_and_bounds():
    p = HT.profile(_tl(), 2.0, 2 * FPS, FPS)  # a window starting on the second downbeat
    assert p["punch"].shape == (2 * FPS,)
    assert abs(p["punch"][0] - (1 + HT.PUNCH_DOWN)) < 1e-9


def test_no_beats_is_a_no_op():
    p = HT.profile({"beats": [], "downbeats": [], "sections": []}, 0.0, 30, FPS)
    assert (p["punch"] == 1.0).all() and not p["flash"].any() and not p["designed"]


def test_apply_identity_and_effects():
    img = np.zeros((64, 64, 3), np.uint8)
    img[24:40, 24:40] = 200
    assert HT.apply(img, 1.0, 0.0) is img
    flashed = HT.apply(img, 1.0, 0.25)
    assert flashed[32, 32, 0] > 230 and flashed[0, 0, 0] == 0  # highlights up, blacks stay black
    zoomed = HT.apply(img, 1.5, 0.0)
    assert (zoomed > 100).sum() > (img > 100).sum()  # the bright square grew
    moved = HT.apply(img, 1.0, 0.0, dx=8.0)
    assert moved[32, 44, 0] > 100 and img[32, 44, 0] == 0
