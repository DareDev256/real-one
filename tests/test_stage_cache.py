"""The stage prep is cached per take (Nov 4): the floor track and the foot clean-plate depend only on the take and its
matte, so a gauntlet that re-renders the same take all day computes them once. Any change to what they depend on
must compute them again."""
import os

import numpy as np

from realone.common import H, W
from realone.stage import cleanplate as CP
from realone.stage import pipeline as PL

N = 6


def _masks(fill=0):
    m = np.zeros((N, W * H // 8), np.uint8)
    m[:, : 1000] = fill
    return m


def _fake_track(calls):
    def track(frames, masks):
        calls.append(1)
        return dict(H_to0=np.stack([np.eye(3)] * N), step_inliers=np.full(N, 50), contact=np.full((N, 2), 7.0))
    return track


def test_the_floor_track_is_computed_once_per_take(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(PL.CAM, "ground_track", _fake_track(calls))
    monkeypatch.setattr(PL, "read_frames", lambda p: iter(()))
    proxy = tmp_path / "norm.mp4"
    proxy.write_bytes(b"x" * 100)
    a = PL.ground_track_cached(proxy, N, _masks(), tmp_path, log=lambda *_: None)
    b = PL.ground_track_cached(proxy, N, _masks(), tmp_path, log=lambda *_: None)
    assert len(calls) == 1
    assert all(np.array_equal(a[k], b[k]) for k in a)
    PL.ground_track_cached(proxy, N, _masks(fill=255), tmp_path, log=lambda *_: None)  # a re-cut matte
    assert len(calls) == 2
    st = proxy.stat()
    os.utime(proxy, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))  # a new proxy for the same name
    PL.ground_track_cached(proxy, N, _masks(), tmp_path, log=lambda *_: None)
    assert len(calls) == 3
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_the_feet_are_refined_once_per_matte_and_track(tmp_path, monkeypatch):
    calls = []

    def refine(proxy, n, masks, H_to0, contact, out_path, log=print, capture_shadow=False):
        calls.append(out_path)
        np.save(out_path, masks)
        return out_path, np.arange(n)
    monkeypatch.setattr(CP, "refine_masks", refine)
    mp = tmp_path / "take.masks.npy"
    np.save(mp, _masks())
    Hs, contact = np.stack([np.eye(3)] * N), np.full((N, 2), 7.0)
    p1, r1 = CP.refine_masks_cached("norm.mp4", N, _masks(), mp, Hs, contact, log=lambda *_: None)
    p2, r2 = CP.refine_masks_cached("norm.mp4", N, _masks(), mp, Hs, contact, log=lambda *_: None)
    assert len(calls) == 1 and p1 == p2 and np.array_equal(r1, r2)
    H2 = Hs.copy()
    H2[3, 0, 2] = 0.5  # the floor track moved on one frame
    CP.refine_masks_cached("norm.mp4", N, _masks(), mp, H2, contact, log=lambda *_: None)
    assert len(calls) == 2
    c2 = contact.copy()
    c2[0, 1] = 9.0  # a different contact point
    CP.refine_masks_cached("norm.mp4", N, _masks(), mp, Hs, c2, log=lambda *_: None)
    assert len(calls) == 3
    CP.refine_masks_cached("norm.mp4", N, _masks(), mp, Hs, contact, log=lambda *_: None, capture_shadow=True)
    assert len(calls) == 4  # no shadow was captured for this matte yet: refine again
