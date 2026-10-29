"""Shot list for one section: cuts on downbeats every N bars; with one take, a cut is a world swap and a reframe.

Framing is a virtual camera on the artist: wide (the whole frame), medium and close (crops centred on the
artist's mask bounding box, smoothed over time inside each shot so the camera breathes instead of tracking).
"""
import numpy as np

from realone.common import FPS, H, W, Refuse

FRAMINGS = {"wide": 1.0, "medium": 1.4, "close": 1.9}
PATTERN = ("wide", "medium", "close", "medium")
TARGET_Y = {"wide": 0.5, "medium": 0.40, "close": 0.17}  # fraction down the artist's bbox the camera centres on
SMOOTH_S = 0.6


def plan_shots(tl, section, cut_bars=2, n_worlds=3, pattern=PATTERN):
    """-> plan dict. Every cut is a detected downbeat, rounded to the nearest output frame."""
    if cut_bars not in (1, 2, 4, 8):
        raise Refuse(f"--cut-bars must be 1, 2, 4 or 8 (got {cut_bars})")
    db = tl["downbeats"]
    bpb = tl["beats_per_bar"]
    a, b = section["start_bar"], section["end_bar"]
    s0 = float(db[a])
    s1 = float(db[b]) if b < len(db) else float(section["end_s"])
    n_frames = int(round((s1 - s0) * FPS))
    phase = tl["beats"].index(db[0]) if db[0] in tl["beats"] else 0
    bars = list(range(a, b, cut_bars))
    starts = [int(round((db[bar] - s0) * FPS)) for bar in bars]
    shots = []
    for k, (bar, f0) in enumerate(zip(bars, starts)):
        f1 = starts[k + 1] if k + 1 < len(starts) else n_frames
        framing = pattern[k % len(pattern)]
        cut_t = s0 + f0 / FPS
        shots.append({
            "index": k, "start_frame": f0, "end_frame": f1, "frames": f1 - f0,
            "cut_time_s": round(cut_t, 4), "bar": bar, "beat_index": phase + bar * bpb, "beat_in_bar": 1,
            "downbeat_s": round(float(db[bar]), 4), "cut_error_ms": round((cut_t - db[bar]) * 1000, 2),
            "world": k % n_worlds, "framing": framing, "zoom": FRAMINGS[framing],
        })
    return {"section": section, "start_s": round(s0, 4), "end_s": round(s1, 4), "n_frames": n_frames,
            "fps": FPS, "cut_bars": cut_bars, "bpm": tl["bpm"], "beats_per_bar": bpb, "shots": shots}


def bboxes(masks, step=4):
    """Per-frame artist bbox (x0, y0, x1, y1) from packed masks, measured on a 1/step grid. NaN when empty."""
    out = np.full((len(masks), 4), np.nan)
    for i, row in enumerate(masks):
        m = np.unpackbits(row)[:W * H].reshape(H, W)[::step, ::step]
        ys, xs = np.nonzero(m)
        if len(xs):
            out[i] = (xs.min() * step, ys.min() * step, xs.max() * step + step, ys.max() * step + step)
    return out


def _smooth(x, sigma):
    if len(x) < 2 or sigma <= 0:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.pad(x, r, mode="edge")
    return np.convolve(xp, k, mode="valid")


def camera_tracks(plan, boxes):
    """Per-frame crop (x0, y0, cw, ch) in source pixels, per shot: centre on the smoothed bbox, clamp to frame."""
    n = plan["n_frames"]
    crops = np.zeros((n, 4))
    b = boxes.copy()
    # fill empty frames from neighbours (an empty mask must not throw the camera)
    good = ~np.isnan(b[:, 0])
    if not good.any():
        b[:] = (0, 0, W, H)
    else:
        idx = np.arange(len(b))
        for j in range(4):
            b[:, j] = np.interp(idx, idx[good], b[good, j])
    for sh in plan["shots"]:
        f0, f1 = sh["start_frame"], sh["end_frame"]
        z = sh["zoom"]
        cw, ch = W / z, H / z
        seg = b[f0:f1]
        cx = _smooth((seg[:, 0] + seg[:, 2]) / 2, SMOOTH_S * FPS)
        top = _smooth(seg[:, 1], SMOOTH_S * FPS)
        hgt = _smooth(seg[:, 3] - seg[:, 1], SMOOTH_S * FPS)
        ty = top + TARGET_Y[sh["framing"]] * hgt
        if z == 1.0:
            crops[f0:f1] = (0, 0, W, H)
            continue
        x0 = np.clip(cx - cw / 2, 0, W - cw)
        y0 = np.clip(ty - ch / 2, 0, H - ch)
        crops[f0:f1, 0], crops[f0:f1, 1], crops[f0:f1, 2], crops[f0:f1, 3] = x0, y0, cw, ch
    return crops
