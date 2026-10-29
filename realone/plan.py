"""Shot list for one section or a run of sections: cuts on downbeats every N bars; with one take, a cut is a world
swap and a reframe.

Framing is a virtual camera on the artist: wide (the whole frame), medium and close (crops centred on the
artist's mask bounding box, smoothed over time inside each shot so the camera breathes instead of tracking).
The framing pattern restarts at every section, so each section opens wide.

Freezes (realone/freeze.py) change the cut list. `still` (0.2.0): a freeze always starts on a cut, and a cut
strictly inside a freeze or on its release is dropped, so the frozen world is the one that resumes; a section
boundary always stays a cut (a new section is a new world list). `bullet`: the freeze downbeat is NOT a cut, even on
a section boundary: the shot running into the freeze keeps its world, framing and camera through the downbeat, so
the particles the viewer watched falling are the ones that hang; cuts inside the freeze and on its release are
dropped too (the world that hung is the world that resumes), except a release that is itself a section boundary.
The first frame of the render is always a shot start.
"""
import numpy as np

from realone.common import FPS, H, W, Refuse

FRAMINGS = {"wide": 1.0, "medium": 1.4, "close": 1.9}
PATTERN = ("wide", "medium", "close", "medium")
TARGET_Y = {"wide": 0.5, "medium": 0.40, "close": 0.17}  # fraction down the artist's bbox the camera centres on
SMOOTH_S = 0.6


def plan_shots(tl, section, cut_bars=2, n_worlds=3, pattern=PATTERN, freezes=(), freeze_style="still"):
    """One section -> plan dict. Every cut is a detected downbeat, rounded to the nearest output frame.
    freezes: [{"start_bar", "end_bar", ...}] from freeze.resolve()."""
    return plan_range(tl, [section], cut_bars, [list(range(n_worlds))], pattern, freezes, freeze_style=freeze_style)


def plan_range(tl, sections, cut_bars=2, worlds=None, pattern=PATTERN, freezes=(), force_cuts=(),
               freeze_style="still"):
    """A contiguous run of sections -> plan dict.

    worlds: per section, a list of world ids (indices into the render's world list); default [0, 1, 2] for all.
    Shot k of a section uses framing pattern[k] and world worlds[section][k + r], where r rotates the list by one
    when the section's first world would repeat the previous shot's world. force_cuts: bars that must be cuts (a
    receipt reveal flips the look there, which is only clean on a cut)."""
    if cut_bars not in (1, 2, 4, 8):
        raise Refuse(f"--cut-bars must be 1, 2, 4 or 8 (got {cut_bars})")
    db = tl["downbeats"]
    bpb = tl["beats_per_bar"]
    a, b = sections[0]["start_bar"], sections[-1]["end_bar"]
    for s, s_next in zip(sections, sections[1:]):
        if s["end_bar"] != s_next["start_bar"]:
            raise Refuse(f"sections must be contiguous (bar {s['end_bar']} vs {s_next['start_bar']})")
    s0 = float(db[a])
    s1 = float(db[b]) if b < len(db) else float(sections[-1]["end_s"])
    n_frames = int(round((s1 - s0) * FPS))
    phase = tl["beats"].index(db[0]) if db[0] in tl["beats"] else 0
    worlds = worlds or [[0, 1, 2]] * len(sections)

    def frame_of(bar):
        t = float(db[bar]) if bar < len(db) else s1
        return int(round((t - s0) * FPS))

    fz = [(f["start_bar"], f["end_bar"]) for f in freezes]
    bullet = freeze_style == "bullet"
    cut_bars_list = []  # (bar, section position)
    for j, sec in enumerate(sections):
        bars = set(range(sec["start_bar"], sec["end_bar"], cut_bars))
        if not bullet:
            bars |= {f0 for f0, _ in fz if sec["start_bar"] <= f0 < sec["end_bar"]}  # a freeze starts on a cut
        bars |= {f for f in force_cuts if sec["start_bar"] <= f < sec["end_bar"]}
        for bar in sorted(bars):
            if bullet:
                if bar != a and any(f0 <= bar < f1 or (bar == f1 and bar != sec["start_bar"]) for f0, f1 in fz):
                    continue  # the downbeat, inside, the release: one world falls, hangs and resumes
            elif bar != sec["start_bar"] and any(f0 < bar <= f1 for f0, f1 in fz):
                continue  # inside the freeze or on its release: the frozen world is the one that resumes
            cut_bars_list.append((bar, j))
    starts = [frame_of(bar) for bar, _ in cut_bars_list]
    shots, k_in, prev_world = [], {}, None
    rot = {}
    for k, ((bar, j), f0) in enumerate(zip(cut_bars_list, starts)):
        f1 = starts[k + 1] if k + 1 < len(starts) else n_frames
        ki = k_in.get(j, 0)
        k_in[j] = ki + 1
        wl = worlds[j]
        if ki == 0:
            rot[j] = 1 if (len(wl) > 1 and wl[0] == prev_world) else 0
        world = wl[(ki + rot[j]) % len(wl)]
        framing = pattern[ki % len(pattern)]
        cut_t = s0 + f0 / FPS
        sec = sections[j]
        shots.append({
            "index": k, "start_frame": f0, "end_frame": f1, "frames": f1 - f0,
            "cut_time_s": round(cut_t, 4), "bar": bar, "beat_index": phase + bar * bpb, "beat_in_bar": 1,
            "downbeat_s": round(float(db[bar]), 4), "cut_error_ms": round((cut_t - db[bar]) * 1000, 2),
            "world": world, "framing": framing, "zoom": FRAMINGS[framing],
            "section_pos": j, "section_index": sec.get("index", -1), "role": sec.get("role", "-"),
        })
        prev_world = world
    fr = []
    for f in freezes:
        fr.append(dict(f, start_frame=frame_of(f["start_bar"]), end_frame=frame_of(f["end_bar"]),
                       start_s=round(float(db[f["start_bar"]]), 4)))
        fr[-1]["frames"] = fr[-1]["end_frame"] - fr[-1]["start_frame"]
    return {"section": sections[0] if len(sections) == 1 else
            {"role": "song", "start_bar": a, "end_bar": b, "start_s": s0, "end_s": s1, "bars": b - a},
            "sections": sections, "start_bar": a, "end_bar": b,
            "start_s": round(s0, 4), "end_s": round(s1, 4), "n_frames": n_frames,
            "fps": FPS, "cut_bars": cut_bars, "bpm": tl["bpm"], "beats_per_bar": bpb, "shots": shots, "freezes": fr}


def shot_index(plan):
    """-> int array: which shot each frame belongs to."""
    out = np.zeros(plan["n_frames"], int)
    for sh in plan["shots"]:
        out[sh["start_frame"]:sh["end_frame"]] = sh["index"]
    return out


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
