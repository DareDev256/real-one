"""Camera moves on the beat: Higgsfield-style presets, done with Meta's models and the artist's filmed pixels.

  --move dolly[:BARS]@SECTION   Dolly zoom (the Vertigo shot): from the section's first downbeat, over BARS bars
                                (default 2), the WORLD's camera pulls back DOLLY_M metres while its lens zooms in so
                                that everything at the artist's distance keeps its size. The artist and the floor
                                under his feet hold still; the world behind him stretches. Needs --stage tracked
                                (it needs the world's depth and the take's camera). Holds the end state to the
                                section's next cut, then resets.
  --move crash@SECTION          Crash zoom: on every downbeat of the section the whole frame snaps CRASH_SCALE x
                                toward the artist's face over CRASH_FRAMES frames (a zoom lens: no parallax, which is
                                right for a zoom), with zoom blur on the snap, and holds to the next downbeat.
  --move swap@SECTION           World swap: a new world on every bar of the section (cycling the section's world
                                list) while the framing runs on uninterrupted; with --stage tracked every world is
                                pinned to the same tracked floor, so his feet stay put through each swap.

SECTION uses the --section/--freeze grammar (hook, hook:2, #3, bars:A-B, verse). Refuses anything it cannot do
honestly (a dolly without the stage, a swap with fewer than 2 worlds, a swap inside a freeze)."""
import numpy as np
import cv2

from realone.common import FPS, H, W, Refuse
from realone.freeze import select_sections

KINDS = ("dolly", "crash", "swap")
DOLLY_M = 3.0
DOLLY_BARS = 2
CRASH_SCALE = 1.28
CRASH_FRAMES = 4


def parse(spec):
    """'dolly:4@hook' -> ('dolly', 4, 'hook'); 'crash@verse' -> ('crash', None, 'verse')."""
    if "@" not in spec:
        raise Refuse(f"--move {spec}: write KIND[:BARS]@SECTION, e.g. dolly:2@hook, crash@hook, swap@verse")
    head, sec = spec.split("@", 1)
    kind, _, bars = head.partition(":")
    if kind not in KINDS:
        raise Refuse(f"--move {spec}: kind must be one of {', '.join(KINDS)}")
    if bars and (not bars.isdigit() or int(bars) < 1):
        raise Refuse(f"--move {spec}: BARS must be a whole number of bars, 1 or more")
    if bars and kind != "dolly":
        raise Refuse(f"--move {spec}: only dolly takes a bar count")
    return kind, (int(bars) if bars else None), sec


def resolve(tl, specs, rendered, plan, freezes=()):
    """--move values -> [{kind, spec, start_bar, end_bar, start_frame, end_frame, downbeat_frames, role}]."""
    if not specs:
        return []
    db = tl["downbeats"]
    a = rendered[0]["start_bar"]
    s0 = float(db[a])
    end_s = float(db[rendered[-1]["end_bar"]]) if rendered[-1]["end_bar"] < len(db) else float(rendered[-1]["end_s"])

    def frame_of(bar):
        t = float(db[bar]) if bar < len(db) else end_s
        return int(round((t - s0) * FPS))

    out = []
    for spec in specs:
        kind, bars, secspec = parse(spec)
        for s in select_sections(tl, secspec, rendered):
            b0 = s["start_bar"]
            b1 = s["end_bar"] if kind != "dolly" else min(s["end_bar"], b0 + (bars or DOLLY_BARS))
            if kind == "dolly" and bars and b0 + bars > s["end_bar"]:
                raise Refuse(f"--move {spec}: {bars} bars runs past the {s['role']} (bars {b0}-{s['end_bar']})")
            if kind == "swap" and any(f["start_bar"] < s["end_bar"] and b0 < f["end_bar"] for f in freezes):
                raise Refuse(f"--move {spec}: a world swap inside a Hook Freeze would break the freeze (one world "
                             f"falls, hangs and resumes); pick another section")
            out.append(dict(kind=kind, spec=spec, role=s["role"], start_bar=b0, end_bar=b1,
                            section_end_bar=s["end_bar"], start_frame=frame_of(b0), end_frame=frame_of(b1),
                            section_end_frame=frame_of(s["end_bar"]),
                            downbeat_frames=[frame_of(x) for x in range(b0, s["end_bar"])]))
    return sorted(out, key=lambda m: (m["start_frame"], m["kind"]))


def specs_from_treatment(path, rendered):
    """A director treatment (realone/director: sections[].move) -> --move specs for the sections being rendered.
    Accepts the treatment itself or a director record that holds it under "treatment"."""
    import json
    from pathlib import Path
    obj = json.loads(Path(path).read_text())
    t = obj.get("treatment", obj)
    if not isinstance(t, dict) or not isinstance(t.get("sections"), list):
        raise Refuse(f"--moves-from {path}: not a treatment (no sections list)")
    inside = {s.get("index") for s in rendered}
    out = []
    for s in t["sections"]:
        mv = s.get("move", "none")
        if mv not in ("none",) + KINDS:
            raise Refuse(f"--moves-from {path}: section {s.get('index')} has an unknown move {mv!r}")
        if mv != "none" and s.get("index") in inside:
            out.append(f"{mv}@#{s['index']}")
    return out


def _ease(u):
    u = np.clip(u, 0, 1)
    return u * u * u * (u * (6 * u - 15) + 10)  # smootherstep: no velocity step at either end


def dolly_profile(n, moves, shot_starts, dz_m=DOLLY_M):
    """-> dz metres per frame: eases in over the move's bars, holds to the section's next cut, 0 elsewhere."""
    dz = np.zeros(n)
    cuts = sorted(set(int(x) for x in shot_starts))
    for m in moves:
        if m["kind"] != "dolly":
            continue
        f0, f1 = m["start_frame"], max(m["start_frame"] + 1, m["end_frame"])
        nxt = next((c for c in cuts if c > f1), m["section_end_frame"])
        nxt = min(nxt, m["section_end_frame"], n)
        idx = np.arange(f0, min(nxt, n))
        dz[idx] = dz_m * _ease((idx - f0) / (f1 - f0))
    return dz


def crash_profile(n, moves, boxes):
    """-> (scale, cx, cy) per frame: a snap toward the face on each downbeat of a crash section, held to the next
    downbeat. The point is the top-centre of the artist's box, 12% down (his face), smoothed."""
    scale = np.ones(n)
    cx = np.full(n, W / 2.0)
    cy = np.full(n, H / 2.0)
    ok = ~np.isnan(boxes[:, 0])
    if ok.any():
        fx = np.where(ok, (boxes[:, 0] + boxes[:, 2]) / 2, np.nan)
        fy = np.where(ok, boxes[:, 1] + 0.12 * (boxes[:, 3] - boxes[:, 1]), np.nan)
        idx = np.arange(n)
        fx = np.interp(idx, idx[ok], fx[ok])
        fy = np.interp(idx, idx[ok], fy[ok])
        k = np.ones(9) / 9
        cx = np.convolve(np.pad(fx, 4, mode="edge"), k, mode="valid")
        cy = np.convolve(np.pad(fy, 4, mode="edge"), k, mode="valid")
    for m in moves:
        if m["kind"] != "crash":
            continue
        beats = m["downbeat_frames"] + [m["section_end_frame"]]
        for d0, d1 in zip(beats, beats[1:]):
            for i in range(d0, min(d1, n)):
                u = (i - d0 + 1) / CRASH_FRAMES
                scale[i] = 1 + (CRASH_SCALE - 1) * (1 - (1 - min(u, 1.0)) ** 3)  # ease-out snap, then hold
    return scale, cx, cy


def apply_crash(frame, s, cx, cy, s_prev=1.0):
    """Scale the finished frame by s about (cx, cy); on a snap (s moved this frame) a
    zoom blur: 3 scales between the last and this one, averaged (the lens moved during the exposure)."""
    if s <= 1.0001:
        return frame

    def zoom(img, z):
        # zooming IN (z >= 1) about any point inside the frame only ever samples inside it: no border can show
        x, y = float(np.clip(cx, 0, W - 1)), float(np.clip(cy, 0, H - 1))
        M = np.float32([[z, 0, (1 - z) * x], [0, z, (1 - z) * y]])
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    if abs(s - s_prev) > 0.01:
        acc = np.zeros(frame.shape, np.float32)
        for z in np.linspace(s_prev, s, 3):
            acc += zoom(frame, max(z, 1.0)).astype(np.float32)
        return np.clip(acc / 3, 0, 255).astype(frame.dtype)
    return zoom(frame, s)


def swap_worlds(n, moves, shot_world_per_frame, section_ids_of_frame):
    """-> world id per frame: the shot's world, except inside a swap section, where bar k shows the section's
    world list [k mod len]. section_ids_of_frame: per frame, the world ids the section owns."""
    fw = np.array(shot_world_per_frame, int).copy()
    for m in moves:
        if m["kind"] != "swap":
            continue
        ids = section_ids_of_frame(m["start_frame"])
        if len(set(ids)) < 2:
            raise Refuse(f"--move {m['spec']}: the {m['role']} has one world; a swap needs at least 2 "
                         f"(give it more with --worlds-json or --section-world)")
        beats = m["downbeat_frames"] + [m["section_end_frame"]]
        for k, (d0, d1) in enumerate(zip(beats, beats[1:])):
            fw[d0:min(d1, n)] = ids[k % len(ids)]
    return fw


def designed_changes(moves):
    """Frames where a move changes the whole frame on purpose (the cut detector must not read them as cuts)."""
    out = set()
    for m in moves:
        if m["kind"] == "crash":
            for d in m["downbeat_frames"]:
                out |= set(range(d, d + CRASH_FRAMES + 1))
        elif m["kind"] == "swap":
            out |= set(m["downbeat_frames"])
    return out
