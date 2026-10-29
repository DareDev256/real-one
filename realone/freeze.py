"""Hook Freeze in the renderer (proven first in spike/freeze_test.py).

On a section's first downbeat the WORLD clock stops for `--freeze-bars` bars: the plate push-in, the drift, the
particles and the virtual camera (reframe and parallax) all hold, while the artist keeps moving at full rate and
the grain keeps animating (a frozen frame with live grain reads as a stopped world; with dead grain it reads as a
still image). The clock then resumes from where it stopped, one frame step at a time, so there is no jump.

Everything the world draws is a pure function of the world clock, so the freeze is provable after the render:
the world layer (plate + particles) is hashed every frame, and `checks()` fails the render when the hashes say
the world moved inside a freeze, stood still outside one, or jumped on the release.
"""
import cv2
import numpy as np

from realone.audio import pick_section
from realone.common import FPS, Refuse

EASE_S = 0.5  # after the release the camera eases from the held framing back to the live track


def select_sections(tl, spec, rendered):
    """Sections named by `spec` that START inside the rendered bars (the grammar of --section, plus role:last).
    None / "" / "section": the rendered section (when rendering the whole song: every hook); a bare role: every
    section with that role; "hook:2", "hook:last", "#3", "bars:A-B": that one."""
    a, b = rendered[0]["start_bar"], rendered[-1]["end_bar"]
    if spec in (None, "", "section"):
        cands = rendered if len(rendered) == 1 else [s for s in rendered if s["role"] == "hook"]
        label = "the rendered section" if len(rendered) == 1 else "every hook"
    elif spec.endswith(":last"):
        role = spec[:-5]
        cands, label = [s for s in tl["sections"] if s["role"] == role][-1:], spec
        if not cands:
            found = ", ".join(f"#{s['index']} {s['role']}" for s in tl["sections"])
            raise Refuse(f"{spec}: no {role} in this song (found: {found})")
    elif spec.startswith(("bars:", "#")) or ":" in spec:
        cands, label = [pick_section(tl, spec)], spec
    else:
        cands, label = [s for s in tl["sections"] if s["role"] == spec], f"every {spec}"
        if not cands:
            found = ", ".join(f"#{s['index']} {s['role']}" for s in tl["sections"])
            raise Refuse(f"{spec}: no {spec} in this song (found: {found})")
    inside = [s for s in cands if a <= s["start_bar"] < b]
    if not inside:
        raise Refuse(f"{spec or 'section'}: {label} does not start inside the rendered bars {a}-{b}")
    return inside


def resolve(tl, specs, bars_n, rendered):
    """--freeze values -> [{"spec", "section_index", "role", "occurrence", "start_bar", "end_bar"}].

    rendered: the contiguous sections being rendered. A spec is
      None / "" / "section": the rendered section (when rendering the whole song: every hook);
      a bare role ("hook"): every section with that role inside the rendered bars;
      "hook:2", "#3", "bars:A-B": that one section (the same grammar as --section).
    The freeze starts on the section's first downbeat and lasts bars_n bars; it may not run past the section or
    the rendered range (refused, not clipped)."""
    specs = [s for s in (specs or [])]
    if not specs:
        return []
    if bars_n < 1:
        raise Refuse(f"--freeze-bars must be at least 1 (got {bars_n})")
    a, b = rendered[0]["start_bar"], rendered[-1]["end_bar"]
    out = {}
    for spec in specs:
        for s in select_sections(tl, spec, rendered):
            fb0, fb1 = s["start_bar"], s["start_bar"] + bars_n
            name = f"{s['role']} (bars {s['start_bar']}-{s['end_bar']})"
            if fb1 > s["end_bar"]:
                raise Refuse(f"--freeze-bars {bars_n} is longer than {name}, which has {s['bars']} bars")
            if fb1 > b:
                raise Refuse(f"--freeze-bars {bars_n} on {name} runs past the rendered bars {a}-{b}")
            out[fb0] = {"spec": spec or "section", "section_index": s.get("index", -1), "role": s["role"],
                        "occurrence": s.get("occurrence", 1), "start_bar": fb0, "end_bar": fb1}
    spans = [out[k] for k in sorted(out)]
    for p, q in zip(spans, spans[1:]):
        if q["start_bar"] < p["end_bar"]:
            raise Refuse(f"freezes overlap at bar {q['start_bar']}")
    return spans


def world_clock(n, freezes):
    """-> float array: the world clock in frames. It advances 1 per frame, except inside a freeze, where every
    frame shows the clock of the freeze's first frame. The first frame after the freeze is one step on."""
    step = np.ones(n)
    step[0] = 0
    for f in freezes:
        step[f["start_frame"] + 1:f["end_frame"]] = 0
    return np.cumsum(step)


def hold_camera(crops, freezes, shot_of, ease_s=EASE_S):
    """The virtual camera holds its framing for the whole freeze, then eases back onto the live track (the
    artist kept moving, so the live track has moved on) over `ease_s`, stopping at the next cut."""
    out = crops.copy()
    n = len(crops)
    ease = max(1, int(round(ease_s * FPS)))
    for f in freezes:
        f0, f1 = f["start_frame"], f["end_frame"]
        held = crops[f0].copy()
        out[f0:f1] = held
        for j in range(ease):
            i = f1 + j
            if i >= n or shot_of[i] != shot_of[f1 - 1]:
                break
            t = (j + 1) / (ease + 1)
            k = t * t * (3 - 2 * t)
            out[i] = held + (crops[i] - held) * k
    return out


def push_fraction(wf, plan):
    """Per frame: how far through its shot's push-in the world is, measured on the world clock (it holds in a
    freeze and resumes where it stopped; without a freeze it is (i - shot start) / shot length, as in 0.1.0)."""
    p = np.zeros(len(wf))
    for sh in plan["shots"]:
        f0, f1 = sh["start_frame"], sh["end_frame"]
        span = max(1.0, wf[f1 - 1] - wf[f0] + 1)
        p[f0:f1] = (wf[f0:f1] - wf[f0]) / span
    return p


def checks(keys, plate_steps, wf, freezes, shot_of, artist_moved, control_moved):
    """The spike's proof, per freeze, on what was actually rendered.

    keys          per-frame hash of the world layer (plate + particles), in render order
    plate_steps   per-frame mean abs difference of the plate (no particles, 135x240) from the previous frame,
                  NaN where the previous frame was not rendered by the same worker
    artist_moved  per freeze: the artist's mask differs between the first and last frozen frame
    control_moved per freeze: the world at the unfrozen clock one frame later differs from the frozen world
                  (so the freeze, not a static world, is what held it)
    -> (report dict, ok)."""
    n = len(keys)
    moving_bad = [i for i in range(1, n) if shot_of[i] == shot_of[i - 1] and wf[i] != wf[i - 1]
                  and keys[i] == keys[i - 1]]
    rep = {"world_moves_outside_freezes": not moving_bad, "frames_that_should_move_but_did_not": moving_bad[:20],
           "freezes": []}
    ok = not moving_bad
    for k, f in enumerate(freezes):
        f0, f1 = f["start_frame"], f["end_frame"]
        r = {"frames": f"{f0}-{f1 - 1}", "n_frames": f1 - f0, "bars": f"{f['start_bar']}-{f['end_bar']}",
             "role": f["role"]}
        r["frozen_identical"] = f1 - f0 >= 2 and len(set(keys[f0:f1])) == 1
        before = [i for i in range(max(1, f0 - 30), f0 + 1) if shot_of[i] == shot_of[i - 1]]
        r["moves_before"] = all(keys[i] != keys[i - 1] for i in before) if before else None
        if not before:
            r["moves_before_note"] = "n/a: the freeze starts on a cut (the frames before are another shot)"
        if f1 >= n:
            r["resumes"], r["no_jump"] = None, None
            r["resume_note"] = "n/a: the freeze runs to the end of the render"
        elif shot_of[f1] != shot_of[f1 - 1]:
            r["resumes"], r["no_jump"] = None, None
            r["resume_note"] = "n/a: the release is a section cut (a new world)"
        else:
            nxt = [i for i in range(f1, min(n, f1 + 4)) if shot_of[i] == shot_of[f1]]
            r["resumes"] = all(keys[i] != keys[i - 1] for i in nxt)
            base = [plate_steps[i] for i in range(f1 + 1, min(n, f1 + 61))
                    if shot_of[i] == shot_of[f1] and np.isfinite(plate_steps[i])]
            step = plate_steps[f1]
            med = float(np.median(base)) if base else float("nan")
            r["release_plate_step"] = round(float(step), 4) if np.isfinite(step) else None
            r["plate_step_median_after"] = round(med, 4) if base else None
            r["clock_step_at_release"] = float(wf[f1] - wf[f1 - 1])
            r["no_jump"] = bool(wf[f1] - wf[f1 - 1] == 1 and np.isfinite(step) and base and step <= 3 * med + 0.05)
        r["artist_moves"] = bool(artist_moved[k])
        r["control_world_would_move"] = bool(control_moved[k])
        r["ok"] = bool(r["frozen_identical"] and r["artist_moves"] and r["control_world_would_move"]
                       and r["moves_before"] is not False and r["resumes"] is not False
                       and r["no_jump"] is not False)
        ok &= r["ok"]
        rep["freezes"].append(r)
    rep["ok"] = bool(ok)
    return rep, bool(ok)


# --------------------------------------------------------------------------- Hook Freeze v2: `bullet`
#
# `still` (0.2.0) stops everything in the world, the camera included: on a phone that reads as a still photo behind
# a moving person (green screen), because nothing in the frame proves that time stopped. `bullet` keeps the camera
# moving through the freeze: the particles hang mid-air on three depth layers and slide past each other at
# different rates (near most, the plate least) while the artist keeps performing. The freeze lands with a hit
# (a 3-frame punch-in and white flash, the world grading cool and desaturated) and releases on the next downbeat
# (colour snaps back, the particles fall again with a short streak burst, the camera carries on).
# Everything stays a pure function of the world clock and the frame index, so units render in parallel.

STYLES = ("bullet", "still")
LEAD_IN_BARS = 2     # a bullet freeze on the render's first frame gets this many bars of the same world first
CAM_TRAVEL = 0.035   # the plate slides this share of the frame width over a freeze (near particles 9x that)
PUNCH = (1.04, 1.022, 1.008)      # scale on the freeze downbeat and the 2 frames after, then 1.0
FLASH = (0.30, 0.15, 0.05)        # white flash, decays in 3 frames
GRADE_IN = (0.45, 0.8)            # the world's cool/desaturated grade ramps in over 3 frames (then 1.0)
BURST = (1.8, 1.4, 1.15)          # streak length (clock rate the motion blur is drawn at) for 3 frames after release


def _smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def lead_in(tl, secs, freezes, offset_s, max_seconds=None, bars=None):
    """A bullet freeze on the render's first frame: nobody would see the world move before it stops. Open `bars`
    (default LEAD_IN_BARS) earlier, inside the first section (so the lead-in is the start of the shot that runs
    into the freeze, in the world that then freezes), when the song and the take cover it and the render stays
    under max_seconds. -> (secs, info): info None when no freeze starts the render, else {"bars": 0 or N, "why"}."""
    nb = LEAD_IN_BARS if bars is None else bars
    if not freezes or freezes[0]["start_bar"] != secs[0]["start_bar"]:
        return secs, None
    sb, db = secs[0]["start_bar"], tl["downbeats"]
    why = None
    if sb - nb < 0:
        why = f"the song has {sb} bars before the freeze, {nb} needed"
    elif db[sb - nb] - offset_s < 0:
        why = f"the take starts {offset_s - db[sb - nb]:.2f}s after the lead-in would (song time {db[sb - nb]:.2f}s)"
    elif max_seconds is not None and secs[-1]["end_s"] - db[sb - nb] > max_seconds:
        why = f"with the lead-in the render would be over --max-seconds {max_seconds:.0f}"
    if why:
        return secs, {"bars": 0, "why": why + "; the render opens on the freeze downbeat, as before"}
    out = list(secs)
    out[0] = dict(secs[0], start_bar=sb - nb, start_s=round(float(db[sb - nb]), 4), bars=secs[0]["bars"] + nb,
                  lead_in_bars=nb)
    return out, {"bars": nb, "from_bar": sb - nb, "freeze_bar": sb,
                 "why": f"the freeze was the render's first frame: the world now falls for {nb} bars first, in the "
                        "world that then freezes"}


def bullet_camera(n, plan, freezes, shot_of, drift_px=16.0, ease_frames=None):
    """-> (dx px, zoom) per frame. The shot that holds a freeze gets a sideways truck on the REAL frame clock that
    eases in before the freeze downbeat and out after the release (smootherstep: no velocity step anywhere), moving
    on every frozen frame, with the plate travelling CAM_TRAVEL of the width between the downbeat and the release.
    The ease spans `ease_frames` (default: half the freeze) each side, clipped to the shot. A constant zoom margin per
    shot keeps the slide and the world drift off the plate's edge. Other shots: 0, 1."""
    dx, zoom = np.zeros(n), np.ones(n)
    for f in freezes:
        f0, f1 = f["start_frame"], f["end_frame"]
        sh = plan["shots"][shot_of[f0]]
        s0, s1 = sh["start_frame"], sh["end_frame"]
        e = int(ease_frames if ease_frames is not None else max(1, (f1 - f0) // 2))
        a, b = max(s0, f0 - e) - 0.5, min(s1 - 1, f1 - 1 + e) + 0.5  # half-frame pad: every in-window step moves
        span = _smootherstep((f1 - 1 - a) / (b - a)) - _smootherstep((f0 - a) / (b - a))
        total = CAM_TRAVEL * 1080 / max(span, 1e-6)
        idx = np.arange(s0, s1)
        dx[s0:s1] = total * (_smootherstep((idx - a) / (b - a)) - 0.5)
        zoom[s0:s1] = 1 + 2 * (np.abs(dx[s0:s1]).max() + drift_px + 4) / 1080
    return dx, zoom


def clock_rate(n, freezes, shot_of):
    """-> per frame, the rate the world clock runs at for the motion blur: 0 on every frozen frame (crisp, hanging),
    a short burst just after the release (longer streaks: the world lurches back into motion), 1 elsewhere."""
    vel = np.ones(n)
    for f in freezes:
        f0, f1 = f["start_frame"], f["end_frame"]
        vel[f0:f1] = 0.0
        for j, b in enumerate(BURST):
            i = f1 + j
            if i < n and shot_of[i] == shot_of[f1 - 1]:
                vel[i] = b
    return vel


def hit_profile(n, freezes):
    """-> (grade 0..1, punch scale, flash 0..1) per frame. The punch and flash land on the freeze downbeat and
    decay over 3 frames; the grade ramps in over 3 frames, holds, and snaps off on the release downbeat."""
    grade, punch, flash = np.zeros(n), np.ones(n), np.zeros(n)
    for f in freezes:
        f0, f1 = f["start_frame"], f["end_frame"]
        grade[f0:f1] = 1.0
        for j, g in enumerate(GRADE_IN):
            if f0 + j < f1:
                grade[f0 + j] = g
        for j, (p, w) in enumerate(zip(PUNCH, FLASH)):
            if f0 + j < min(n, f1):
                punch[f0 + j], flash[f0 + j] = p, w
    return grade, punch, flash


def apply_hit(frame, punch, flash):
    """The punch (scale about the frame centre) and the white flash, on a finished uint8 frame."""
    if punch == 1.0 and flash == 0.0:
        return frame
    out = frame
    if punch != 1.0:
        h, w = frame.shape[:2]
        M = np.float32([[punch, 0, (1 - punch) * w / 2], [0, punch, (1 - punch) * h / 2]])
        out = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if flash > 0:
        out = (out.astype(np.float32) + (255 - out.astype(np.float32)) * flash).clip(0, 255).astype(np.uint8)
    return out


def particle_steps(states):
    """Mean wrapped displacement of the particles between consecutive frames (world space), NaN at frame 0."""
    from realone.worlds import WRAP_PAD
    span = np.array([1080 + 2 * WRAP_PAD, 1920 + 2 * WRAP_PAD], np.float64)
    out = np.full(len(states), np.nan)
    for i in range(1, len(states)):
        a, b = states[i - 1], states[i]
        if a is None or b is None or len(a) == 0 or a.shape != b.shape:
            continue
        d = np.abs(b - a)
        d = np.minimum(d, span - d)
        out[i] = float(np.hypot(d[:, 0], d[:, 1]).mean())
    return out


def checks_bullet(keys, state_keys, pstep, cam, plate_steps, wf, freezes, shot_of, artist_moved, control_moved,
                  world_of=None, section_starts=()):
    """The `bullet` proof, per freeze, on what the render units actually fed and drew.

    keys          per-frame hash of the rendered world layer (graded plate + particles): with the camera moving it
                  must CHANGE on every frozen frame (the camera's move is in the pixels)
    state_keys    per-frame hash of the world-space state the units drew from: every particle position, the world
                  clock fed to the plate and the push fraction: IDENTICAL on every frozen frame
    pstep         per-frame mean particle displacement in world space (particle_steps)
    cam           per-frame camera dx the units used (px)
    plate_steps, artist_moved, control_moved: as in `checks` (control: the state one clock step on differs)
    world_of      per-frame world id (default: the shot index stands in for it)
    section_starts frames where a section starts (a release there may be a cut)
    The plan rule, proven on the plan that was rendered: no cut on the freeze downbeat (n/a only when the freeze is
    the render's first frame), and one world from the frame before the downbeat through the release frame.
    -> (report dict, ok)."""
    world_of = np.asarray(shot_of if world_of is None else world_of)
    n = len(keys)
    moving_bad = [i for i in range(1, n) if shot_of[i] == shot_of[i - 1] and wf[i] != wf[i - 1]
                  and (state_keys[i] == state_keys[i - 1] or keys[i] == keys[i - 1])]
    rep = {"style": "bullet", "world_moves_outside_freezes": not moving_bad,
           "frames_that_should_move_but_did_not": moving_bad[:20], "freezes": []}
    ok = not moving_bad
    for k, f in enumerate(freezes):
        f0, f1 = f["start_frame"], f["end_frame"]
        r = {"frames": f"{f0}-{f1 - 1}", "n_frames": f1 - f0, "bars": f"{f['start_bar']}-{f['end_bar']}",
             "role": f["role"]}
        if f0 == 0:
            r["no_cut_on_downbeat"] = None
            r["no_cut_note"] = "n/a: the freeze is the render's first frame (no lead-in; see lead_in)"
        else:
            r["no_cut_on_downbeat"] = bool(shot_of[f0] == shot_of[f0 - 1])
        last = f1 if f1 < n and f1 not in set(section_starts) else f1 - 1
        r["same_world_through"] = bool(len(set(world_of[max(0, f0 - 1):last + 1].tolist())) == 1
                                       and len(set(shot_of[max(0, f0 - 1):last + 1].tolist())) == 1)
        r["state_frozen_identical"] = f1 - f0 >= 2 and len(set(state_keys[f0:f1])) == 1
        r["frozen_identical"] = r["state_frozen_identical"]  # the name the director and the summary read
        csteps = np.diff(np.asarray(cam[f0:f1], np.float64))
        travel = float(cam[f1 - 1] - cam[f0])
        r["camera_travel_px"] = round(travel, 1)
        r["camera_moves"] = bool(len(csteps) and np.all(np.abs(csteps) > 1e-6) and abs(travel) >= 0.01 * 1080)
        r["rendered_world_moves_in_freeze"] = all(keys[i] != keys[i - 1] for i in range(f0 + 1, f1))
        before = [i for i in range(max(1, f0 - 30), f0 + 1) if shot_of[i] == shot_of[i - 1]]
        r["moves_before"] = all(state_keys[i] != state_keys[i - 1] for i in before) if before else None
        if not before:
            r["moves_before_note"] = "n/a: the freeze starts on a cut (the frames before are another shot)"
        if f1 >= n:
            r["resumes"], r["no_jump"] = None, None
            r["resume_note"] = "n/a: the freeze runs to the end of the render"
        elif shot_of[f1] != shot_of[f1 - 1]:
            r["resumes"], r["no_jump"] = None, None
            r["resume_note"] = "n/a: the release is a section cut (a new world)"
        else:
            nxt = [i for i in range(f1, min(n, f1 + 4)) if shot_of[i] == shot_of[f1]]
            r["resumes"] = all(state_keys[i] != state_keys[i - 1] and keys[i] != keys[i - 1] for i in nxt)
            after = [i for i in range(f1 + 1, min(n, f1 + 61)) if shot_of[i] == shot_of[f1]]
            pb = [pstep[i] for i in after if np.isfinite(pstep[i])]
            pmed = float(np.median(pb)) if pb else float("nan")
            cb = [abs(cam[i] - cam[i - 1]) for i in after]
            cmed = float(np.median(cb)) if cb else float("nan")
            cin = float(np.median(np.abs(csteps))) if len(csteps) else float("nan")
            cstep = abs(float(cam[f1] - cam[f1 - 1]))
            r["release_particle_step_px"] = round(float(pstep[f1]), 2) if np.isfinite(pstep[f1]) else None
            r["particle_step_median_after_px"] = round(pmed, 2) if pb else None
            r["release_camera_step_px"] = round(cstep, 3)
            r["camera_step_median_px"] = {"in_freeze": round(cin, 3), "after": round(cmed, 3) if cb else None}
            r["clock_step_at_release"] = float(wf[f1] - wf[f1 - 1])
            pstep_ok = bool(np.isfinite(pstep[f1]) and pb and pstep[f1] <= 3 * pmed + 0.5)
            cam_ok = bool(cb and cstep <= 3 * max(cmed, cin) + 0.5)
            step = plate_steps[f1]
            base = [plate_steps[i] for i in after if np.isfinite(plate_steps[i])]
            med = float(np.median(base)) if base else float("nan")
            # the camera moves the plate too: a jump is measured against the larger of its motion in the freeze
            # (camera only) and after the release (camera + clock)
            inside = [plate_steps[i] for i in range(f0 + 1, f1) if np.isfinite(plate_steps[i])]
            med_in = float(np.median(inside)) if inside else 0.0
            plate_ok = bool(np.isfinite(step) and base and step <= 3 * max(med, med_in) + 0.05)
            r["release_plate_step"] = round(float(step), 4) if np.isfinite(step) else None
            r["plate_step_median_after"] = round(med, 4) if base else None
            r["plate_step_median_in_freeze"] = round(med_in, 4)
            r["no_jump"] = bool(wf[f1] - wf[f1 - 1] == 1 and pstep_ok and cam_ok and plate_ok)
            r["no_jump_parts"] = {"clock": bool(wf[f1] - wf[f1 - 1] == 1), "particles": pstep_ok, "camera": cam_ok,
                                  "plate": plate_ok}
        r["artist_moves"] = bool(artist_moved[k])
        r["control_world_would_move"] = bool(control_moved[k])
        r["ok"] = bool(r["state_frozen_identical"] and r["camera_moves"] and r["rendered_world_moves_in_freeze"]
                       and r["no_cut_on_downbeat"] is not False and r["same_world_through"]
                       and r["artist_moves"] and r["control_world_would_move"]
                       and r["moves_before"] is not False and r["resumes"] is not False
                       and r["no_jump"] is not False)
        ok &= r["ok"]
        rep["freezes"].append(r)
    rep["ok"] = bool(ok)
    return rep, bool(ok)
