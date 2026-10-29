"""Hook Freeze in the renderer (proven first in spike/freeze_test.py).

On a section's first downbeat the WORLD clock stops for `--freeze-bars` bars: the plate push-in, the drift, the
particles and the virtual camera (reframe and parallax) all hold, while the artist keeps moving at full rate and
the grain keeps animating (a frozen frame with live grain reads as a stopped world; with dead grain it reads as a
still image). The clock then resumes from where it stopped, one frame step at a time, so there is no jump.

Everything the world draws is a pure function of the world clock, so the freeze is provable after the render:
the world layer (plate + particles) is hashed every frame, and `checks()` fails the render when the hashes say
the world moved inside a freeze, stood still outside one, or jumped on the release.
"""
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
