"""Hook Freeze: the world clock holds inside a freeze and resumes one step at a time; the world layer is a pure
function of it (identical bytes in a freeze, moving outside, no jump on the release); the plan and the camera
honour it; the checks fail when the world moves inside a freeze or jumps on the release."""
import numpy as np
import pytest

from realone import freeze as FZ
from realone.audio import pick_section
from realone.common import FPS, H, W, Refuse
from realone.plan import plan_range, plan_shots, shot_index
from realone.worlds import Particles, plate_neon_alley, plate_thumb, world_key, world_layer


def test_world_clock_holds_inside_the_freeze_and_steps_by_one():
    wf = FZ.world_clock(100, [{"start_frame": 10, "end_frame": 40}])
    steps = np.diff(wf)
    assert set(np.unique(steps)) <= {0.0, 1.0}
    assert np.all(wf[10:40] == wf[10])                      # every frozen frame shows the same clock
    assert np.all(steps[:9] == 1) and steps[9] == 1          # moving into the freeze
    assert wf[40] == wf[39] + 1                              # the first released frame is ONE step on
    assert np.all(steps[39:] == 1)
    assert wf[-1] == 99 - 29                                 # 29 steps were held, not skipped


@pytest.fixture(scope="module")
def plate():
    return plate_neon_alley().clip(0, 255).astype(np.uint8)


def _render_world(plate, plan, freezes, crops, push=0.04):
    n = plan["n_frames"]
    wf = FZ.world_clock(n, freezes)
    shot_of = shot_index(plan)
    crops = FZ.hold_camera(crops, freezes, shot_of)
    pf = FZ.push_fraction(wf, plan)
    parts = Particles("rain")
    keys, steps, prev = [], np.full(n, np.nan), None
    for i in range(n):
        w, fr, base = world_layer(plate, parts, crops[i], pf[i], push, 10.0 + wf[i] / FPS)
        keys.append(world_key(w, fr))
        th = plate_thumb(base)
        if prev is not None:
            steps[i] = np.abs(th - prev).mean()
        prev = th
    return keys, steps, wf, shot_of


def _tiny_plan(n, freeze):
    return {"n_frames": n, "shots": [{"index": 0, "start_frame": 0, "end_frame": n, "frames": n}],
            "freezes": [freeze]}


def test_world_layer_is_frozen_moves_outside_and_resumes_without_a_jump(plate):
    n, fz = 48, {"start_frame": 8, "end_frame": 24, "start_bar": 0, "end_bar": 1, "role": "hook"}
    plan = _tiny_plan(n, fz)
    crops = np.tile([0.0, 0.0, W, H], (n, 1))
    keys, steps, wf, shot_of = _render_world(plate, plan, [fz], crops)
    assert len(set(keys[8:24])) == 1                                   # frozen: identical bytes
    assert all(keys[i] != keys[i - 1] for i in range(1, 9))            # moving into it
    assert all(keys[i] != keys[i - 1] for i in range(24, n))           # resumes
    base = np.median(steps[26:])
    assert steps[24] <= 3 * base + 0.05, (steps[24], base)             # one step on, not a jump
    rep, ok = FZ.checks(keys, steps, wf, [fz], shot_of, [True], [True])
    assert ok, rep


def test_a_release_at_real_time_is_caught_as_a_jump(plate):
    """The bug the no-jump check exists for: after the freeze the clock snaps to real time."""
    n, fz = 48, {"start_frame": 8, "end_frame": 24, "start_bar": 0, "end_bar": 1, "role": "hook"}
    plan = _tiny_plan(n, fz)
    shot_of = shot_index(plan)
    wf = FZ.world_clock(n, [fz])
    wf[24:] = np.arange(24, n) * 6  # snaps forward (exaggerated so a 48-frame test sees what 2 bars would)
    parts = Particles("rain")
    keys, steps, prev = [], np.full(n, np.nan), None
    for i in range(n):
        w, fr, base = world_layer(plate, parts, (0, 0, W, H), wf[i] / n, 0.04, 10.0 + wf[i] / FPS)
        keys.append(world_key(w, fr))
        th = plate_thumb(base)
        if prev is not None:
            steps[i] = np.abs(th - prev).mean()
        prev = th
    rep, ok = FZ.checks(keys, steps, wf, [fz], shot_of, [True], [True])
    assert not ok and rep["freezes"][0]["no_jump"] is False, rep


def test_checks_fail_when_the_world_moves_inside_the_freeze():
    n = 30
    fz = {"start_frame": 5, "end_frame": 15, "start_bar": 0, "end_bar": 1, "role": "hook"}
    wf = FZ.world_clock(n, [fz])
    keys = [f"k{int(v)}" for v in wf]
    keys[9] = "moved"
    steps = np.full(n, 0.1)
    rep, ok = FZ.checks(keys, steps, wf, [fz], np.zeros(n, int), [True], [True])
    assert not ok and rep["freezes"][0]["frozen_identical"] is False
    keys = [f"k{int(v)}" for v in wf]
    keys[20] = keys[19]  # the world stood still outside the freeze
    rep, ok = FZ.checks(keys, steps, wf, [fz], np.zeros(n, int), [True], [True])
    assert not ok and rep["frames_that_should_move_but_did_not"] == [20]
    rep, ok = FZ.checks([f"k{int(v)}" for v in wf], steps, wf, [fz], np.zeros(n, int), [False], [True])
    assert not ok  # the artist did not move: that is a still image, not a freeze


def test_plan_freeze_starts_on_a_cut_and_drops_cuts_inside_and_on_the_release(timeline, song):
    sec = pick_section(timeline, "hook")
    fz = FZ.resolve(timeline, ["hook"], 2, [sec])
    assert [(f["start_bar"], f["end_bar"]) for f in fz] == [(12, 14)]
    plan = plan_shots(timeline, sec, cut_bars=1, n_worlds=3, freezes=fz)
    f = plan["freezes"][0]
    starts = [s["start_frame"] for s in plan["shots"]]
    assert f["start_frame"] == 0 and plan["shots"][0]["framing"] == "wide"
    assert not [s for s in starts if f["start_frame"] < s <= f["end_frame"]]  # no cut inside or on the release
    assert [s["bar"] for s in plan["shots"]] == [12, 15, 16, 17, 18, 19]
    bar_frames = song[2]["bar_s"] * FPS
    assert abs(f["frames"] - 2 * bar_frames) <= 1


def test_resolve_hook_means_every_hook_in_range_and_refuses_what_does_not_fit(timeline):
    secs = timeline["sections"]
    assert [f["start_bar"] for f in FZ.resolve(timeline, ["hook"], 2, secs)] == [12, 28]
    assert [f["start_bar"] for f in FZ.resolve(timeline, ["hook:2"], 2, secs)] == [28]
    assert [f["start_bar"] for f in FZ.resolve(timeline, ["section"], 4, secs)] == [12, 28]  # bare = every hook
    with pytest.raises(Refuse, match="longer than"):
        FZ.resolve(timeline, ["hook"], 9, secs)
    with pytest.raises(Refuse, match="does not start inside"):
        FZ.resolve(timeline, ["hook:2"], 2, [pick_section(timeline, "hook")])
    plan = plan_range(timeline, secs, 2, None, freezes=FZ.resolve(timeline, ["hook"], 2, secs))
    assert len(plan["freezes"]) == 2


def test_camera_holds_then_eases_back_without_a_jump():
    n = 120
    live = np.zeros((n, 4))
    live[:, 0] = np.linspace(0, 240, n)  # the artist walks right; the camera tracks
    live[:, 2], live[:, 3] = W / 1.9, H / 1.9
    fz = {"start_frame": 20, "end_frame": 60}
    out = FZ.hold_camera(live, [fz], np.zeros(n, int))
    assert np.all(out[20:60] == live[20])
    gap = live[60, 0] - live[20, 0]
    steps = np.abs(np.diff(out[:, 0]))
    assert steps[59] < 0.15 * gap                     # the release is a small step, not the whole gap
    e = int(round(FZ.EASE_S * FPS))
    assert np.allclose(out[60 + e:], live[60 + e:])
    assert steps.max() < 0.2 * gap                    # the ease never jumps (a jump is the whole gap at once)
