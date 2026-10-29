"""Hook Freeze v2 (`--freeze-style bullet`): the particles hang in world space on three depth layers while the
camera keeps moving; the freeze lands with a punch, a flash and a cool grade and releases with no jump. The checks
prove it on what was drawn: the world-space state identical through the freeze, the camera path and the rendered
world moving, the artist moving, no particle or camera jump on the release, an unfrozen control that moves."""
import hashlib

import numpy as np
import pytest

from realone import freeze as FZ
from realone.cli import build_parser
from realone.common import FPS, H, W
from realone.plan import shot_index
from realone.worlds import (DEPTH_LAYERS, DepthParticles, Particles, grade_world, make_particles, plate_neon_alley,
                            plate_thumb, world_key, world_layer)


@pytest.fixture(scope="module")
def plate():
    return plate_neon_alley().clip(0, 255).astype(np.uint8)


def _plan(n, f0, f1):
    fz = {"start_frame": f0, "end_frame": f1, "start_bar": 0, "end_bar": 1, "role": "hook"}
    return {"n_frames": n, "shots": [{"index": 0, "start_frame": 0, "end_frame": n, "frames": n}],
            "freezes": [fz]}, fz


def _skey(st, wt, pf):
    return hashlib.md5(st.tobytes() + np.float64([0, wt, pf]).tobytes()).hexdigest()


def _render(plate, plan, fz, state_clock=None, cam_override=None, draw_cam=True):
    """A miniature of the render unit: what it feeds world_layer, what it records, then checks_bullet."""
    n = plan["n_frames"]
    shot_of = shot_index(plan)
    wf = FZ.world_clock(n, [fz])
    dx, zoom = FZ.bullet_camera(n, plan, [fz], shot_of)
    if cam_override is not None:
        dx = cam_override
    vel = FZ.clock_rate(n, [fz], shot_of)
    grade, _, _ = FZ.hit_profile(n, [fz])
    pf = FZ.push_fraction(wf, plan)
    parts = DepthParticles("rain")
    keys, skeys, states, steps, prev = [], [], [], np.full(n, np.nan), None
    for i in range(n):
        wt = 10.0 + wf[i] / FPS
        pwt = 10.0 + state_clock[i] / FPS if state_clock is not None else wt
        cam = (float(dx[i]), float(vel[i]), float(zoom[i])) if (draw_cam or not vel[i] == 0) else (0.0, 0.0, float(zoom[i]))
        w, fr, base = world_layer(plate, parts, (0, 0, W, H), pf[i], 0.04, pwt, cam=cam)
        w, fr = grade_world(w, grade[i]), grade_world(fr, grade[i])
        keys.append(world_key(w, fr))
        st = parts.state(pwt)
        states.append(st.astype(np.float32))
        skeys.append(_skey(st, pwt, pf[i]))
        th = plate_thumb(base)
        if prev is not None:
            steps[i] = np.abs(th - prev).mean()
        prev = th
    ctl = [not np.array_equal(parts.state(10.0 + wf[fz["start_frame"]] / FPS),
                              parts.state(10.0 + (wf[fz["start_frame"]] + 1) / FPS))]
    return FZ.checks_bullet(keys, skeys, FZ.particle_steps(states), dx, steps, wf, [fz], shot_of, [True], ctl)


def test_three_depth_layers_scale_size_blur_brightness_speed_and_parallax_with_nearness():
    assert [L["name"] for L in DEPTH_LAYERS] == ["far", "mid", "near"]
    for k in ("size", "blur", "bright", "speed", "parallax"):
        v = [L[k] for L in DEPTH_LAYERS]
        assert v[0] < v[1] < v[2], (k, v)
    p = DepthParticles("rain")
    assert all(L["n"] > 0 for L in p.layers_) and p.n == sum(L["n"] for L in p.layers_)
    assert isinstance(make_particles("rain", "bullet"), DepthParticles)
    assert type(make_particles("rain", "still")) is Particles  # still keeps the 0.2.0 particles


def test_state_is_a_pure_function_of_the_world_clock_and_the_camera_only_moves_the_drawing():
    p = DepthParticles("rain")
    assert np.array_equal(p.state(12.0), DepthParticles("rain").state(12.0))
    assert not np.array_equal(p.state(12.0), p.state(12.0 + 1 / FPS))
    b0, f0 = p.layers(12.0, cam=0.0, vel=0.0)
    b1, f1 = p.layers(12.0, cam=30.0, vel=0.0)
    assert not np.array_equal(b0, b1) and not np.array_equal(f0, f1)   # the camera moved what is drawn
    assert np.array_equal(b0, p.layers(12.0, cam=0.0, vel=0.0)[0])     # and nothing else did


def test_moving_rain_is_streaks_and_frozen_rain_is_short_crisp_marks():
    p = DepthParticles("rain")
    mv, _ = p.layers(12.0, vel=1.0)
    fr, _ = p.layers(12.0, vel=0.0)
    lit_mv = int((mv.max(axis=2) > 0.08).sum())
    lit_fr = int((fr.max(axis=2) > 0.08).sum())
    assert lit_mv > 2.0 * lit_fr, (lit_mv, lit_fr)                     # a streak covers far more than a drop
    assert fr.max() > mv.max()                                         # the hanging drop is brighter, crisp


def test_bullet_freeze_passes_its_checks(plate):
    plan, fz = _plan(60, 0, 30)
    rep, ok = _render(plate, plan, fz)
    r = rep["freezes"][0]
    assert ok, rep
    assert r["state_frozen_identical"] and r["camera_moves"] and r["rendered_world_moves_in_freeze"]
    assert r["no_jump"] and r["resumes"] and abs(r["camera_travel_px"]) >= 0.03 * W


def test_particles_on_the_real_clock_inside_the_freeze_are_caught(plate):
    plan, fz = _plan(60, 0, 30)
    rep, ok = _render(plate, plan, fz, state_clock=np.arange(60, dtype=float))
    assert not ok and rep["freezes"][0]["state_frozen_identical"] is False


def test_a_held_camera_is_caught(plate):
    plan, fz = _plan(60, 0, 30)
    rep, ok = _render(plate, plan, fz, cam_override=np.zeros(60))
    r = rep["freezes"][0]
    assert not ok and r["camera_moves"] is False and r["rendered_world_moves_in_freeze"] is False


def test_a_camera_path_the_render_ignored_is_caught(plate):
    """The camera array moves but the frozen frames were drawn without it: the pixels must show the move."""
    plan, fz = _plan(60, 0, 30)
    rep, ok = _render(plate, plan, fz, draw_cam=False)
    r = rep["freezes"][0]
    assert not ok and r["camera_moves"] is True and r["rendered_world_moves_in_freeze"] is False


def test_particles_snapping_to_real_time_on_the_release_are_caught(plate):
    """The plate clock resumes one step on, but the particles are fed real time after the release."""
    plan, fz = _plan(60, 0, 30)
    sc = FZ.world_clock(60, [fz]).copy()
    sc[30:] = np.arange(30, 60)
    rep, ok = _render(plate, plan, fz, state_clock=sc)
    r = rep["freezes"][0]
    assert not ok and r["no_jump"] is False and r["no_jump_parts"]["particles"] is False
    assert r["no_jump_parts"]["clock"] is True


def test_a_camera_jump_on_the_release_is_caught(plate):
    plan, fz = _plan(60, 0, 30)
    dx, _ = FZ.bullet_camera(60, plan, [fz], shot_index(plan))
    dx = dx.copy()
    dx[30:] += 60
    rep, ok = _render(plate, plan, fz, cam_override=dx)
    r = rep["freezes"][0]
    assert not ok and r["no_jump_parts"]["camera"] is False


def test_bullet_checks_fail_on_a_still_artist_a_dead_control_and_a_world_still_outside():
    n = 40
    fz = {"start_frame": 5, "end_frame": 20, "start_bar": 0, "end_bar": 1, "role": "hook"}
    wf = FZ.world_clock(n, [fz])
    skeys = [f"s{int(v)}" for v in wf]
    keys = [f"k{i}" for i in range(n)]
    cam = np.arange(n) * 2.0
    pstep = np.full(n, 10.0)
    pstep[6:20] = 0.0
    steps = np.full(n, 0.1)
    so = np.zeros(n, int)
    rep, ok = FZ.checks_bullet(keys, skeys, pstep, cam, steps, wf, [fz], so, [True], [True])
    assert ok, rep
    assert not FZ.checks_bullet(keys, skeys, pstep, cam, steps, wf, [fz], so, [False], [True])[1]
    assert not FZ.checks_bullet(keys, skeys, pstep, cam, steps, wf, [fz], so, [True], [False])[1]
    sk = list(skeys)
    sk[30] = sk[29]
    rep, ok = FZ.checks_bullet(keys, sk, pstep, cam, steps, wf, [fz], so, [True], [True])
    assert not ok and rep["frames_that_should_move_but_did_not"] == [30]


def test_camera_truck_eases_in_and_out_moves_every_frozen_frame_and_its_zoom_hides_the_plate_edge():
    plan, fz = _plan(240, 60, 120)
    so = shot_index(plan)
    dx, zoom = FZ.bullet_camera(240, plan, [fz], so)
    st = np.diff(dx)
    assert abs((dx[119] - dx[60]) - FZ.CAM_TRAVEL * W) < 1e-6          # a few percent of the width per freeze
    assert np.all(st[60:119] > 0)                                       # moving on every frozen frame
    assert st[0] == 0 and st[-1] == 0 and st[59] < st[89] > st[120]     # eased: still, faster mid-freeze, still
    assert np.abs(np.diff(st)).max() < 0.05 * st.max()                  # no velocity step anywhere
    assert np.all(W * (zoom - 1) / 2 >= np.abs(dx).max() + 16)          # the margin covers the slide + the drift
    vel = FZ.clock_rate(240, [fz], so)
    assert np.all(vel[60:120] == 0) and list(vel[120:123]) == list(FZ.BURST) and np.all(vel[123:] == 1)


def test_the_hit_punch_flash_and_grade():
    g, p, f = FZ.hit_profile(100, [{"start_frame": 10, "end_frame": 50}])
    assert 1.03 <= p[10] <= 1.05 and p[9] == 1 and p[13] == 1 and np.all(np.diff(p[10:14]) < 0)
    assert 0.25 <= f[10] <= 0.35 and f[13] == 0 and np.all(np.diff(f[10:14]) < 0)
    assert g[9] == 0 and 0 < g[10] < g[11] < g[12] == 1 and np.all(g[12:50] == 1) and g[50] == 0
    img = np.full((64, 36, 3), 100, np.uint8)
    assert FZ.apply_hit(img, 1.0, 0.0) is img
    assert abs(float(FZ.apply_hit(img, 1.0, 0.3).mean()) - (100 + 155 * 0.3)) < 1.0
    col = np.zeros((4, 4, 3), np.float32)
    col[..., 2] = 200  # pure red (BGR)
    out = grade_world(col, 1.0)
    lum = 200 * 0.299
    assert abs((out[0, 0, 2] / 0.92 - lum) / (200 - lum) - 0.45) < 0.02  # saturation x0.45
    grey = grade_world(np.full((2, 2, 3), 120, np.float32), 1.0)
    assert grey[0, 0, 0] > 120 > grey[0, 0, 2]                         # and cooler: blue up, red down
    assert grade_world(col, 0.0) is col


def test_cli_freeze_style_defaults_to_bullet_and_a_bare_freeze_still_parses():
    p = build_parser()
    a = p.parse_args(["render", "--clip", "x.mp4", "--song", "y.wav", "--freeze"])
    assert a.freeze == ["section"] and a.freeze_style == "bullet"
    a = p.parse_args(["render", "--clip", "x.mp4", "--song", "y.wav", "--freeze", "--freeze-style", "still"])
    assert a.freeze_style == "still"


# --------------------------------------------------------------------------- the bullet plan rule + lead-in

def _cuts(plan):
    return [sh["bar"] for sh in plan["shots"]]


def test_bullet_plan_no_cut_on_the_freeze_downbeat_and_one_world_through_the_release(timeline):
    from realone.audio import pick_section
    from realone.plan import plan_range
    lead = pick_section(timeline, "bars:11-20")
    fzs = FZ.resolve(timeline, ["hook"], 2, [lead])
    still = plan_range(timeline, [lead], 2, None, freezes=fzs)
    bullet = plan_range(timeline, [lead], 2, None, freezes=fzs, freeze_style="bullet")
    assert _cuts(still) == [11, 12, 15, 17, 19]                         # still: unchanged, the freeze on a cut
    assert _cuts(bullet) == [11, 15, 17, 19]                            # bullet: falls from 11, hangs 12-14, resumes
    f = bullet["freezes"][0]
    so = shot_index(bullet)
    assert so[f["start_frame"] - 1] == so[f["start_frame"]] == so[f["end_frame"]]


def test_bullet_plan_on_a_section_boundary_keeps_the_previous_world_through_the_freeze(timeline):
    from realone.plan import plan_range
    secs = timeline["sections"]
    fzs = FZ.resolve(timeline, ["hook"], 2, secs)
    still = plan_range(timeline, secs, 2, None, freezes=fzs)
    bullet = plan_range(timeline, secs, 2, None, freezes=fzs, freeze_style="bullet")
    for sb in (12, 28):                                                 # both hooks start a section
        assert sb in _cuts(still) and sb not in _cuts(bullet) and sb + 2 not in _cuts(bullet)
    so = shot_index(bullet)
    for f in bullet["freezes"]:
        sh = bullet["shots"][so[f["start_frame"]]]
        assert sh["start_frame"] < f["start_frame"] and sh["end_frame"] > f["end_frame"]
        assert sh["role"] != "hook"                                     # the world running in is the one that hangs
    hooks = [sh for sh in bullet["shots"] if sh["role"] == "hook"]
    assert [sh["bar"] for sh in hooks] == [16, 18, 32, 34] and hooks[0]["framing"] == "wide"
    assert {s["start_bar"] for s in secs} - {12, 28} <= set(_cuts(bullet))  # every other section boundary stays


def test_bullet_freeze_on_the_first_frame_gets_a_lead_in_in_the_same_world(timeline):
    from realone.audio import pick_section
    from realone.plan import plan_range
    hook = pick_section(timeline, "hook")
    fzs = FZ.resolve(timeline, ["section"], 2, [hook])
    secs, info = FZ.lead_in(timeline, [hook], fzs, offset_s=0.0, max_seconds=40)
    assert info["bars"] == 2 and secs[0]["start_bar"] == 10 and secs[0]["role"] == "hook" and hook["start_bar"] == 12
    plan = plan_range(timeline, secs, 2, None, freezes=fzs, freeze_style="bullet")
    f = plan["freezes"][0]
    assert f["start_frame"] > 0 and _cuts(plan) == [10, 16, 18]
    so = shot_index(plan)
    assert so[0] == so[f["start_frame"]] == so[f["end_frame"]]          # lead-in, freeze, release: one shot
    # the take does not cover the lead-in, or the song has no room, or it would run long: as before, said why
    s2, i2 = FZ.lead_in(timeline, [hook], fzs, offset_s=timeline["downbeats"][11], max_seconds=40)
    assert s2[0] is hook and i2["bars"] == 0 and "take starts" in i2["why"]
    s3, i3 = FZ.lead_in(timeline, [hook], fzs, offset_s=0.0, max_seconds=20)
    assert i3["bars"] == 0 and "max-seconds" in i3["why"]
    assert FZ.lead_in(timeline, [hook], [], 0.0, 40) == ([hook], None)


def test_bullet_checks_catch_a_cut_on_the_downbeat_and_a_world_swap_through_the_freeze():
    n = 60
    fz = {"start_frame": 20, "end_frame": 40, "start_bar": 0, "end_bar": 1, "role": "hook"}
    wf = FZ.world_clock(n, [fz])
    skeys = [f"s{int(v)}" for v in wf]
    keys = [f"k{i}" for i in range(n)]
    cam = np.arange(n) * 2.0
    pstep = np.full(n, 10.0)
    pstep[21:40] = 0.0
    steps = np.full(n, 0.1)
    one = np.zeros(n, int)
    rep, ok = FZ.checks_bullet(keys, skeys, pstep, cam, steps, wf, [fz], one, [True], [True], world_of=one)
    assert ok and rep["freezes"][0]["no_cut_on_downbeat"] and rep["freezes"][0]["same_world_through"], rep
    cut = one.copy()
    cut[20:] = 1                                                        # a cut on the freeze downbeat
    rep, ok = FZ.checks_bullet(keys, skeys, pstep, cam, steps, wf, [fz], cut, [True], [True], world_of=cut)
    assert not ok and rep["freezes"][0]["no_cut_on_downbeat"] is False
    w = one.copy()
    w[30:] = 2                                                          # same shot, but the world swapped mid-freeze
    rep, ok = FZ.checks_bullet(keys, skeys, pstep, cam, steps, wf, [fz], one, [True], [True], world_of=w)
    assert not ok and rep["freezes"][0]["same_world_through"] is False
    rel = one.copy()
    rel[40:] = 1                                                        # a cut on the release (not a section start)
    rep, ok = FZ.checks_bullet(keys, skeys, pstep, cam, steps, wf, [fz], rel, [True], [True], world_of=rel)
    assert not ok and rep["freezes"][0]["same_world_through"] is False
    rep, ok = FZ.checks_bullet(keys, skeys, pstep, cam, steps, wf, [fz], rel, [True], [True], world_of=rel,
                               section_starts=[0, 40])
    assert rep["freezes"][0]["same_world_through"] is True              # a section boundary there is allowed
