"""The tracked stage (realone/stage): camera solve against a synthetic floor filmed by a KNOWN camera, the
floor-coverage refusal, the depth padding, and the integration bounds. Keyless, no model download (the Core ML
depth model is only exercised by the lab, not here)."""
import cv2
import numpy as np
import pytest

from realone.common import H, W, Refuse
from realone.stage import camera as CAM
from realone.stage import integrate as IG
from realone.stage.depth import IN_H, IN_W, pad_to_model_aspect
from realone.stage.world3d import World3D, floor_horizon_row

HEIGHT = 1.5
PITCH = np.radians(9.0)
VFOV = 50.0


def _cam_pose(k, speed=0.05, sway=0.01):
    """Camera k: pitched down PITCH, dollying BACK `speed` m per frame with a small sideways sway. World frame:
    x right, y down, z forward; floor at y = HEIGHT. -> (R_wc, C) camera rotation (world->camera) and centre."""
    c, s = np.cos(PITCH), np.sin(PITCH)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])  # rotate about x: forward axis tips toward the floor
    C = np.array([sway * np.sin(k / 6), 0.0, -speed * k])
    return R, C


def _floor_frame(k, tex, K, mpp=0.004, origin=(-8.0, 0.5), speed=0.05, sway=0.01):
    """Render the floor texture (top-down, mpp metres per texel, texel (0,0) at world (origin x, origin z)) as
    camera k sees it: a plane-to-image homography from texture coords."""
    R, C = _cam_pose(k, speed, sway)
    # world point on the floor from texel (u, v): X = (ox + u mpp, HEIGHT, oz + v mpp)
    A = np.array([[mpp, 0, origin[0]], [0, 0, HEIGHT], [0, mpp, origin[1]]])
    Hm = K @ R @ (A - np.outer(C, [0, 0, 1.0]))  # x_cam = R (X - C), X = A [u, v, 1]
    img = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=(90, 90, 90))
    return img


@pytest.fixture(scope="module")
def synthetic_take():
    rng = np.random.default_rng(7)
    tex = (rng.random((2500, 4000)) * 255).astype(np.uint8)
    tex = cv2.GaussianBlur(tex, (0, 0), 1.2)
    tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    K = CAM.intrinsics(VFOV)
    frames = [_floor_frame(k, tex, K) for k in range(24)]
    return frames, K, tex


def _no_person(i):
    m = np.zeros((H, W), bool)
    m[1500:1900, 480:600] = True  # a "person" block standing on the floor: masked out of the track
    return m


def test_solve_recovers_a_dolly_back_and_the_horizon(synthetic_take):
    frames, K, _ = synthetic_take
    tr = CAM.ground_track(frames, _no_person, ground_top=0.45)
    s = CAM.solve_camera(tr["H_to0"], K, HEIGHT)
    true_horizon = H / 2 - K[1, 1] * np.tan(PITCH)
    assert abs(s["horizon_y"] - true_horizon) < 12, (s["horizon_y"], true_horizon)
    R, t, res = CAM.solve_on_floor(tr["H_to0"], K, s["n0"], HEIGHT)
    C_last = -R[-1].T @ t[-1]  # camera centre of the last frame, in camera-0 coordinates
    # 23 frames x 0.05 m back = 1.15 m, mostly along camera-0's -z (pitched 9 deg: a little along -y... +y)
    assert C_last[2] < -0.9, C_last          # it moved BACK (the twin solution reads a dolly IN)
    assert abs(np.linalg.norm(C_last) - 1.15) < 0.15, C_last
    mx, med = CAM.floor_reproj_err(tr["H_to0"], K, R, t, s["n0"], HEIGHT)
    assert med < 2.0 and mx < 5.0, (mx, med)


def test_track_refuses_a_floor_with_no_texture():
    flat = np.full((H, W, 3), 128, np.uint8)
    for x in range(100, 1000, 100):  # nine dots: some features, far too few to trust
        cv2.circle(flat, (x, 1400), 6, (20, 20, 20), -1)
    with pytest.raises(Refuse, match="ground features"):
        CAM.ground_track([flat, flat.copy(), flat.copy()], _no_person)


def test_solve_refuses_a_locked_off_take(synthetic_take):
    _, K, tex = synthetic_take
    # a tripod: the camera breathes half a millimetre a frame; any normal from that is noise
    still = [_floor_frame(k, tex, K, speed=0.0005, sway=0.0002) for k in range(6)]
    tr = CAM.ground_track(still, _no_person, ground_top=0.45)
    with pytest.raises(Refuse, match="moved the camera enough"):
        CAM.solve_camera(tr["H_to0"], K, HEIGHT)


def test_height_from_figure():
    # feet 400 px below the horizon, head 1200 px above the feet: camera at 1.75 * 400 / 1200
    assert CAM.height_from_figure(1700, 500, 1300) == pytest.approx(1.75 * 400 / 1200)
    with pytest.raises(Refuse):
        CAM.height_from_figure(1000, 500, 1300)  # feet above the horizon


def _synthetic_world(horizon_row=900, Sh=2048, Sw=1152):
    """A still whose lower part is a floor seen from a level-ish camera and whose top is far: disparity linear in
    the row below the horizon (what a flat floor gives), ~0 above."""
    v = np.arange(Sh, dtype=np.float32)[:, None]
    disp = np.clip((v - horizon_row) / (Sh - horizon_row), 0, None) * 0.9 + 0.02
    disp = np.repeat(disp, Sw, 1).astype(np.float32)
    rng = np.random.default_rng(3)
    still = (rng.random((Sh, Sw, 3)) * 255).astype(np.uint8)
    return still, disp


def test_floor_horizon_row_finds_the_floor_line():
    still, disp = _synthetic_world(horizon_row=900)
    vh, r = floor_horizon_row(disp)
    assert abs(vh - 900) < 6 and r > 0.99


def test_world_floor_is_pinned_and_backdrop_never_empty():
    still, disp = _synthetic_world()
    K = CAM.intrinsics(VFOV)
    n = np.array([0.0, np.cos(PITCH), np.sin(PITCH)])
    w = World3D(still, disp, K, HEIGHT, n, still_f=K[0, 0] * 0.8)
    yaw = np.radians(10)  # a pan past the still's edge: the backdrop must still fill the frame
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    img, zb, fv = w.render(Ry, np.array([0.0, 0.0, 0.3]))
    assert img.shape == (H, W, 3) and np.isfinite(img).all()
    assert (img.sum(2) > 0).mean() > 0.999          # the opaque backdrop: no empty pixel anywhere
    assert fv[int(H * 0.9)].mean() > 0.5            # the lower frame is floor (a shadow may fall there)
    assert fv[: int(H * 0.1)].mean() < 0.1          # the top is not


def test_depth_padding_round_trips():
    img = np.zeros((2048, 1152, 3), np.uint8)
    padded, (x0, y0, w, h) = pad_to_model_aspect(img)
    assert abs(padded.shape[1] / padded.shape[0] - IN_W / IN_H) < 0.01
    assert (w, h) == (1152, 2048) and y0 == 0 and x0 > 0


def test_ring_grade_is_bounded_and_keeps_a_colour_lean():
    m = np.zeros((H, W), bool)
    m[800:1400, 400:700] = True
    fg = np.full((H, W, 3), 0.5, np.float32)
    plate = np.full((H, W, 3), 0.9, np.float32)                  # bright neutral around him when filmed
    dark = np.full((H, W, 3), 0.05, np.float32)                  # near black around him now
    _, gain = IG.RingGrade(ema=1.0)(fg, m, plate, dark)
    lo = IG.LUM_LIMIT[0] * IG.CHROMA_LIMIT[0] - 1e-6
    hi = IG.LUM_LIMIT[1] * IG.CHROMA_LIMIT[1] + 1e-6
    assert all(lo <= x <= hi for x in gain)
    # a DARK, WARM world (dusk): the level is clamped, but the warm lean must survive the clamp (BGR: R > B)
    dusk = np.zeros((H, W, 3), np.float32)
    dusk[..., 2], dusk[..., 1], dusk[..., 0] = 0.30, 0.16, 0.07
    _, g2 = IG.RingGrade(ema=1.0)(fg, m, plate, dusk)
    assert g2[2] > g2[0] * 1.08, g2


def test_key_ramp_lights_the_side_facing_the_sun():
    m = np.zeros((H, W), bool)
    m[600:1700, 400:700] = True
    fg = np.full((H, W, 3), 0.5, np.float32)
    same = np.full((H, W, 3), 0.5, np.float32)
    out, _ = IG.RingGrade(ema=1.0)(fg, m, same, same, key=(0.9, (0.6, 0.8, 1.0)))  # shadow right: sun left
    assert out[1000, 410].mean() > out[1000, 690].mean() + 0.02
    assert out[1000, 410, 2] > out[1000, 410, 0]                  # the lit side takes the key's warm colour


def test_contact_shadow_stays_on_the_floor():
    m = np.zeros((H, W), bool)
    m[600:1700, 450:650] = True
    floor = np.zeros((H, W), np.float32)
    floor[1500:] = 1.0  # the drop shadow reaches up to ~row 1315: everything above the floor must stay clean
    s = IG.contact_shadow(m, floor)
    assert s[:1500].max() == 0.0
    assert s[1690:1720, 450:650].mean() > 0.1       # dark where the feet meet the floor
    assert s.max() <= 0.85
    # the drop shadow falls AWAY from the camera (up the frame) for light from behind it, never toward the lens
    tall = np.ones((H, W), np.float32)
    s2 = IG.contact_shadow(m, tall, ao=0.0, drop=0.5)
    assert s2[1350:1650, 450:650].mean() > 0.2 and s2[1760:, :].max() < 0.05


def test_clean_plate_takes_floor_out_of_the_matte_edge_and_keeps_the_artist():
    """An artist (a painted block) dancing sideways across a tracked floor, as in a take: the floor under any part
    of him is seen clear by some neighbour. His matte also holds a strip of real floor under his feet."""
    from realone.stage.cleanplate import CleanPlate
    rng = np.random.default_rng(11)
    tex = cv2.cvtColor(cv2.GaussianBlur((rng.random((2500, 4000)) * 255).astype(np.uint8), (0, 0), 1.2),
                       cv2.COLOR_GRAY2BGR)
    K = CAM.intrinsics(VFOV)
    frames = [_floor_frame(k, tex, K, speed=0.015) for k in range(60)]

    def parts(k):
        x = 300 + 8 * k                               # 8 px a frame sideways
        body = np.zeros((H, W), bool)
        body[1400:1700, x:x + 100] = True
        chip = np.zeros((H, W), bool)
        chip[1700:1714, x + 20:x + 80] = True         # real floor the matte wrongly holds, under his feet
        dark = np.zeros((H, W), bool)
        dark[1500:1690, x:x + 8] = True               # a near-black jacket edge
        stripe = np.zeros((H, W), bool)
        stripe[1650:1658, x + 3:x + 97] = True        # a stripe on him, 3 px inside his edge
        return body, chip, dark, stripe

    masks = []
    for k, f in enumerate(frames):
        body, chip, dark, stripe = parts(k)
        f[body] = (200, 60, 30)
        f[dark] = (5, 5, 5)
        masks.append(body | chip)
    tr = CAM.ground_track(frames, lambda i: masks[i], ground_top=0.45)
    cp = CleanPlate(tr["H_to0"])
    for i, f in enumerate(frames[:55]):  # streaming, as refine_masks does: frame 30 is refined once 54 is in
        cp.feed(i, f, masks[i])
    body, chip, dark, stripe = parts(30)
    clean, valid = cp.floor_at(30, (0, 0, W, H))
    assert valid[chip].mean() > 0.9 and valid[stripe].mean() > 0.9   # the case is real: neighbours saw that floor
    frames[30][stripe] = clean[stripe].astype(np.uint8)  # his stripe is floor-coloured, yet it is his
    out, removed = cp.refine(30, frames[30], masks[30], 1713)
    assert removed > 0.5 * chip.sum(), removed       # most of the floor strip leaves the matte
    assert out[body].all()                           # every artist pixel stays: the dark edge, the stripe, all of him
    assert not (out & ~masks[30]).any()              # nothing is ever added


def test_clean_plate_never_judges_where_no_neighbour_saw_the_floor():
    """Where the clean floor is unknown (every neighbour had the artist there), the unknown floor reads as black;
    a near-black jacket edge must not be mistaken for it."""
    from realone.stage.cleanplate import CleanPlate
    cp = CleanPlate(np.repeat(np.eye(3)[None], 3, 0))
    m = np.zeros((H, W), bool)
    m[1300:1700, 450:650] = True
    frame = np.full((H, W, 3), 6, np.uint8)                # a night take: dark floor
    frame[m] = (4, 4, 4)                                   # a black outfit
    roi_shape = lambda roi: (roi[3] - roi[1], roi[2] - roi[0])
    cp.floor_at = lambda i, roi: (np.zeros(roi_shape(roi) + (3,), np.float32), np.zeros(roi_shape(roi), bool))
    out, removed = cp.refine(1, frame, m, 1699)
    assert removed == 0 and out[m].all()


def test_planted_feet_get_a_dense_pocket_and_lifted_feet_do_not():
    m = np.zeros((H, W), bool)
    m[600:1700, 450:650] = True
    floor = np.ones((H, W), np.float32)
    lifted = IG.contact_shadow(m, floor, planted=0.0)
    planted = IG.contact_shadow(m, floor, planted=1.0)
    under = (slice(1706, 1726), slice(470, 630))
    assert planted[under].mean() > lifted[under].mean() + 0.2      # the pocket appears only when planted
    assert planted[:1690].max() <= lifted[:1690].max() + 1e-6       # and only under the sole (blur reaches ~6 px up)


def test_planted_is_read_from_the_tracked_floor_not_the_screen():
    n = 12
    # the camera moves: the floor slides 16 px a frame across the screen (a swing foot moves at least that fast)
    H_to0 = np.array([np.array([[1, 0, 0], [0, 1, -16.0 * i], [0, 0, 1]]) for i in range(n)])
    on_floor = np.array([[540.0, 1700.0 + 16.0 * i] for i in range(n)])  # the foot moves WITH the floor: planted
    p = IG.planted_from_track(on_floor, H_to0)
    assert p[2:-2].min() > 0.9
    on_screen = np.array([[540.0, 1700.0] for i in range(n)])            # still on screen = sliding on the floor
    # (a slide under ~6 px/frame reads as mostly planted: the contact point itself jitters about that much)
    assert IG.planted_from_track(on_screen, H_to0)[2:-2].max() < 0.1
