"""Stage camera: where the phone was, frame by frame, measured from the real ground it filmed.

The artist's own floor is tracked with the artist masked out (a planar track: every ground point between two
frames is related by one homography). The homography is split into the camera's rotation and its translation over
the ground's distance (H = K (R - t n^T / d) K^-1), and the ground's normal gives the horizon. With a camera height
in metres the translation is metric, and the generated world is rendered through that same camera
(realone/stage/world3d.py), so the floor under the artist's feet moves exactly as the floor he stood on.
Nothing here touches the artist's pixels."""
import cv2
import numpy as np

from realone.common import H, W, Refuse

LK = dict(winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
MIN_POINTS = 40       # fewer tracked ground points than this in a frame pair: the track is not trusted
MIN_INLIERS = 25
MAX_FB_ERR = 1.0      # forward-backward LK error, px
PERSON_GUARD_PX = 41  # the mask is dilated by this before features are picked (hair, motion blur, shadow edge)


def intrinsics(vfov_deg, w=W, h=H):
    """Pinhole K for a w x h frame with this VERTICAL field of view, principal point at the centre."""
    f = (h / 2) / np.tan(np.radians(vfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def iphone_main_vfov_916():
    """The iPhone main camera (26 mm full-frame equivalent) filmed upright at 9:16: the long side is the
    sensor's long side, 36 mm equivalent -> vertical FOV = 2 atan(18 / 26) ~ 69.4 deg."""
    return float(np.degrees(2 * np.arctan(18 / 26)))


def contact_point(mask):
    """Lowest mask rows -> (x, y) where the artist meets the floor (the median column of the bottom 6 rows)."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return (np.nan, np.nan)
    yb = ys.max()
    sel = ys >= yb - 6
    return (float(np.median(xs[sel])), float(yb))


def ground_track(frames, masks, ground_top=0.55):
    """frames: iterable of BGR uint8 (W x H); masks: callable i -> bool (H, W) artist mask.
    -> dict(H_to0 (n,3,3) frame-t ground -> frame-0 ground, step_inliers (n,), contact (n,2)).
    Refuses when the floor cannot be tracked (too few points or inliers in any frame pair) instead of guessing."""
    prev, acc, Hs, inl, contact = None, np.eye(3), [], [], []
    guard = np.ones((PERSON_GUARD_PX, PERSON_GUARD_PX), np.uint8)
    for i, fr in enumerate(frames):
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        m = masks(i)
        contact.append(contact_point(m))
        allow = np.zeros((H, W), np.uint8)
        allow[int(H * ground_top):, :] = 255
        allow[cv2.dilate(m.astype(np.uint8), guard) > 0] = 0
        if prev is None:
            inl.append(-1)
        else:
            p0 = cv2.goodFeaturesToTrack(prev[0], 1500, 0.005, 9, mask=prev[1])
            if p0 is None or len(p0) < MIN_POINTS:
                raise Refuse(f"stage track: frame {i}: only {0 if p0 is None else len(p0)} ground features "
                             f"(need {MIN_POINTS}); the floor has no texture to track")
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev[0], g, p0, None, **LK)
            pb, stb, _ = cv2.calcOpticalFlowPyrLK(g, prev[0], p1, None, **LK)
            ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm(p0 - pb, axis=2)[:, 0] < MAX_FB_ERR)
            if ok.sum() < MIN_POINTS:
                raise Refuse(f"stage track: frame {i}: {int(ok.sum())} ground points survived the "
                             f"forward-backward check (need {MIN_POINTS})")
            Hm, inm = cv2.findHomography(p1[ok], p0[ok], cv2.RANSAC, 2.0)  # frame i -> frame i-1
            if Hm is None or int(inm.sum()) < MIN_INLIERS:
                raise Refuse(f"stage track: frame {i}: no ground homography with {MIN_INLIERS}+ inliers")
            acc = acc @ Hm
            acc = acc / acc[2, 2]
            inl.append(int(inm.sum()))
        Hs.append(acc.copy())
        prev = (g, allow)
    if not Hs:
        raise Refuse("stage track: no frames")
    return dict(H_to0=np.array(Hs), step_inliers=np.array(inl), contact=np.array(contact))


def classify_motion(H_to0, K):
    """locked | rotation_only | translating, from how far the frame-centre ray and the ground moved.
    (shotcheck-style; used to say which shots needed the solve and to refuse a solve on a locked shot.)"""
    pts = np.float32([[[W * 0.5, H * 0.85]], [[W * 0.2, H * 0.95]], [[W * 0.8, H * 0.95]]])
    drift = max(float(np.abs(cv2.perspectiveTransform(pts, h) - pts).max()) for h in H_to0)
    if drift < 3:
        return "locked", drift
    return "moving", drift


def _pick(sols, prev_n):
    """decomposeHomographyMat returns up to 4 (R, t, n) in Malis' convention: X2 = R X1 + t, and the plane is
    n^T X1 = d with d > 0, so n points from the camera TO the plane. The floor is below the camera and camera y
    points down, so the floor's n has n_y > 0. Its (-t, -n) twin is the same picture with the camera moving the
    other way (it read a dolly back as a dolly in on the test plate), so the sign of n_y decides; then the
    solution closest to the previous frame's normal."""
    Rs, ts, ns = sols
    best, score = None, -1e9
    for R, t, n in zip(Rs, ts, ns):
        n = n[:, 0]
        if n[1] <= 0:
            continue
        s = n[1] + (3.0 * float(n @ prev_n) if prev_n is not None else 0.0)
        if s > score:
            best, score = (R, t[:, 0], n), s
    if best is None:
        raise Refuse("stage solve: no decomposition puts the floor below the camera")
    return best


MIN_PARALLAX = 0.02  # |t| / d below this, a frame pair says nothing about the floor's normal


def solve_camera(H_to0, K, height_m):
    """Ground homographies -> the floor's normal in camera 0 (and per-frame poses for the frames that moved).
    Each frame-i -> frame-0 homography is split into (R, t/d, n) (Malis' convention: view 1 = camera i); frames
    whose translation is under MIN_PARALLAX of the floor distance are skipped for the normal (a near-identity
    homography's normal is noise). Returns dict(R, t: X_cam_t = R X_cam_0 + t, identity where skipped;
    n0 (unit, pointing down to the floor: n0 . X = height for floor points); horizon_y; used (frames that voted))."""
    n_frames = len(H_to0)
    Rs, ts = [np.eye(3)], [np.zeros(3)]
    prev_n, n0s, used = None, [], []
    for i in range(1, n_frames):
        k, Rs_, ts_, ns_ = cv2.decomposeHomographyMat(H_to0[i], K)
        try:
            R_i0, t_over_d, n_i = _pick((Rs_, ts_, ns_), prev_n)
        except Refuse:
            Rs.append(np.eye(3))
            ts.append(np.zeros(3))
            continue
        if np.linalg.norm(t_over_d) >= MIN_PARALLAX:
            prev_n = n_i
            n0s.append(R_i0 @ n_i)  # camera i's normal into camera 0 (X_0 = R_i0 X_i + t)
            used.append(i)
        t_i0 = t_over_d * height_m
        Rs.append(R_i0.T)
        ts.append(-R_i0.T @ t_i0)
    if len(n0s) < 3:
        raise Refuse(f"stage solve: only {len(n0s)} frames moved the camera enough to see the floor's slope "
                     f"(need 3; a locked-off take needs --lens-mm and a level-camera assumption, not built yet)")
    n0 = np.median(np.array(n0s), axis=0)
    n0 /= np.linalg.norm(n0)
    l = np.linalg.inv(K).T @ n0
    horizon_y = float(-(l[0] * W / 2 + l[2]) / l[1])
    return dict(R=np.array(Rs), t=np.array(ts), n0=n0, horizon_y=horizon_y, used=used)


def _basis(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def pose_on_floor(H_to0_i, K, n0, height_m):
    """One frame's camera with the floor held fixed: frame 0 -> frame i maps floor points by
    K (R + t n0^T / d) K^-1. Directions along the floor go through R alone, so R is the Procrustes fit of the two
    floor directions (and their cross product, the normal) and t = d (s A n0 - R n0). -> (R, t, residual) where
    residual is how far A stays from a rigid motion (0 for a perfect lens and track)."""
    A = np.linalg.inv(K) @ np.linalg.inv(H_to0_i) @ K
    e1, e2 = _basis(n0)
    a1, a2 = A @ e1, A @ e2
    s = 2.0 / (np.linalg.norm(a1) + np.linalg.norm(a2))
    a1, a2 = a1 * s, a2 * s
    src = np.stack([e1, e2, n0], 1)
    dst = np.stack([a1, a2, np.cross(a1, a2) / max(np.linalg.norm(np.cross(a1, a2)), 1e-9)], 1)
    U, _, Vt = np.linalg.svd(dst @ src.T)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    t = height_m * (s * (A @ n0) - R @ n0)
    resid = float(np.linalg.norm(R @ e1 - a1) + np.linalg.norm(R @ e2 - a2))
    return R, t, resid


def floor_reproj_err(H_to0, K, R, t, n0, height_m, rows=(0.72, 0.85, 0.97), every=5):
    """Max and median px between where the solved camera puts floor points and where the track put them."""
    Ki = np.linalg.inv(K)
    pts = np.float32([[x * W, y * H] for y in rows for x in (0.15, 0.5, 0.85)])
    rays = (Ki @ np.c_[pts, np.ones(len(pts))].T).T
    X0 = rays * (height_m / (rays @ n0))[:, None]
    errs = []
    for i in range(0, len(H_to0), every):
        Xt = (R[i] @ X0.T).T + t[i]
        p = (K @ Xt.T).T
        p = p[:, :2] / p[:, 2:]
        q = cv2.perspectiveTransform(pts[None], np.linalg.inv(H_to0[i]))[0]
        errs.append(float(np.abs(p - q).max()))
    return max(errs), float(np.median(errs))


def solve_on_floor(H_to0, K, n0, height_m):
    R, t, res = zip(*(pose_on_floor(h, K, n0, height_m) for h in H_to0))
    return np.array(R), np.array(t), np.array(res)


def fit_lens_and_floor(H_to0, vfovs=np.arange(24.0, 80.0, 1.0)):
    """The lens is the one under which every tracked floor homography is closest to a rigid camera move over one
    fixed floor (a 1-D search; the iPhone's own lens is a good start but a stock or zoomed clip is not)."""
    best = None
    for vf in vfovs:
        K = intrinsics(vf)
        try:
            n0 = solve_camera(H_to0, K, 1.0)["n0"]
        except Refuse:
            continue
        _, _, res = solve_on_floor(H_to0, K, n0, 1.0)
        score = float(np.median(res))
        if best is None or score < best[0]:
            best = (score, float(vf), K, n0)
    if best is None:
        raise Refuse("stage solve: no lens gives a floor with an up-facing normal")
    return best


def height_from_figure(contact_y, top_y, horizon_y, figure_m=1.75):
    """Camera height from a standing figure on the floor: a figure_m tall person whose feet are at contact_y and
    head at top_y, with the horizon at horizon_y (camera height = figure_m * (feet - horizon) / (feet - head))."""
    if not (contact_y > horizon_y and contact_y > top_y):
        raise Refuse("stage solve: the artist's feet are not below the horizon; cannot size the camera height")
    return figure_m * (contact_y - horizon_y) / (contact_y - top_y)
