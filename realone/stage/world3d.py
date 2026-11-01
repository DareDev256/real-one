"""Stage world: a generated world still rebuilt in depth and filmed through the artist's own camera.

The still is split the way a compositor would build it by hand:
  sky     everything past FAR_M; moves with the camera's rotation only (a plane at infinity)
  floor   the pixels that sit on the fitted floor plane; one plane-induced homography per frame, then pinned to
          the TRACKED floor of the take, so the world's floor moves exactly as the floor the artist stood on
  cards   everything else, in N_CARDS inverse-depth slices, each a camera-facing plane at its own distance:
          real parallax (near cards slide further than far ones). Each card carries an inpainted band behind its
          nearer neighbours, so a disocclusion shows plausible background, never a hole.
All layers are warped with cv2.warpPerspective (backward warps: no cracks), sky -> floor -> far cards -> near.
The still is treated as the view of the camera at the ANCHOR frame (the one furthest back), so every other frame
sees a subset of it; it is resampled to the plate's lens with MARGIN extra field of view for rotation.
Pure function of (still, disparity, camera, frame index): a Hook Freeze holds the frame index."""
import cv2
import numpy as np

from realone.common import H, W, Refuse

MARGIN = 1.12
N_CARDS = 8
FAR_M = 400.0
FLOOR_REL_TOL = 0.18   # a pixel is floor when its inverse depth is within 18% of the floor plane's
FILL_BAND = 48         # px of inpainted background each card carries behind nearer content
FEATHER = 1.5


def _rot_between(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def floor_horizon_row(disp, start=0.45):
    """Row where the still's floor would meet the sky: per-row median inverse depth over the lower rows is a line
    in the row (a flat floor seen by a pinhole camera); its zero, after removing the far offset, is the horizon."""
    h, w = disp.shape
    rows = np.arange(int(h * start), h, 6)
    med = np.array([np.median(disp[r, w // 4: 3 * w // 4]) for r in rows])
    b = float(np.percentile(disp, 1))
    a1, a0 = np.polyfit(rows, med - b, 1)
    if a1 <= 0:
        raise Refuse("stage world: the still's lower half does not get nearer toward the bottom (no floor)")
    r = float(np.corrcoef(rows, med)[0, 1])
    return -a0 / a1, r


class World3D:
    def __init__(self, still_bgr, disp, K, height_m, n_anchor, n_cards=N_CARDS, margin=MARGIN, still_f=None,
                 geometry_only=False):
        """still_bgr/disp: the world still and its relative inverse depth. K: the plate's lens. height_m: camera
        height. n_anchor: the floor normal (pointing to the floor) in the ANCHOR frame's camera.
        still_f: the still's own focal length in plate pixels (None: the plate's own lens). A generated still has
        no true lens; a wider one is chosen when the take's camera tilts past the still's edge (fit_still_lens)."""
        self.K = K
        self.d = float(height_m)
        Sw, Sh = int(round(W * margin)), int(round(H * margin))
        self.S = cv2.resize(still_bgr, (Sw, Sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        D = cv2.resize(disp, (Sw, Sh), interpolation=cv2.INTER_LINEAR)
        f = K[0, 0] if still_f is None else float(still_f)
        self.KS = np.array([[f, 0, Sw / 2], [0, f, Sh / 2], [0, 0, 1.0]])
        vh, self.floor_r = floor_horizon_row(D)
        theta = np.arctan2(Sh / 2 - vh, f)           # the still's camera pitch (down positive)
        self.nS = np.array([0.0, np.cos(theta), np.sin(theta)])
        self.horizon_still = vh
        # metric inverse depth: on the floor 1/z = (n . r) / d; disparity is affine in 1/z
        uu, vv = np.meshgrid(np.arange(Sw, dtype=np.float32), np.arange(Sh, dtype=np.float32))
        q = ((vv - Sh / 2) / f) * self.nS[1] + self.nS[2]          # n . K^-1 [u, v, 1] (n_x = 0)
        cand = (vv > vh + 0.06 * Sh) & (np.abs(uu - Sw / 2) < Sw * 0.3)
        A = np.c_[q[cand], np.ones(int(cand.sum()))]
        coef, *_ = np.linalg.lstsq(A, D[cand], rcond=None)
        for _ in range(3):  # trim the worst 20% (props standing on the floor) and refit
            res = np.abs(A @ coef - D[cand])
            keep = res < np.percentile(res, 80)
            coef, *_ = np.linalg.lstsq(A[keep], D[cand][keep], rcond=None)
        alpha, b = coef
        if alpha <= 0:
            raise Refuse("stage world: inverse depth does not grow toward the floor's near edge")
        zinv = np.maximum((D - b) / (alpha * self.d), 1.0 / FAR_M).astype(np.float32)
        zinv_floor = np.maximum(q / self.d, 1e-6)
        floor = (vv > vh + 2) & (np.abs(zinv - zinv_floor) < FLOOR_REL_TOL * zinv_floor)
        floor = cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        sky = (zinv <= 1.0 / FAR_M * 1.0001) & ~floor
        rest = ~floor & ~sky
        self.zinv, self.floor_mask, self.sky_mask = zinv, floor, sky
        # rotation from the still's camera into the anchor camera (floor normals agree, no yaw or roll added)
        self.R_al = _rot_between(self.nS, n_anchor)
        if geometry_only:
            self.layers = []
            return
        # layers: (rgb float32, alpha float32, plane normal in S coords or None for infinity, plane distance)
        self.layers = []
        # the backdrop: the whole still at infinity, opaque, so no frame ever shows an empty pixel
        self.layers.append((self.S, np.ones(self.S.shape[:2], np.float32), None, None))
        self.layers.append(self._layer(floor, self.nS, self.d, fill_from=None))
        if rest.any():
            zr = zinv[rest]
            edges = np.quantile(zr, np.linspace(0, 1, n_cards + 1))
            for k in range(n_cards):  # far (small zinv) to near
                lo, hi = edges[k], edges[k + 1] + (1e-9 if k == n_cards - 1 else 0)
                own = rest & (zinv >= lo) & (zinv < hi)
                if own.sum() < 50:
                    continue
                zc = 1.0 / float(np.median(zinv[own]))
                nearer = rest & (zinv >= hi)
                self.layers.append(self._layer(own, np.array([0, 0, 1.0]), zc, fill_from=nearer))
        self.card_count = len(self.layers) - 2

    def _layer(self, own, normal, dist, fill_from):
        alpha = own.astype(np.float32)
        rgb = self.S
        if fill_from is not None and fill_from.any():
            band = cv2.dilate(own.astype(np.uint8), np.ones((2 * FILL_BAND + 1,) * 2, np.uint8)) > 0
            hole = band & fill_from
            if hole.any():
                small = cv2.resize(self.S, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).clip(0, 255)
                hm = cv2.resize(hole.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
                filled = cv2.inpaint(small.astype(np.uint8), hm * 255, 6, cv2.INPAINT_TELEA).astype(np.float32)
                filled = cv2.resize(filled, (self.S.shape[1], self.S.shape[0]), interpolation=cv2.INTER_LINEAR)
                rgb = np.where(hole[..., None], filled, self.S)
                alpha = np.maximum(alpha, hole.astype(np.float32))
        if FEATHER > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), FEATHER)
        return (rgb, alpha, normal, dist)

    # ------------------------------------------------------------------ per frame

    def plane_h(self, R, t, normal, dist):
        """Homography still -> frame for the plane normal . X_S = dist under X_frame = R X_S + t (None: infinity)."""
        M = R if normal is None else R + np.outer(t, normal) / dist
        return self.K @ M @ np.linalg.inv(self.KS)

    def render(self, R_t, t_t, correction=np.eye(3), virtual=None):
        """R_t, t_t: the frame's camera relative to the ANCHOR camera (X_t = R_t X_anchor + t_t).
        correction: a homography applied after every layer (pins the floor to the tracked floor).
        -> (rgb float32 0..255 (H, W, 3), zinv float32 (H, W) the world's inverse depth seen from this frame,
        floor float32 (H, W) how much of each pixel is visible floor: where a shadow may fall)."""
        R = R_t @ self.R_al
        out = np.zeros((H, W, 3), np.float32)
        zbuf = np.zeros((H, W), np.float32)
        floor_vis = np.zeros((H, W), np.float32)
        if virtual is not None:  # a camera move on top of the phone's own: (t_v, f_scale) -> X_v = X + t_v, f' = f*s
            t_v, fs = virtual
            Kv = self.K.copy()
            Kv[0, 0] *= fs
            Kv[1, 1] *= fs
            Z = Kv @ np.linalg.inv(self.K)
            correction = Z @ correction @ np.linalg.inv(Z)
        for k, (rgb, alpha, normal, dist) in enumerate(self.layers):
            if virtual is None:
                Hm = correction @ self.plane_h(R, t_t, normal, dist)
            else:
                M = R if normal is None else R + np.outer(t_t, normal) / dist
                if normal is not None:
                    M = M + np.outer(np.asarray(t_v, float), normal) / dist
                Hm = correction @ Kv @ M @ np.linalg.inv(self.KS)
            border = cv2.BORDER_REPLICATE if k == 0 else cv2.BORDER_CONSTANT  # the backdrop is never empty
            a = cv2.warpPerspective(alpha, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=border)
            if not a.any():
                continue
            c = cv2.warpPerspective(rgb, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            out = out * (1 - a[..., None]) + c * a[..., None]
            zl = cv2.warpPerspective(self.zinv, Hm, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
            zbuf = zbuf * (1 - a) + zl * a
            floor_vis = a if k == 1 else floor_vis * (1 - a)
        return out, zbuf, floor_vis

    def floor_h(self, R_t, t_t):
        return self.plane_h(R_t @ self.R_al, t_t, self.nS, self.d)

    def sky_on_top(self, band=0.06):
        """True when the still's top rows are open sky: the farthest thing in the still AND nearly textureless.
        Mirroring those rows past the top edge is invisible, so a camera that sees above the still's top is
        allowed there and nowhere else (a skyline or a ceiling up there is refused, never mirrored)."""
        rows = max(1, int(self.S.shape[0] * band))
        far = float(np.median(self.zinv[:rows])) <= float(np.percentile(self.zinv, 8)) * 1.25
        g = cv2.cvtColor(np.clip(self.S[:rows], 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        smooth = float(np.median(np.abs(cv2.Laplacian(g, cv2.CV_32F)))) < 2.0
        return far and smooth

    def uncovered(self, Hm, inset=4, lower=False, sky_ok=False):
        """Share of the frame's border samples that land outside the still under still->frame homography Hm.
        lower=True: only the bottom edge and the lowest quarter of the sides (where a floor must be).
        sky_ok=True: samples past the still's TOP edge do not count (open sky, mirrored at render)."""
        Sh, Sw = self.S.shape[:2]
        xs = np.linspace(0, W - 1, 24)
        ys = np.linspace(0.75 * H if lower else 0, H - 1, 40)
        pts = np.float32(([] if lower else [[x, 0] for x in xs]) + [[x, H - 1] for x in xs] +
                         [[0, y] for y in ys] + [[W - 1, y] for y in ys])
        q = cv2.perspectiveTransform(pts[None], np.linalg.inv(Hm))[0]
        out = (q[:, 0] < inset) | (q[:, 0] > Sw - 1 - inset) | (q[:, 1] > Sh - 1 - inset)
        if not sky_ok:
            out |= q[:, 1] < inset
        else:  # past the top is fine, but the sides of that region must still be inside the still
            out &= ~(q[:, 1] < inset) | (q[:, 0] < inset) | (q[:, 0] > Sw - 1 - inset)
        return float(out.mean())


def stage_corrections(world, poses, floor_exact):
    """Per frame: (R, t) relative to the anchor and the correction that pins the floor to the tracked floor.
    poses: [(R_t, t_t)] relative to the anchor; floor_exact: [still->frame floor homography that the track
    says]. -> list of correction homographies."""
    return [fe @ np.linalg.inv(world.floor_h(R, t)) for (R, t), fe in zip(poses, floor_exact)]


def fit_still_lens(still_bgr, disp, K, height_m, n_anchor, poses, floor_exact_fn, every=6, step=0.94, tries=14):
    """The longest still lens (closest to the plate's own) under which every sampled frame's floor and backdrop
    stay inside the still. floor_exact_fn(world, i) -> the tracked still->frame floor homography for frame i.
    -> (still_f, report). Refuses rather than render a frame with no world under it."""
    f = K[0, 0]
    idx = list(range(0, len(poses), every)) + [len(poses) - 1]
    for k in range(tries):
        g = World3D(still_bgr, disp, K, height_m, n_anchor, still_f=f, geometry_only=True)
        sky = g.sky_on_top()
        worst = 0.0
        for i in idx:
            R, t = poses[i]
            fe = floor_exact_fn(g, i)
            corr = fe @ np.linalg.inv(g.floor_h(R, t))
            worst = max(worst, g.uncovered(corr @ g.plane_h(R @ g.R_al, t, None, None), sky_ok=sky),
                        g.uncovered(fe, lower=True))
        if worst == 0.0:
            return f, dict(still_f=f, plate_f=float(K[0, 0]), tries=k + 1)
        f *= step
    raise Refuse(f"stage world: the take's camera leaves the still even at a {f / K[0, 0]:.2f}x wider lens")
