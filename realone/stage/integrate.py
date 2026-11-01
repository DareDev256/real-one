"""Stage integration: what makes a filmed artist sit IN a generated world rather than on top of it.

Harvested techniques (re-implemented, credited in the README), all applied to the WORLD except the ring grade and
the light wrap, which are the only operations that touch the artist's filmed pixels (the Receipt counts those
pixels as filmed + graded, never generated):
  contact shadow   ambient occlusion where his body meets the floor plus a drop shadow cast along the world's key
                   light, both on the floor only (pickups' shadow pass, synthesised instead of lifted)
  soft match       the world blurred until its fine detail matches the take's (a crisp still behind a phone-sharp
                   person is the first tell); measured, not guessed (platelock's fidelity idea)
  depth defocus    thin-lens circle of confusion from the world's depth, focused at his feet (hero-frame-rig)
  ring grade       his colour moved by a bounded fraction of how the light AROUND him changed, plate ring vs world
                   ring, smoothed over time (pickups' envblend)
  grain match      the take's own sensor noise measured and given to the world (platelock's estimate_grain)"""
import cv2
import numpy as np

from realone.common import H, W

RING_IN, RING_OUT = 14, 70
GRADE_STRENGTH = 0.6


def ring(mask, r_in=RING_IN, r_out=RING_OUT):
    m = mask.astype(np.uint8)
    a = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_out + 1,) * 2))
    b = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_in + 1,) * 2))
    return (a > 0) & ~(b > 0)


def _lin(x):
    return np.power(np.clip(x, 0, 1), 2.2)


LUM_LIMIT = (0.70, 1.20)     # how far the light level on him may move
CHROMA_LIMIT = (0.90, 1.12)  # how far each channel may lean (a warm dusk, a cool night), relative to the level
KEY_RAMP = 0.14              # brightness difference across his body toward the world's key light, each side


def _luma(rgb_lin):
    return rgb_lin @ np.array([0.0722, 0.7152, 0.2126])  # BGR order


class RingGrade:
    """The artist's grade from how the light AROUND him changed: the take's ring vs the world's ring, in linear
    light, split into a level (luminance) and a colour lean (chroma) so a dark warm world can still warm him,
    each bounded, EMA over time. key=(dx, colour): a ramp across his body toward the world's key light (dx < 0:
    light from screen left), warmed by the key's own colour on the lit side. Filmed pixels, graded, never
    generated; returns the per-channel mean gain it applied, for the Receipt."""

    def __init__(self, strength=GRADE_STRENGTH, ema=0.12):
        self.k, self.a, self.g = strength, ema, None

    def __call__(self, fg01, mask, plate01, world01, key=None):
        r = ring(mask)
        if r.sum() < 500:
            return fg01, (1.0, 1.0, 1.0) if self.g is None else tuple(self.g)
        p = _lin(plate01[r]).mean(0) + 1e-4
        w = _lin(world01[r]).mean(0) + 1e-4
        lp, lw = float(_luma(p)), float(_luma(w))
        lum = float(np.clip((lw / lp) ** self.k, *LUM_LIMIT))
        chroma = np.clip(((w / lw) / (p / lp)) ** self.k, *CHROMA_LIMIT)
        g = lum * chroma
        self.g = g if self.g is None else (1 - self.a) * self.g + self.a * g
        lin = _lin(fg01) * self.g
        if key is not None and mask.any():
            dx, kcol = key
            xs = np.nonzero(mask.any(0))[0]
            x0, x1 = xs.min(), xs.max()
            half = max(1.0, (x1 - x0) / 2)
            u = np.clip((np.arange(W, dtype=np.float32) - (x0 + x1) / 2) / half, -1, 1)  # -1 left .. 1 right
            side = -np.sign(dx) if abs(dx) > 0.05 else 0.0   # shadow goes right (dx > 0) -> light from the left
            ramp = 1 + KEY_RAMP * side * u                    # brighter toward the light
            kc = np.asarray(kcol, np.float32)
            kc = kc / max(float(kc.mean()), 1e-4)
            warm = 1 + 0.5 * (np.clip(kc, 0.8, 1.25) - 1)[None, None, :] * np.clip(side * u, 0, 1)[None, :, None]
            lin = lin * ramp[None, :, None] * warm
        out = np.power(np.clip(lin, 0, 1), 1 / 2.2)
        return out, tuple(float(x) for x in self.g)


def sharpness(gray01, region):
    """Fine-detail energy: the mean absolute Laplacian over region, top 2% trimmed (a few hard edges).
    Not the median: on 8-bit pixels the median moves in 1/255 steps and reads two different focuses as equal."""
    lap = cv2.Laplacian(gray01.astype(np.float32), cv2.CV_32F, ksize=3)
    v = np.abs(lap[region])
    if not v.size:
        return 0.0
    return float(np.mean(v[v <= np.percentile(v, 98)]))


def soft_match_sigma(plate_gray01, plate_region, world_gray01, world_region, sigmas=np.arange(0, 4.01, 0.25)):
    """The Gaussian sigma that brings the world's fine detail down to the take's (0 when the world is already
    softer). Measured on the take's own background, so a sharp 4K take keeps a sharp world."""
    target = sharpness(plate_gray01, plate_region)
    best = 0.0
    for s in sigmas:
        g = world_gray01 if s == 0 else cv2.GaussianBlur(world_gray01, (0, 0), float(s))
        best = float(s)
        if sharpness(g, world_region) <= target:
            break
    return best, target


def depth_defocus(world, zinv, z_focus, coc_px_per_dioptre=11.0, max_coc=6.0, n_layers=6):
    """Thin-lens blur by inverse-depth layer: blur radius = k |1/z - 1/z_focus| px (k from the phone lens,
    ~ f_px x aperture), composited by layer weight. Only the world is blurred."""
    coc = np.clip(coc_px_per_dioptre * np.abs(zinv - 1.0 / z_focus), 0, max_coc)
    edges = np.linspace(0, max_coc, n_layers)
    out = np.zeros_like(world)
    wsum = np.zeros(world.shape[:2], np.float32)
    for k, c in enumerate(edges):
        w = np.clip(1 - np.abs(coc - c) / (edges[1] - edges[0] + 1e-6), 0, 1)
        if not w.any():
            continue
        img = world if c < 0.3 else cv2.GaussianBlur(world, (0, 0), float(c) * 0.6)
        out += img * w[..., None]
        wsum += w
    return out / np.maximum(wsum, 1e-6)[..., None]


def estimate_grain(gray01, region):
    """Sensor noise sigma from the MAD of a high-pass on the take's flat background (platelock)."""
    hp = gray01 - cv2.GaussianBlur(gray01, (0, 0), 1.2)
    v = hp[region]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else 0.0


def add_grain(world255, sigma01, seed):
    rng = np.random.default_rng(seed)
    n = rng.normal(0, sigma01 * 255, world255.shape[:2]).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.6)
    return world255 + n[..., None]


def planted_from_track(contact, H_to0, still_px=4.0, moving_px=14.0):
    """-> float (n,) 0..1: how planted the lowest foot is, from how fast its contact point moves ON THE TRACKED
    FLOOR (frame-0 floor px per frame; a planted sole is still there while the camera moves). 1 below still_px,
    0 above moving_px, smooth between, lightly smoothed over 3 frames. A slide slower than ~6 px/frame reads as
    mostly planted: the lowest matte pixel jitters about that much, so slow foot slides are not told apart."""
    n = len(contact)
    fl = np.full((n, 2), np.nan)
    for i in range(n):
        if np.isfinite(contact[i][0]):
            fl[i] = cv2.perspectiveTransform(np.float32([[contact[i]]]), H_to0[i])[0, 0]
    v = np.full(n, moving_px)
    v[1:] = np.linalg.norm(np.diff(fl, axis=0), axis=1)
    v[~np.isfinite(v)] = moving_px
    p = np.clip((moving_px - v) / (moving_px - still_px), 0, 1)
    p = p * p * (3 - 2 * p)
    return np.convolve(np.pad(p, 1, mode="edge"), np.ones(3) / 3, mode="valid")


def contact_pocket(mask, planted, depth_px=None, strength=0.5):
    """The dense dark pocket right under a planted sole (gauntlet r4: the real plate's floor under a planted sole
    is about half as bright as beside it; it fades as the heel lifts). Drawn under the lowest foot's columns only,
    `planted` 0..1 scales it. -> float32 (H, W) 0..strength."""
    ys, xs = np.nonzero(mask)
    out = np.zeros((H, W), np.float32)
    if not len(ys) or planted <= 0.01:
        return out
    yb, hgt = ys.max(), max(1, ys.max() - ys.min())
    depth = int(depth_px or max(10, 0.035 * hgt))
    low = ys >= yb - max(6, int(0.02 * hgt))
    for x in np.unique(xs[low]):
        col = np.nonzero(mask[:, x])[0]
        sy = int(col.max())
        y1 = min(H, sy + depth)
        if y1 > sy + 1:
            out[sy + 1:y1, x] = np.linspace(1.0, 0.0, y1 - sy - 1, dtype=np.float32) ** 0.6
    out = cv2.GaussianBlur(out, (0, 0), 2.5)
    return np.clip(out * 1.6, 0, 1) * strength * float(planted)


def contact_shadow(mask, floor_frame, light_dir=(0.0, 1.0), length=0.35, ao=0.2, drop=0.35, planted=0.0):
    """-> float32 (H, W) shadow amount 0..1 on the floor only.
    ao: occlusion right where the body meets the floor (the lowest mask pixels of every column, a squashed
    copy of the body's footprint). drop: the body flattened onto the floor along light_dir (image-space (dx, dy)
    of the shadow's direction, (0, 1) = straight away from the camera, up the frame), `length` x body height."""
    m = mask.astype(np.float32)
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return np.zeros((H, W), np.float32)
    yb, yt = ys.max(), ys.min()
    hgt = max(1, yb - yt)
    # ambient occlusion: the bottom 6% of the body squashed to a thin band at the floor line
    low = m.copy()
    low[: int(yb - 0.06 * hgt)] = 0
    a = cv2.GaussianBlur(low, (0, 0), 0.025 * hgt)
    a = cv2.GaussianBlur(np.maximum(a, cv2.GaussianBlur(low, (0, 0), 0.008 * hgt)), (0, 0), 3)
    a = np.clip(a * 2.2, 0, 1)
    # drop shadow: scale the mask about the floor line by -length vertically (flipped up the frame), lean by dx
    dx, dy = light_dir
    # a body point `up` px above the floor line lands `length * up` px further from the camera (up the frame by dy,
    # sideways by dx): y' = yb - L dy (yb - y), x' = x + L dx (yb - y)
    M = np.float32([[1, -dx * length, dx * length * yb], [0, length * dy, yb * (1 - length * dy)]])
    d = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    d = cv2.GaussianBlur(d, (0, 0), 0.02 * hgt)
    s = np.clip(ao * a + drop * d + contact_pocket(mask, planted), 0, 0.85)
    return s * floor_frame


# ---------------------------------------------------------------- round 2: measured depth of field, motion blur, sun

BG_BAND = (0.15, 0.45)  # rows behind and around the artist, above the floor: where the eye compares focus


def focus_ratio(gray01, mask, band=BG_BAND):
    """Background fine detail over the artist's own fine detail (his interior, eroded off the matte edge).
    The take's own ratio says how far out of focus its background was; a composite should read the same."""
    m = mask.astype(np.uint8)
    inner = cv2.erode(m, np.ones((9, 9), np.uint8)) > 0
    bgm = ~(cv2.dilate(m, np.ones((41, 41), np.uint8)) > 0)
    bgm[: int(H * band[0])] = False
    bgm[int(H * band[1]):] = False
    s_art = sharpness(gray01, inner)
    return (sharpness(gray01, bgm) / s_art) if s_art > 1e-6 else float("nan")


def calibrate_defocus(samples, ks=(0, 4, 8, 14, 22, 32, 45, 60, 80, 110)):
    """samples: [(plate_gray01, mask, world_bgr255, zinv, z_focus)]. -> (k, plate ratio, composite ratio at k).
    k is the thin-lens blur in px per dioptre of depth from focus (depth_defocus), chosen so the composite's
    background/performer focus ratio matches the take's on the same frames: measured, never assumed."""
    target, got = [], {k: [] for k in ks}
    for pg, m, w, z, zf in samples:
        rp = focus_ratio(pg, m)
        if not np.isfinite(rp):
            continue
        target.append(rp)
        a = (cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0))[..., None]
        for k in ks:
            wd = depth_defocus(w, z, zf, coc_px_per_dioptre=k, max_coc=14.0, n_layers=8) if k else w
            comp = pg[..., None] * 255 * a + wd * (1 - a)
            cg = cv2.cvtColor(np.clip(comp, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            got[k].append(focus_ratio(cg, m))
    if not target:
        return 0, float("nan"), float("nan")
    t = float(np.median(target))
    best = min(ks, key=lambda k: abs(float(np.nanmedian(got[k])) - t))
    return int(best), t, float(np.nanmedian(got[best]))


def motion_blur(world, G, shutter=0.5, samples=5):
    """Blur the world along the take's own floor motion: G maps the previous frame's floor to this frame's.
    The exposure spans `shutter` of a frame interval centred on the frame; sub-frame views are I + d (G - I)."""
    if G is None or np.allclose(G, np.eye(3), atol=1e-4):
        return world
    acc = np.zeros_like(world)
    for d in np.linspace(-shutter / 2, shutter / 2, samples):
        Gd = np.eye(3) + d * (G - np.eye(3))
        acc += cv2.warpPerspective(world, Gd, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / samples


def sun_from_still(still_bgr, horizon_row, min_contrast=1.35):
    """The world's key light read off the still: the brightest compact patch of sky above the horizon.
    -> (shadow direction (dx, dy) in image space, length, drop strength). A sun in the frame is IN FRONT of the
    camera, so shadows come TOWARD the lens (dy < 0, down the frame) and away from the sun sideways; the lower the
    sun, the longer the shadow. No clear sun (overcast, night): occlusion only, a faint drop straight back."""
    h, w = still_bgr.shape[:2]
    hr = int(np.clip(horizon_row, 8, h - 1))
    sky = cv2.cvtColor(still_bgr[:hr], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if sky.size == 0:
        return (0.0, 1.0), 0.2, 0.12
    blur = cv2.GaussianBlur(sky, (0, 0), max(4, w / 40))
    y, x = np.unravel_index(int(np.argmax(blur)), blur.shape)
    if blur[y, x] < min_contrast * float(np.median(blur)):
        return (0.0, 1.0), 0.2, 0.12
    dx = float(np.clip((w / 2 - x) / (w / 2), -1, 1))
    elev = float(np.clip((hr - y) / hr, 0, 1))
    return (dx * 0.9, -1.0), float(np.clip(0.9 - 0.6 * elev, 0.3, 0.9)), 0.4


def key_colour(still_bgr, horizon_row):
    """Mean colour (BGR, 0..1) of the brightest 2% of the sky: what colour the world's key light is."""
    hr = int(np.clip(horizon_row, 8, still_bgr.shape[0] - 1))
    sky = still_bgr[:hr].reshape(-1, 3).astype(np.float32) / 255
    if not len(sky):
        return (1.0, 1.0, 1.0)
    lum = sky.mean(1)
    top = sky[lum >= np.percentile(lum, 98)]
    return tuple(float(x) for x in top.mean(0))
