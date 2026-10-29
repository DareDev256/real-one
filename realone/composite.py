"""Per-frame compositing: reframe, edge despill, colour match, light wrap, grain, and the three looks.

The artist's pixels are the phone's pixels. Nothing here generates or repaints the artist: colour match is a
gentle global grade toward the world, despill only touches the edge band and small enclosed pockets of the
source background, and every look keeps the filmed person filmed.
"""
import cv2
import numpy as np

from realone.common import FPS, H, W

LOOKS = ("clean", "collage", "receipt")
BOIL_FPS = 12
PAPER = np.array([0.955, 0.955, 0.945], np.float32)  # BGR, near-white paper (not cream)
INK = np.array([0.065, 0.065, 0.075], np.float32)    # BGR, the dark outline used on a light world
LIGHT_LUMA = 0.72    # a world pixel this bright (BT.709 luma, 0..1) swallows the near-white paper border
LIGHT_SHARE = 0.35   # switch the border to a dark outline when this share of the band under it is that bright


def border_ink(world, hard, border_px=12):
    """Collage legibility: is the world under the paper border light enough to swallow it?
    world (H, W, 3) float 0..255, hard (H, W) bool artist mask. The band is where the border sits (outside the
    artist, out to the border's widest jag, 1.6x its width). -> ("dark" | "white", share of the band brighter than
    LIGHT_LUMA). A share, not a mean: the acid stage's rings average to mid grey and still swallow half the border."""
    d = cv2.distanceTransform((~hard).astype(np.uint8), cv2.DIST_L2, 3)
    band = (d > 0) & (d <= 1.6 * border_px)
    if not band.any():
        return "white", 0.0
    px = world[band].astype(np.float32) / 255
    lum = px[:, 0] * 0.0722 + px[:, 1] * 0.7152 + px[:, 2] * 0.2126
    share = float((lum > LIGHT_LUMA).mean())
    return ("dark" if share > LIGHT_SHARE else "white"), share


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def refine_alpha(mask_bool, erode_px, feather):
    m = mask_bool.astype(np.uint8) * 255
    if erode_px > 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)))
    a = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), feather)
    return a * a * (3 - 2 * a)


def lab_stats(bgr_f, weight=None):
    lab = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    if weight is not None:
        lab = lab[weight.reshape(-1)]
        if len(lab) < 200:
            return None
    return lab.mean(axis=0), lab.std(axis=0) + 1e-3


def luma3(img):
    """BGR -> grey (BT.709 luma), still 3 channels, same dtype/scale."""
    y = img[..., 0] * 0.0722 + img[..., 1] * 0.7152 + img[..., 2] * 0.2126
    return np.repeat(y[..., None], 3, axis=2)


def boil_step(global_frame):
    """Which 12 fps boil step a 30 fps frame belongs to (integer maths: no float drift at step edges)."""
    return (global_frame * BOIL_FPS) // FPS


# --------------------------------------------------------------------------- camera

def crop_matrix(crop):
    """crop = (x0, y0, cw, ch) in source pixels -> 2x3 affine mapping that crop onto the full output frame."""
    x0, y0, cw, ch = crop
    z = W / cw
    return np.float32([[z, 0, -z * x0], [0, z, -z * y0]]), z


def is_identity(crop):
    x0, y0, cw, ch = crop
    return abs(cw - W) < 0.5 and abs(x0) < 0.5 and abs(y0) < 0.5


# --------------------------------------------------------------------------- despill

def _normconv(img, w, sigma):
    num = cv2.GaussianBlur(img * w[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)[..., None], den


def despill(fg, hard, target, **kw):
    """estimate + apply in one call (tests and single-look use)."""
    est = despill_estimate(fg, hard, **kw)
    return despill_apply(fg, est, target)


def despill_estimate(fg, hard, band_px=8, reach_px=120, max_pocket_px=9000, scale=1.0):
    """Colour decontamination of the artist's edge (the look-independent half).

    fg      (H, W, 3) float32 0..1, the source pixels (output space)
    hard    (H, W) bool, the artist mask (output space)
    target  (H, W, 3) float32 0..1, what should sit behind the edge (blurred world, grey world, or paper)

    Model: near the edge a pixel P is a mix of the artist's local colour F and the source background's local
    colour B: P = a F + (1 - a) B. F and B are smooth local estimates (normalised convolution of confident
    pixels, at quarter resolution). Where the mix is real (F and B differ, P lies on the F-B line) the background
    share (1 - a) is swapped for the target: P' = P + (1 - a)(T - B).
    - The thin band (`band_px` inside the boundary) gets this everywhere.
    - Deeper in (up to `reach_px`; a gap under the chin can sit 80 px from the outer edge), only small enclosed pockets that read as pure background (a gap between
      chin, hand and collar that the mask filled) get it, and their alpha is cut so the world shows through.
      A pocket bigger than `max_pocket_px` is not a pocket (a blue shirt on a blue wall), so it is left alone.
    scale: the reframe zoom. Distances are in source pixels, so a 1.9x close-up needs 1.9x the reach and
    1.9^2 the pocket areas (without it, a pocket 50 px from the edge sits 95 px away in the close-up and is missed).
    Returns an estimate for despill_apply(fg, est, target) -> (fg', alpha_cut (H, W) 0..1, stats).
    """
    q = 4
    s = max(1.0, float(scale))
    band_px, reach_px = band_px * s, reach_px * s
    min_pocket_px, max_pocket_px = 24 * s * s, max_pocket_px * s * s
    it = lambda n: max(1, int(round(n * s)))  # noqa: E731
    hq, wq = H // q, W // q
    fq = cv2.resize(fg, (wq, hq), interpolation=cv2.INTER_AREA)
    mq = cv2.resize(hard.astype(np.uint8) * 255, (wq, hq), interpolation=cv2.INTER_AREA) > 127
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mq8 = mq.astype(np.uint8)
    ring = (cv2.dilate(mq8, k3, iterations=it(4)) > 0) & ~(cv2.dilate(mq8, k3, iterations=it(1)) > 0)
    core = cv2.erode(mq8, k3, iterations=it(1)) > 0
    B, bw = _normconv(fq, ring.astype(np.float32), 6 * s)
    B2, _ = _normconv(fq, ring.astype(np.float32), 18 * s)
    B = np.where((bw > 0.02)[..., None], B, B2)
    F0, _ = _normconv(fq, core.astype(np.float32), 4 * s)

    def unmix(P, Fc, Bc):
        d = Fc - Bc
        dn = np.sqrt((d * d).sum(-1)) + 1e-6
        a = np.clip(((P - Bc) * d).sum(-1) / (dn * dn), 0, 1)
        resid = np.sqrt(((P - (Bc + a[..., None] * d)) ** 2).sum(-1))
        conf = smoothstep(0.06, 0.16, dn) * np.clip(1 - resid / (0.6 * dn + 0.03), 0, 1)
        return (1 - a) * conf

    s0 = unmix(fq, F0, B) * mq
    w1 = (core & (s0 < 0.3)).astype(np.float32)  # confident artist pixels: drop the pocket from F's estimate
    F, fw = _normconv(fq, w1, 5 * s)
    F = np.where((fw > 0.02)[..., None], F, F0)

    dist = cv2.distanceTransform(hard.astype(np.uint8), cv2.DIST_L2, 3)
    region = hard & (dist <= reach_px)
    ys, xs = np.nonzero(region)
    if len(ys) == 0:
        return None
    Fu = cv2.resize(F, (W, H), interpolation=cv2.INTER_LINEAR)[ys, xs]
    Bu = cv2.resize(B, (W, H), interpolation=cv2.INTER_LINEAR)[ys, xs]
    P = fg[ys, xs]
    spill = unmix(P, Fu, Bu)
    full = np.zeros((H, W), np.float32)
    full[ys, xs] = spill
    # pockets: connected bg-like blobs inside the mask, small enough to be gaps rather than clothing
    sq = cv2.resize(full, (wq, hq), interpolation=cv2.INTER_AREA)
    cand = ((sq > 0.35) & mq).astype(np.uint8)  # 0.35: a pocket's narrow tip averages with skin at 1/4 res
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA] * q * q
        keep[i] = min_pocket_px <= area <= max_pocket_px
    pocket_q = keep[lab].astype(np.float32)
    # widen the pocket region by 2 quarter-px; the cut below is still gated by each pixel's own spill
    pocket_q = cv2.GaussianBlur(cv2.dilate(pocket_q, k3, iterations=it(2)), (0, 0), 0.8)
    pocket = cv2.resize(pocket_q, (W, H), interpolation=cv2.INTER_LINEAR)[ys, xs]
    edge = np.clip(1 - (dist[ys, xs] - 0.5) / band_px, 0, 1)
    w = np.maximum(edge, pocket) * spill
    cut = np.zeros((H, W), np.float32)
    cut[ys, xs] = pocket * smoothstep(0.35, 0.8, spill)
    # grow the cut a few px into the pocket's anti-aliased rim (half room, half collar), still gated per pixel by
    # that pixel's own room share, so a pure artist pixel next to a pocket cannot be eaten
    rim = np.zeros((H, W), np.float32)
    rim[ys, xs] = pocket * smoothstep(0.12, 0.45, spill)
    r = int(round(2 * s))
    grown = cv2.dilate(cut, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    cut = np.maximum(cut, grown * rim)
    stats = {"band_px": int((edge > 0).sum()), "touched_px": int((w > 0.05).sum()),
             "pocket_px": int((cut > 0.5).sum()), "mean_edge_spill": round(float(spill[edge > 0.5].mean()), 4)
             if (edge > 0.5).any() else 0.0}
    return {"ys": ys, "xs": xs, "P": P, "B": Bu, "w": w, "cut": cut, "stats": stats}


def despill_apply(fg, est, target):
    """Swap the estimated source-background share for `target` (an (H, W, 3) image or a BGR colour)."""
    if est is None:
        return fg, np.zeros((H, W), np.float32), {"band_px": 0, "pocket_px": 0}
    ys, xs, P, Bu, w = est["ys"], est["xs"], est["P"], est["B"], est["w"]
    T = target[ys, xs] if np.ndim(target) == 3 else np.asarray(target, np.float32)[None, :]
    out = fg.copy()
    out[ys, xs] = np.clip(P + w[:, None] * (T - Bu), 0, 1)
    return out, est["cut"], est["stats"]


# --------------------------------------------------------------------------- compositor

class Grain:
    """A small bank of pre-drawn grain frames (shared by every look; cycling 8 frames reads as live grain)."""

    def __init__(self, n=8, seed=1000):
        r = np.random.default_rng(seed)
        self.bank = [cv2.GaussianBlur(r.standard_normal((H, W)).astype(np.float32), (0, 0), 0.7) * 1.9
                     for _ in range(n)]

    def apply(self, out, idx, strength):
        g = self.bank[idx % len(self.bank)]
        if (idx // len(self.bank)) % 2:
            g = g[::-1, ::-1]
        lum = out.mean(axis=2) / 255
        gg = g * strength * (0.55 + 2.0 * lum * (1 - lum))
        return np.clip(out + gg[..., None], 0, 255)


class ColorMatch:
    """The spike's LAB mean/std transfer toward the world, smoothed over time (EMA), at `strength`."""

    def __init__(self, strength):
        self.k = strength
        self.ema = None

    def __call__(self, fg, alpha, target_stats):
        if self.k <= 0:
            return fg
        small = cv2.resize(fg, (270, 480), interpolation=cv2.INTER_AREA)
        sa = cv2.resize(alpha, (270, 480), interpolation=cv2.INTER_AREA) > 0.5
        st = lab_stats(small, sa)
        if st is not None:
            mu, sd = st
            self.ema = (mu, sd) if self.ema is None else (0.85 * self.ema[0] + 0.15 * mu, 0.85 * self.ema[1] + 0.15 * sd)
        if self.ema is None:
            return fg
        mu, sd = self.ema
        t_mu, t_sd = target_stats
        gain = 1 + self.k * (np.clip(t_sd / sd, 0.6, 1.4) - 1)
        shift = self.k * (t_mu - mu)
        lab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB)
        lab = (lab - mu) * gain + mu + shift
        return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR), 0, 1)


class Collage:
    """Paper cut-out: an irregular white border (the dilated mask), a hard offset shadow, paper grain, all
    re-randomised at 12 fps (stop-motion boil) while the artist inside stays at 30 fps."""

    def __init__(self, border_px=12, jag_px=3.5, shadow=(20, 28), shadow_dark=0.74, seed=4242):
        self.border, self.jag, self.shadow, self.dark, self.seed = border_px, jag_px, shadow, shadow_dark, seed
        self._k = None

    def _step(self, k):
        if self._k == k:
            return
        r = np.random.default_rng(self.seed + k)
        nz = r.standard_normal((H // 16 + 2, W // 16 + 2)).astype(np.float32)
        nz = cv2.resize(cv2.GaussianBlur(nz, (0, 0), 1.2), (W, H), interpolation=cv2.INTER_CUBIC)
        self.radius = np.clip(self.border + self.jag * nz / (nz.std() + 1e-6), 0.5 * self.border, 1.6 * self.border)
        fib = r.standard_normal((H // 3, W // 3)).astype(np.float32)
        fib = cv2.resize(cv2.GaussianBlur(fib, (0, 0), sigmaX=0.6, sigmaY=3.0), (W, H), interpolation=cv2.INTER_LINEAR)
        fine = r.standard_normal((H, W)).astype(np.float32)
        self.tex = (1.0 + 0.018 * fib / (fib.std() + 1e-6) + 0.012 * fine)[..., None]  # paper grain, any colour
        self.paper = self.tex * PAPER
        jx, jy = r.integers(-3, 4, 2)
        self.offset = (self.shadow[0] + int(jx), self.shadow[1] + int(jy))
        self._k = k

    def layers(self, hard, k, ink=None):
        """-> (paper_alpha, shadow_alpha, paper_rgb 0..1) for boil step k; ink: the border colour (default paper)."""
        self._step(k)
        d_out = cv2.distanceTransform((~hard).astype(np.uint8), cv2.DIST_L2, 3)
        pa = np.clip(self.radius - d_out + 0.5, 0, 1)
        pa[hard] = 1.0
        sx, sy = self.offset
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        sa = cv2.warpAffine(pa, M, (W, H), borderValue=0)
        sa = cv2.GaussianBlur(sa, (0, 0), 1.2) * self.dark
        return pa, sa, (self.paper if ink is None else self.tex * ink)


class Compositor:
    def __init__(self, args):
        self.a = args
        self.grain = Grain()
        self.match = {look: ColorMatch(args.match) for look in LOOKS}  # one EMA per look: renders stay independent
        self.collage = Collage()
        self.counter = None

    def frame(self, src_bgr, mask_bool, crop, world, front, world_stats, idx, gframe, looks, native=None,
              ns=(1.0, 1.0), counter=False, ink=None):
        """-> ({look: uint8 frame}, info). world/front float32 0..255 (already the shot's world camera).
        native: the same frame at source resolution (9:16 crop, `ns` native px per proxy px). When given, a
        punch-in takes its pixels from it instead of upscaling the 1080 proxy, and a close shot's mask is
        edge-refined against those pixels (realone/native.py).
        counter: draw the Real Receipt counter on the receipt look: 0/False no, 1/True top left, 2 top right,
        3 lower left (receipt.PLACES).
        ink: the collage border colour: None = near-white paper; INK on a light world (see border_ink).
        info[look]["filmed"]: the share of the frame that is the artist's filmed pixels (the artist mask in output
        space minus any pocket the despill cut open), the same for every look."""
        a = self.a
        alpha_src = refine_alpha(mask_bool, a.erode, a.feather)
        fg = src_bgr.astype(np.float32) / 255
        info_native = None
        if is_identity(crop):
            hard, alpha, zoom = mask_bool, alpha_src, 1.0
        else:
            M, zoom = crop_matrix(crop)
            if native is not None:
                from realone.native import REFINE_MIN_UPSAMPLE, native_crop, refine_mask_in_crop
                fg = native_crop(native, crop, ns)
                info_native = "native"
            else:
                fg = np.clip(cv2.warpAffine(fg, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT), 0, 1)
            if native is not None and zoom >= REFINE_MIN_UPSAMPLE:
                # the mask edge now follows the native pixels: erode/feather in output px per native px
                hard = refine_mask_in_crop(fg, mask_bool, crop)
                k = max(1.0, zoom / min(ns))
                alpha = refine_alpha(hard, max(1, int(round(a.erode * k))), a.feather * k)
                info_native = "native+refined"
            else:
                alpha = cv2.warpAffine(alpha_src, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
                hard = cv2.warpAffine(mask_bool.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR,
                                      borderValue=0) > 0.5
        wb_small = cv2.GaussianBlur(cv2.resize(world, (270, 480), interpolation=cv2.INTER_AREA), (0, 0), 4.5)
        wb = cv2.resize(wb_small, (W, H), interpolation=cv2.INTER_LINEAR) / 255
        self.est = despill_estimate(fg, hard, scale=zoom) if a.despill else None
        cut = self.est["cut"] if self.est is not None else 0.0
        filmed = float((hard * (1.0 - cut)).mean())
        outs, info = {}, {}
        for look in looks:
            if look == "clean":
                o, inf = self._clean(fg, alpha, hard, world, front, wb, world_stats)
            elif look == "receipt":
                o, inf = self._receipt(fg, alpha, hard, world, front, wb)
            else:
                o, inf = self._collage(fg, alpha, hard, world, front, wb, world_stats, gframe, ink)
            inf = dict(inf, source=info_native or "proxy", filmed=filmed)
            if counter and look == "receipt":
                if self.counter is None:
                    from realone.receipt import Counter
                    self.counter = Counter()
                inf["counter_text"] = self.counter.draw(o, filmed, int(counter))
                inf["counter_place"] = int(counter)
            outs[look] = self.grain.apply(o, idx, a.grain if look != "collage" else a.grain * 0.8).astype(np.uint8)
            info[look] = inf
        return outs, info

    def _wrap(self, fg, alpha, wb):
        near = cv2.GaussianBlur(1 - alpha, (0, 0), 7)
        w = (np.clip(near * 2, 0, 1) * alpha * self.a.wrap)[..., None]
        return fg * (1 - w) + wb * w + wb * w * 0.25

    def _clean(self, fg, alpha, hard, world, front, wb, wstats):
        fgd, cut, st = despill_apply(fg, self.est, wb)
        al = alpha * (1 - cut)
        f = self.match["clean"](fgd, al, wstats)
        f = self._wrap(f, al, wb)
        out = f * al[..., None] * 255 + world * (1 - al[..., None])
        return out + front * (1 - 0.35 * al[..., None]), st

    def _receipt(self, fg, alpha, hard, world, front, wb):
        wg, fgr, wbg = luma3(world), luma3(front), luma3(wb)
        fgd, cut, st = despill_apply(fg, self.est, wbg)
        al = alpha * (1 - cut)
        out = fgd * al[..., None] * 255 + wg * (1 - al[..., None])
        return out + fgr * (1 - al[..., None]), st  # grey particles stay off the filmed artist

    def _collage(self, fg, alpha, hard, world, front, wb, wstats, gframe, ink=None):
        k = boil_step(gframe)
        pa, sa, paper = self.collage.layers(hard, k, ink)
        fgd, cut, st = despill_apply(fg, self.est, PAPER if ink is None else ink)
        al = refine_alpha(hard, 1, 0.8) * (1 - cut)  # a crisper scissor edge than the clean look
        f = self.match["collage"](fgd, al, wstats)
        base = world * (1 - sa[..., None])
        base = base * (1 - pa[..., None]) + paper * 255 * pa[..., None]
        out = f * al[..., None] * 255 + base * (1 - al[..., None])
        st = dict(st, boil_step=int(k), ink="white" if ink is None else "dark")
        return out + front * 0.6 * (1 - pa[..., None]), st
