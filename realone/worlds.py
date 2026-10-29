"""Worlds: Muse Image plates (keyed, one per prompt, cached) or procedural plates (TEST ONLY), plus the spike's
code-drawn particle layer and a world camera. Everything that moves is a pure function of the world clock."""
import base64
import hashlib
import json
import cv2
import numpy as np

from realone.common import H, OUT, W, Refuse
from realone.meta_api import IMAGE_MODEL, IMAGE_USD

WORLD_SIZE = "1080x1920"  # sets the aspect ratio only (docs: image-generation, `size`)
WORLD_WRAP = (
    ". Vertical 9:16 empty background plate for a music video: no people, no text, no logos, "
    "cinematic lighting, camera at chest height, open floor space in the lower centre, no motion blur."
)


# --------------------------------------------------------------------------- Muse Image (keyed)

def gen_world(client, prompt, wrap=True):
    """Muse Image 9:16 via POST /v1/images/generations (docs: image-generation), cached by prompt hash in
    out/worlds/. Returns (path, billed_usd).
    # UNCONFIRMED: `size="1080x1920"` (docs say size sets the aspect ratio only and accepts any "WxH");
    # `extra_body.tool_enablement` per the docs' extra_body example."""
    full = prompt + (WORLD_WRAP if wrap else "")
    d = OUT / "worlds"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"muse_{hashlib.sha1(full.encode()).hexdigest()[:12]}.png"
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists() and json.loads(meta.read_text()).get("prompt") == full:
        print(f"world: cache hit {path.name} (same prompt), no API call")
        return path, 0.0
    if client is None:
        raise Refuse("a Muse Image world is not cached and there is no MODEL_API_KEY")
    print(f"Muse Image: generating '{prompt[:60]}' ...")
    resp = client.images.generate(
        model=IMAGE_MODEL, prompt=full, n=1, size=WORLD_SIZE, output_format="png",
        extra_body={"reasoning_strength": "high",
                    "tool_enablement": {"enable_web_search": False, "enable_shell": False}})
    img = cv2.imdecode(np.frombuffer(base64.b64decode(resp.data[0].b64_json), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise Refuse("Muse Image returned something that is not a decodable image")
    cv2.imwrite(str(path), img)
    meta.write_text(json.dumps({"prompt": full, "model": IMAGE_MODEL, "size_requested": WORLD_SIZE,
                                "size_returned": [img.shape[1], img.shape[0]]}, indent=2))
    return path, IMAGE_USD


def muse_world_cached(prompt, wrap=True):
    full = prompt + (WORLD_WRAP if wrap else "")
    path = OUT / "worlds" / f"muse_{hashlib.sha1(full.encode()).hexdigest()[:12]}.png"
    return path.exists()


# --------------------------------------------------------------------------- procedural plates (TEST ONLY)

def _grad(c0, c1, n):
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    return np.array(c0, np.float32) * (1 - t) + np.array(c1, np.float32) * t


def plate_neon_alley():
    """Night alley: magenta/teal neon tubes on near-black walls, a wet floor with streaked reflections."""
    img = np.zeros((H, W, 3), np.float32)
    img[:] = _grad((40, 8, 18), (14, 4, 8), H)[:, None, :]  # BGR: deep violet to black
    vy = int(H * 0.47)
    for x0, col in ((90, (230, 60, 255)), (250, (255, 230, 40)), (830, (230, 60, 255)), (990, (255, 230, 40))):
        cv2.rectangle(img, (x0 - 9, 260), (x0 + 9, vy + 120), col, -1)
    for y0, col in ((330, (255, 230, 40)), (520, (230, 60, 255))):
        cv2.rectangle(img, (380, y0 - 7), (700, y0 + 7), col, -1)
    glow = cv2.GaussianBlur(img, (0, 0), 28) * 1.6
    img = np.maximum(img, glow)
    floor = np.zeros_like(img)
    floor[vy + 160:] = (24, 12, 16)
    for k in range(-9, 10):  # perspective lines to a central vanishing point
        cv2.line(floor, (W // 2, vy), (W // 2 + k * 260, H), (70, 40, 60), 3, cv2.LINE_AA)
    refl = cv2.flip(img[vy - 400:vy + 160], 0)
    refl = cv2.resize(refl, (W, H - vy - 160))
    refl = cv2.GaussianBlur(refl, (0, 0), 6)
    refl = cv2.resize(cv2.resize(refl, (W, (H - vy - 160) // 6)), (W, H - vy - 160))  # vertical streaks
    img[vy + 160:] = floor[vy + 160:] + refl * 0.45
    return img


def plate_sunset_roof():
    """Rooftop at sunset: orange-to-violet sky, a low sun, a black skyline with lit windows, a dark roof."""
    img = np.zeros((H, W, 3), np.float32)
    hz = int(H * 0.62)
    img[:hz] = _grad((120, 30, 70), (40, 120, 255), hz)[:, None, :]  # violet top to hot orange horizon
    cv2.circle(img, (W // 2 + 190, hz - 470), 120, (150, 230, 255), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    rng = np.random.default_rng(3)
    x = 0
    while x < W:
        bw = int(rng.integers(70, 180))
        bh = int(rng.integers(160, 560))
        cv2.rectangle(img, (x, hz - bh), (x + bw, hz), (20, 14, 18), -1)
        for wy in range(hz - bh + 24, hz - 20, 34):
            for wx in range(x + 12, x + bw - 14, 26):
                if rng.random() < 0.35:
                    cv2.rectangle(img, (wx, wy), (wx + 10, wy + 14), (110, 210, 255), -1)
        x += bw + int(rng.integers(0, 14))
    img[hz:] = _grad((38, 30, 40), (14, 10, 14), H - hz)[:, None, :]
    cv2.line(img, (0, hz + 120), (W, hz + 70), (60, 70, 120), 4, cv2.LINE_AA)  # parapet edge catching light
    return img


def plate_acid_stage():
    """Stage: a saturated chartreuse cyc, a hard black-and-white target and floor stripes, one spotlight pool."""
    img = np.zeros((H, W, 3), np.float32)
    img[:] = (20, 235, 200)  # BGR chartreuse
    c = (W // 2, int(H * 0.33))
    for i, r in enumerate(range(520, 40, -80)):
        cv2.circle(img, c, r, (10, 10, 10) if i % 2 == 0 else (245, 245, 245), -1, cv2.LINE_AA)
    fy = int(H * 0.66)
    img[fy:] = (12, 12, 12)
    for k in range(-12, 14):
        pts = np.array([[k * 120, fy], [k * 120 + 60, fy], [k * 240 + 120 - W // 2, H], [k * 240 - W // 2, H]], np.int32)
        cv2.fillPoly(img, [pts], (230, 230, 230), cv2.LINE_AA)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    pool = np.exp(-(((xx - W / 2) / 360) ** 2 + ((yy - H * 0.86) / 140) ** 2))
    img = img * (0.85 + 0.45 * pool[..., None])
    return img


LOCAL_PLATES = [("neon alley", plate_neon_alley, "rain"), ("sunset roof", plate_sunset_roof, "embers"),
                ("acid stage", plate_acid_stage, "dust")]


def local_plates():
    """TEST ONLY: three visibly different procedural plates, cached in out/worlds/. -> [(name, path, particles)]"""
    d = OUT / "worlds"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (name, fn, parts) in enumerate(LOCAL_PLATES):
        p = d / f"TEST_LOCAL_plate_{i}.png"
        if not p.exists():
            img = fn()
            g = np.random.default_rng(i).standard_normal((H, W, 1)).astype(np.float32) * 3.0
            cv2.imwrite(str(p), np.clip(img + g, 0, 255).astype(np.uint8))
        out.append((name, p, parts))
    return out


def load_plate(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise Refuse(f"cannot read world image {path}")
    return img


# --------------------------------------------------------------------------- world camera + particles

def world_view(plate, zoom, center, dx=0.0, dy=0.0):
    """Cover-fit the plate to 9:16, then zoom by `zoom` about frame point `center` (x, y), plus a drift."""
    ph, pw = plate.shape[:2]
    s0 = max(W / pw, H / ph)
    cx, cy = center
    s = s0 * zoom
    # plate pixel p -> frame: cover-fit puts plate centre at frame centre; then zoom about (cx, cy)
    tx = W / 2 - s0 * pw / 2
    ty = H / 2 - s0 * ph / 2
    M = np.float32([[s, 0, zoom * (tx - cx) + cx + dx], [0, s, zoom * (ty - cy) + cy + dy]])
    return cv2.warpAffine(plate, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


PARALLAX = 0.25  # the world zooms a quarter as much as the artist: the artist feels closer than the world


def world_layer(plate, parts, crop, push_frac, push, wt, cam=None):
    """The world for one frame, a pure function of the WORLD clock `wt` (seconds) and the camera `crop`:
    -> (world float32 0..255 with the back particles, front particles float32 0..255, plate-only float32).
    The world zooms a quarter as much as the artist (cheap parallax) toward the same point, plus `push` over
    the shot (`push_frac` 0..1, measured on the world clock) and a slow sideways drift.
    cam (Hook Freeze v2, `bullet`): (dx, vel, zoom) on the REAL frame clock: the plate slides dx px sideways (the
    particle layers slide further, by depth), `vel` is the world clock's rate (0 frozen: crisp hanging marks),
    `zoom` a margin so the slide never shows the plate's edge. None: the 0.2.0 maths, byte for byte."""
    x0, y0, cw, ch = crop
    z = W / cw
    ccx, ccy = x0 + cw / 2, y0 + ch / 2
    cdx, vel, czoom = cam if cam is not None else (0.0, 1.0, 1.0)
    zw = (1 + PARALLAX * (z - 1)) * (1 + push * push_frac)
    if cam is not None:
        zw *= czoom
    centre = (W / 2 + (ccx - W / 2) * PARALLAX, H / 2 + (ccy - H / 2) * PARALLAX)
    wbase = world_view(plate, zw, centre, dx=16 * np.sin(0.55 * wt) + cdx).astype(np.float32)
    back, front = parts.layers(wt) if cam is None else parts.layers(wt, cam=cdx, vel=vel)
    return np.clip(wbase + back * 255, 0, 255), front * 255, wbase


def world_key(world, front):
    """Hash of the world layer (what the freeze check compares frame to frame)."""
    return hashlib.md5(world.tobytes() + front.tobytes()).hexdigest()


def plate_thumb(wbase):
    return cv2.resize(wbase, (135, 240), interpolation=cv2.INTER_AREA)


class Particles:
    """Code-drawn particles, positions a pure function of the WORLD clock (a frozen clock is a frozen layer)."""

    def __init__(self, kind, seed=7):
        self.kind = kind
        r = np.random.default_rng(seed)
        n = {"dust": 150, "rain": 240, "embers": 110, "none": 0}[kind]
        self.n = n
        self.x0 = r.uniform(0, W, n)
        self.y0 = r.uniform(0, H, n)
        self.depth = r.uniform(0, 1, n)  # 0 far .. 1 near
        self.phase = r.uniform(0, 6.28, n)
        self.front = self.depth > 0.82  # ~18% drawn over the artist

    def layers(self, wt):
        back = np.zeros((H, W, 3), np.float32)
        front = np.zeros((H, W, 3), np.float32)
        if self.n == 0:
            return back, front
        d, ph = self.depth, self.phase
        for i in range(self.n):
            tgt = front if self.front[i] else back
            z = d[i]
            if self.kind == "dust":
                x = (self.x0[i] + (6 + 20 * z) * wt + 18 * np.sin(wt * 0.5 + ph[i])) % W
                y = (self.y0[i] - (10 + 30 * z) * wt) % H
                rad = 1.2 + 4.5 * z
                a = (0.22 + 0.5 * z) * (0.7 + 0.3 * np.sin(wt * 1.7 + ph[i]))
                cv2.circle(tgt, (int(x), int(y)), int(round(rad)), (a * 0.80, a * 0.92, a * 1.0), -1, cv2.LINE_AA)
            elif self.kind == "rain":
                spd = 1300 + 900 * z
                x = (self.x0[i] - 0.18 * spd * wt) % W
                y = (self.y0[i] + spd * wt) % (H + 200) - 100
                ln = 40 + 70 * z
                a = 0.18 + 0.3 * z
                cv2.line(tgt, (int(x), int(y)), (int(x - 0.18 * ln), int(y - ln)),
                         (a * 0.92, a * 0.86, a * 0.80), 1 + int(z > 0.6), cv2.LINE_AA)
            else:  # embers
                x = (self.x0[i] + 25 * np.sin(wt * 0.8 + ph[i]) + 8 * wt) % W
                y = (self.y0[i] - (60 + 170 * z) * wt) % H
                a = (0.35 + 0.65 * z) * (0.6 + 0.4 * np.sin(wt * 6 + ph[i]))
                cv2.circle(tgt, (int(x), int(y)), 1 + int(2.6 * z), (a * 0.16, a * 0.55, a * 1.0), -1, cv2.LINE_AA)
        back = cv2.GaussianBlur(back, (0, 0), 1.3)
        front = cv2.GaussianBlur(front, (0, 0), 3.1) * 1.6  # near = soft bokeh
        if self.kind == "embers":
            back += cv2.GaussianBlur(back, (0, 0), 9) * 1.5
        return back, front



# --------------------------------------------------------------------------- Hook Freeze v2 (bullet): depth particles

# Three depth layers, far -> near. Everything that sells depth scales with nearness: count falls, size, blur,
# brightness and fall speed rise, and the camera parallax (how far a layer slides per pixel the PLATE slides) rises.
# `front`: drawn over the artist. Speeds are px/s at a world clock rate of 1.
DEPTH_LAYERS = (
    {"name": "far", "count": 1.0, "size": 1.6, "blur": 0.8, "bright": 0.42, "speed": 0.55, "parallax": 2.5, "front": False},
    {"name": "mid", "count": 0.5, "size": 2.8, "blur": 1.1, "bright": 0.75, "speed": 1.0, "parallax": 4.5, "front": False},
    {"name": "near", "count": 0.14, "size": 5.5, "blur": 2.2, "bright": 1.0, "speed": 1.7, "parallax": 9.0, "front": True},
)
DEPTH_KINDS = {  # base count (far layer), base fall speed px/s (+ down, - up), sideways px/s, motion-blur shutter
    "rain": {"n": 300, "fall": 1500.0, "side": -260.0, "shutter": 0.9},
    "dust": {"n": 260, "fall": -150.0, "side": 120.0, "shutter": 0.9},
    "embers": {"n": 200, "fall": -420.0, "side": 60.0, "shutter": 0.8},
    "none": {"n": 0, "fall": 0.0, "side": 0.0, "shutter": 0.0},
}
FROZEN_GAIN = 1.6  # brightness of a frozen particle relative to a moving one
WRAP_PAD = 160  # particles live on a canvas this much larger than the frame each side, so a wrap is never seen


class DepthParticles:
    """Hook Freeze v2 particles on three depth layers (far, mid, near).

    World-space positions are a pure function of the WORLD clock `wt` (`state(wt)`): a frozen clock is frozen
    particles. How they are drawn also takes the camera `cam` (px the plate has slid sideways; each layer slides
    `parallax` times as far, so frozen layers slide past each other) and `vel`, the world clock's rate this frame
    (0 inside a freeze): a moving particle is a motion-blur streak `speed * shutter / FPS` long, a frozen one a crisp
    short mark that visibly hangs."""

    depth_layers = DEPTH_LAYERS

    def __init__(self, kind, seed=7):
        self.kind = kind
        k = DEPTH_KINDS[kind]
        self.k = k
        r = np.random.default_rng(seed)
        self.layers_ = []
        for li, L in enumerate(DEPTH_LAYERS):
            n = int(round(k["n"] * L["count"]))
            self.layers_.append({**L, "n": n, "x0": r.uniform(0, W + 2 * WRAP_PAD, n),
                                 "y0": r.uniform(0, H + 2 * WRAP_PAD, n), "ph": r.uniform(0, 6.28, n),
                                 "jit": r.uniform(0.8, 1.2, n),
                                 # per-particle length and brightness variation, widest far away (uniform dashes
                                 # read as code-drawn)
                                 "lenj": r.uniform(1 - 0.45 * (1 - li / 2), 1 + 0.45 * (1 - li / 2), n),
                                 "brj": r.uniform(1 - 0.35 * (1 - li / 2), 1 + 0.25 * (1 - li / 2), n)})
        self.n = sum(L["n"] for L in self.layers_)

    def _vel(self, L):
        """px/s of each particle in the layer at clock rate 1 (x, y)."""
        return self.k["side"] * L["speed"] * L["jit"], self.k["fall"] * L["speed"] * L["jit"]

    def _pos(self, L, wt):
        vx, vy = self._vel(L)
        wob = 0.0
        if self.kind == "dust":
            wob = 22 * np.sin(0.5 * wt + L["ph"])
        elif self.kind == "embers":
            wob = 28 * np.sin(0.8 * wt + L["ph"])
        x = (L["x0"] + vx * wt + wob) % (W + 2 * WRAP_PAD)
        y = (L["y0"] + vy * wt) % (H + 2 * WRAP_PAD)
        return x, y

    def state(self, wt):
        """World-space positions of every particle, (N, 2) float64, a pure function of the world clock."""
        if self.n == 0:
            return np.zeros((0, 2))
        return np.concatenate([np.stack(self._pos(L, wt), 1) for L in self.layers_])

    def layers(self, wt, cam=0.0, vel=1.0):
        back = np.zeros((H, W, 3), np.float32)
        front = np.zeros((H, W, 3), np.float32)
        if self.n == 0:
            return back, front
        sh = self.k["shutter"]
        colour = {"rain": (0.95, 0.88, 0.80), "dust": (0.80, 0.92, 1.0), "embers": (0.16, 0.55, 1.0)}[self.kind]
        for L in self.layers_:
            x, y = self._pos(L, wt)
            sx = (x + L["parallax"] * cam) % (W + 2 * WRAP_PAD) - WRAP_PAD
            sy = y - WRAP_PAD
            vx, vy = self._vel(L)
            # motion blur: the distance travelled while the shutter is open (none when the clock stands still)
            bx, by = vx * vel * sh / 30.0, vy * vel * sh / 30.0
            size = L["size"]
            a = L["bright"]
            if self.kind == "embers":
                a = a * (0.6 + 0.4 * np.sin(wt * 6 + L["ph"]))
            elif self.kind == "dust":
                a = a * (0.7 + 0.3 * np.sin(wt * 1.7 + L["ph"]))
            if vel == 0:  # a frozen particle's light is no longer smeared along a streak: it reads as a bright point
                a = a * FROZEN_GAIN
            a = np.broadcast_to(a, x.shape) * L["brj"]
            canvas = np.zeros((H, W, 3), np.float32)
            th = max(1, int(round(size if self.kind != "rain" else 0.55 * size)))  # rain: thin lines, not pills
            for i in range(L["n"]):
                c = (a[i] * colour[0], a[i] * colour[1], a[i] * colour[2])
                x1, y1 = sx[i], sy[i]
                if self.kind == "rain":
                    # frozen: a short crisp drop (its length a fixed 5 sizes, along the fall direction)
                    lj = L["lenj"][i]
                    hx, hy = (bx[i] * lj, by[i] * lj) if vel > 0 else (0.0, 0.0)
                    if abs(hx) + abs(hy) < 5.0 * size * lj:
                        nrm = np.hypot(vx[i], vy[i]) + 1e-6
                        hx, hy = vx[i] / nrm * 5.0 * size * lj, vy[i] / nrm * 5.0 * size * lj
                    cv2.line(canvas, (int(x1), int(y1)), (int(x1 - hx), int(y1 - hy)), c, th, cv2.LINE_AA)
                else:
                    rad = max(1, int(round(0.9 * size)))
                    if vel > 0 and abs(by[i]) + abs(bx[i]) > 2:
                        cv2.line(canvas, (int(x1), int(y1)), (int(x1 - bx[i]), int(y1 - by[i])), c, 2 * rad,
                                 cv2.LINE_AA)
                    else:
                        cv2.circle(canvas, (int(x1), int(y1)), rad, c, -1, cv2.LINE_AA)
            canvas = cv2.GaussianBlur(canvas, (0, 0), L["blur"])
            if self.kind == "embers":
                canvas += cv2.GaussianBlur(canvas, (0, 0), 3 + 3 * L["size"]) * 1.2
            if L["front"]:
                front += canvas * 1.5
            else:
                back += canvas
        return back, front


def make_particles(kind, style="still"):
    """The particle layer for a world: the 0.2.0 `Particles` for `still` (and every render without a freeze, so
    those stay byte-identical), the depth particles for the Hook Freeze v2 `bullet` style."""
    return DepthParticles(kind) if style == "bullet" else Particles(kind)


def grade_world(img, g):
    """The frozen-world grade at strength g (0..1): saturation x0.45 and a slight cool shift at g=1. BGR 0..255."""
    if g <= 0:
        return img
    luma = img[..., 0:1] * 0.114 + img[..., 1:2] * 0.587 + img[..., 2:3] * 0.299
    sat = 1 - 0.55 * g
    out = luma + (img - luma) * sat
    cool = np.array([1 + 0.10 * g, 1 + 0.02 * g, 1 - 0.08 * g], np.float32)  # B up, R down
    return np.clip(out * cool, 0, 255)
