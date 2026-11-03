"""Living world: the generated world is one still image, so its lamps sat dead while the artist moved (James, Nov 2,
on the Savv render: "even how they had the lamps, the background should technically be active ... moving"). Muse
Wonder's own world prompt asked for "flickering amber security lamps"; a still cannot flicker.

What moves, on the world layer before the artist is composited (his pixels are never touched):
- the lamps: every light in the frame, warm or any colour brighter than its surroundings (found per frame, so it
  works through the tracked stage's camera and every punch-in) breathes with a slow per-region flicker and swells on the kick: a downbeat, smaller on the
  backbeat;
- a dying lamp: on some bars one region stutters out and back, the way a failing security light does;
- haze: a drifting fog texture, lit only where the lamps' glow reaches, so the light has a volume that moves.

Everything is a function of the WORLD clock `wt` (the render's world time, which stands still in a Hook Freeze), so
a frozen world freezes its lamps too and the freeze proof stays true, and a re-render is the same video."""
import numpy as np

from realone.common import H, W

SCALE = 4               # work on a 1/4 size copy: the glow is soft, and this keeps it to a few ms a frame
WARM_MIN = 0.12         # red over blue, as a share of red: what counts as a warm light
LUMA_LO, LUMA_HI = 0.42, 0.78
TOPHAT_SIGMA = 10       # small px: a light is brighter than its surroundings this wide
FLICKER = 0.22          # per-region flicker depth (x1 +/- this)
FLICKER_HZ = 2.2        # how fast the flicker field changes
PULSE_DOWN, PULSE_BACK = 0.85, 0.4  # the lamps' swell on the kick
PULSE_DECAY_S = 0.16
FOG_STRENGTH = 0.9
FOG_DRIFT = (-95.0, -22.0)  # px/s the haze drifts (full-size px), left and slightly up
DROPOUT_EVERY = 3       # about one bar in this many has a dying lamp


def _value_noise(seed, gw, gh, w, h):
    """A smooth random field: a gw x gh grid of random values, bicubic up to w x h."""
    import cv2
    g = np.random.default_rng(seed).random((gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


class LivingWorld:
    """One per render worker. `beats`, `downbeats`: song times (s); `beats_per_bar` for the backbeats."""

    def __init__(self, beats, downbeats, beats_per_bar=4, seed=11):
        import cv2
        self.beats = np.asarray(beats or [], float)
        self.downs = np.asarray(downbeats or [], float)
        self.bpb = int(beats_per_bar or 4)
        self.seed = seed
        self.w, self.h = W // SCALE, H // SCALE
        # the fog: three octaves of value noise, wider than the frame so it can drift and wrap without a seam
        fw, fh = self.w * 2, self.h
        fog = (_value_noise(seed, 8, 10, fw, fh) * 0.55 + _value_noise(seed + 1, 20, 24, fw, fh) * 0.3
               + _value_noise(seed + 2, 48, 56, fw, fh) * 0.15)
        fog = np.concatenate([fog, fog[:, ::-1]], axis=1)[:, :fw]  # mirrored so the wrap point is continuous
        self.fog = cv2.GaussianBlur(fog, (0, 0), 3)
        self._cells = {}

    # ---------------------------------------------------------------- time fields
    def pulse(self, t):
        """The kick envelope at song time t: 1 on a downbeat, smaller on the backbeat, decaying."""
        if self.beats.size == 0:
            return 0.0
        k = int(np.searchsorted(self.beats, t + 1e-6)) - 1
        if k < 0:
            return 0.0
        tb = self.beats[k]
        is_down = bool(self.downs.size) and np.min(np.abs(self.downs - tb)) < 1e-3
        if is_down:
            amp = PULSE_DOWN
        else:
            d0 = self.downs[self.downs <= tb + 1e-3]
            pos = int(np.sum((self.beats > d0[-1] + 1e-3) & (self.beats <= tb + 1e-3))) if d0.size else 1
            amp = PULSE_BACK if (self.bpb == 4 and pos in (1, 3)) else 0.0
        return float(amp * np.exp(-(t - tb) / PULSE_DECAY_S))

    def _cell(self, k):
        if k not in self._cells:
            if len(self._cells) > 8:
                self._cells.clear()
            self._cells[k] = _value_noise(self.seed * 1000 + k, 5, 8, self.w, self.h)
        return self._cells[k]

    def flicker(self, t):
        """Per-region flicker field (h, w) around 1, smooth in time and space."""
        x = t * FLICKER_HZ
        k = int(np.floor(x))
        f = x - k
        f = f * f * (3 - 2 * f)
        field = self._cell(k) * (1 - f) + self._cell(k + 1) * f
        return 1 + FLICKER * (2 * field - 1)

    def dropout(self, t):
        """The dying lamp: on about one bar in DROPOUT_EVERY, one region stutters out for a quarter second."""
        out = np.ones((self.h, self.w), np.float32)
        if self.downs.size < 2:
            return out
        b = int(np.searchsorted(self.downs, t + 1e-6)) - 1
        if b < 0 or b + 1 >= self.downs.size:
            return out
        r = np.random.default_rng(self.seed * 7919 + b)
        if r.integers(DROPOUT_EVERY) != 0:
            return out
        bar = self.downs[b + 1] - self.downs[b]
        t0 = self.downs[b] + bar * r.uniform(0.35, 0.7)
        u = (t - t0) / 0.26
        if not 0 <= u < 1:
            return out
        depth = (0.85, 0.2, 0.9, 0.55, 0.95, 0.3, 0.0)[int(u * 7)]  # off, half back, off, flick, off, back
        cx, cy = r.uniform(0.1, 0.9) * self.w, r.uniform(0.15, 0.75) * self.h
        yy, xx = np.ogrid[:self.h, :self.w]
        blob = np.exp(-(((xx - cx) / (0.16 * self.w)) ** 2 + ((yy - cy) / (0.12 * self.h)) ** 2))
        return (1 - depth * blob).astype(np.float32)

    # ---------------------------------------------------------------- the frame
    def lamp_mask(self, small):
        """small: float32 (h, w, 3) BGR 0..255 -> (h, w) 0..1, the lights: warm bright pixels (sodium and tungsten
        lamps), or any colour brighter than its surroundings (a top-hat: neon, fluorescents and their reflections;
        warm-only found nothing in the stand-in's blue world, Nov 2). A bright sky is not a light: it is not
        brighter than what is around it."""
        import cv2
        b, g, r = small[..., 0] / 255, small[..., 1] / 255, small[..., 2] / 255
        luma = 0.114 * b + 0.587 * g + 0.299 * r
        warm = np.clip((r - b) / (r + 1e-3) - WARM_MIN, 0, 1) / (1 - WARM_MIN)
        bright = np.clip((luma - LUMA_LO) / (LUMA_HI - LUMA_LO), 0, 1)
        lum = luma.astype(np.float32)
        tophat = (np.clip((lum - cv2.GaussianBlur(lum, (0, 0), TOPHAT_SIGMA) - 0.06) / 0.18, 0, 1)
                  * np.clip((lum - 0.35) / 0.3, 0, 1))
        return np.maximum(bright * np.sqrt(warm), tophat).astype(np.float32)

    def apply(self, world, t):
        """world: float32 (H, W, 3) 0..255, the world layer this frame; t: the world clock (song seconds).
        -> the living world layer (a new array)."""
        import cv2
        small = cv2.resize(world, (self.w, self.h), interpolation=cv2.INTER_AREA)
        m = self.lamp_mask(small)
        if m.max() < 0.05:
            return world  # no lamps in this world: nothing to bring to life
        m = np.clip(m / max(0.35, float(np.percentile(m, 99.7))), 0, 1)  # the brightest lamps reach 1
        k = self.flicker(t) * self.dropout(t)
        env = self.pulse(t)
        lamps = small * m[..., None]
        glow = (cv2.GaussianBlur(lamps, (0, 0), 4) * 1.0 + cv2.GaussianBlur(lamps, (0, 0), 14) * 1.3)
        # the lamps follow the flicker; their bloom follows it and swells on the kick (between kicks only the
        # flicker and the haze move, so the world stays the still Wonder made, breathing)
        delta = lamps * (k - 1)[..., None] + glow * (0.6 * (k - 1) + 1.5 * env)[..., None]
        # haze lit by the lamps' reach, drifting with the world clock
        reach = cv2.GaussianBlur(m, (0, 0), 18)
        reach = reach / (reach.max() + 1e-6)
        fw = self.fog.shape[1]
        ox = int(round((FOG_DRIFT[0] * t) / SCALE)) % fw
        oy = int(round((FOG_DRIFT[1] * t) / SCALE))
        fog = np.roll(np.roll(self.fog, -ox, axis=1), -oy, axis=0)[:, :self.w]
        fog = np.clip((fog - 0.35) * 2.2, 0, 1)
        tint = cv2.GaussianBlur(small * m[..., None], (0, 0), 20) / (cv2.GaussianBlur(m, (0, 0), 20)[..., None] + 1e-3)
        delta += (fog * reach * (0.6 + 0.4 * k) * FOG_STRENGTH)[..., None] * np.clip(tint, 0, 255) * (1 + 0.8 * env)
        up = cv2.resize(delta, (W, H), interpolation=cv2.INTER_LINEAR)
        return np.clip(world + up, 0, 255)
