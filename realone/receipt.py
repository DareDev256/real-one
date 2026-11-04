"""Real Receipt (0.2.0): the hard number, and a counter that shows it.

Every render measures, per frame, how much of the picture is the artist's own filmed pixels: the artist mask in
output space (after the reframe), minus any pocket of the source room the despill cut open to show the world,
over the whole 1080x1920 frame. Everything else (the generated plate, the particles, the collage paper, the
counter itself) counts as generated. `out/<tag>_receipt.json` keeps every frame and the whole video.

`--receipt-counter` draws that number on receipt sections, set like a till receipt (label, dotted leader,
right-aligned figure, a thin share bar), white with a soft shadow, inside the 9:16 safe area, in whichever of three
places (top left, top right, lower left) the artist covers least over the shot: chosen once per shot so it never
moves mid-shot, and kept off the face in punch-ins. It is greyscale on purpose: in the receipt look colour belongs
to the filmed pixels only.
"""
from pathlib import Path

import cv2
import numpy as np

from realone.common import H, W

FONTS = ["/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
         "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf",
         "/System/Library/Fonts/Helvetica.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"]
X0, Y0 = 64, 176          # top-left of the block (inside the area a phone UI leaves alone)
BLOCK_W = 430
BLOCK_H = 130
SIZE = 40
TRACK = 3                 # letter spacing, px
PLACES = {1: (X0, Y0), 2: (W - X0 - BLOCK_W, Y0), 3: (X0, 1300)}  # top left, top right, lower left (clear of
#                                                                   the caption band a phone UI puts at the bottom)


FACE_SHARE = 0.24   # the top of the artist's box this deep is his head (realone/moments.py uses the same)
FACE_WEIGHT = 10.0  # covering his head costs this much more than covering his body


def place_for(boxes_out):
    """Counter position code (1 top left, 2 top right, 3 lower left) for a shot: the place whose block the artist's
    output-space bbox (x0, y0, x1, y1 per frame) overlaps least on average, his head weighted FACE_WEIGHT times
    (in a close shot his body covers every place, and the counter sat over his mouth: gauntlet round 1, Nov 4);
    ties go in that order."""
    def area(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
        return max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))

    best, best_cost = 1, None
    for code, (x, y) in PLACES.items():
        bx0, by0, bx1, by1 = x - 16, y - 16, x + BLOCK_W + 16, y + BLOCK_H + 16
        ov = []
        for x0, y0, x1, y1 in boxes_out:
            fy1 = y0 + FACE_SHARE * (y1 - y0)
            ov.append(area(x0, y0, x1, y1, bx0, by0, bx1, by1)
                      + FACE_WEIGHT * area(x0, y0, x1, fy1, bx0, by0, bx1, by1))
        cost = float(np.mean(ov)) if ov else 0.0
        if best_cost is None or cost < best_cost - 1e-6:
            best, best_cost = code, cost
    return best


def _font(size):
    from PIL import ImageFont
    for f in FONTS:
        if Path(f).exists():
            try:
                return ImageFont.truetype(f, size), Path(f).name
            except OSError:
                continue
    return ImageFont.load_default(size), "PIL default"


def split(filmed_frac):
    """-> (filmed %, generated %) with one decimal, summing to exactly 100.0."""
    f = round(float(filmed_frac) * 100, 1)
    return f, round(100.0 - f, 1)


class Counter:
    """Renders the receipt block for a value (layers cached by its text: a value repeats often) and inks it to the
    ground: white with a dark shadow on a dark world, near-black with a light halo on a light one (the acid stage's
    white rings swallowed white type). Hysteresis (switch at luma 0.58 one way, 0.48 back) so a shot cannot flicker;
    `reset()` at each render unit (a unit starts on a cut)."""

    LIGHT_ON, LIGHT_OFF = 0.58, 0.48

    def __init__(self):
        self.font, self.font_name = _font(SIZE)
        self._cache = {}
        self.mode = None

    def reset(self):
        self.mode = None

    def _text(self, draw, x, y, s, fill):
        for ch in s:
            draw.text((x, y), ch, font=self.font, fill=fill)
            x += draw.textlength(ch, font=self.font) + TRACK
        return x

    def _width(self, draw, s):
        return sum(draw.textlength(ch, font=self.font) + TRACK for ch in s) - TRACK

    def layers(self, filmed_frac):
        """-> (main alpha, dim alpha, track alpha, text): FILMED row + filled bar / GENERATED row / bar track."""
        f, g = split(filmed_frac)
        key = f"{f:.1f}"
        if key in self._cache:
            return self._cache[key]
        from PIL import Image, ImageDraw
        lh = int(SIZE * 1.18)
        h = 2 * lh + 34
        pad = 16
        size = (BLOCK_W + 2 * pad, h + 2 * pad)
        main, dim, track = (Image.new("L", size, 0) for _ in range(3))
        dm, dd, dt = ImageDraw.Draw(main), ImageDraw.Draw(dim), ImageDraw.Draw(track)
        for k, (lab, num, target) in enumerate((("FILMED", f"{f:.1f}%", dm), ("GENERATED", f"{g:.1f}%", dd))):
            y = pad + k * lh
            x_end = self._text(target, pad, y, lab, 255)
            nw = self._width(target, num)
            xn = pad + BLOCK_W - nw
            self._text(target, xn, y, num, 255)
            dot_y = y + int(SIZE * 0.78)
            x = x_end + 10
            while x < xn - 12:  # dotted leader, like a till receipt
                target.ellipse((x, dot_y, x + 3, dot_y + 3), fill=255)
                x += 11
        by = pad + 2 * lh + 10
        fill = int(round(BLOCK_W * f / 100))
        dt.rectangle((pad + fill, by, pad + BLOCK_W, by + 5), fill=255)
        dm.rectangle((pad, by, pad + fill, by + 5), fill=255)
        out = tuple(np.asarray(im, np.float32) / 255 for im in (main, dim, track)) + (
            f"FILMED {f:.1f}%  GENERATED {g:.1f}%",)
        self._cache[key] = out
        return out

    def draw(self, frame_f, filmed_frac, place=1):
        """Composite onto a float32 0..255 BGR frame in place at PLACES[place]. -> the text drawn.
        Ink: the one that fewer pixels under the block fight (white vs bright pixels, near-black vs dark ones),
        switched with hysteresis. On a mixed ground (rings, stripes) a soft feathered pad of the opposite tone goes
        behind the block, only as strong as the conflict; on a plain ground there is no pad, just type."""
        a_main, a_dim, a_track, text = self.layers(filmed_frac)
        ph, pw = a_main.shape
        px, py = PLACES.get(int(place), PLACES[1])
        x0, y0 = px - 16, py - 16
        roi = frame_f[y0:y0 + ph, x0:x0 + pw]
        lum = (roi[..., 0] * 0.0722 + roi[..., 1] * 0.7152 + roi[..., 2] * 0.2126) / 255
        bright, dark = float((lum > 0.6).mean()), float((lum < 0.4).mean())
        score = bright - dark
        if self.mode is None:
            self.mode = "dark" if score > 0 else "light"
        elif self.mode == "light" and score > 0.15:
            self.mode = "dark"
        elif self.mode == "dark" and score < -0.15:
            self.mode = "light"
        light = self.mode == "light"
        ink, dim_ink, track_ink, halo_ink = (255, 185, 95, 0) if light else (18, 60, 165, 255)
        conflict = bright if light else dark
        pad_a = min(0.62, conflict * 1.4)
        if pad_a > 0.02:
            if getattr(self, "_pad", None) is None or self._pad.shape != a_main.shape:
                m = np.zeros(a_main.shape, np.float32)
                m[10:-10, 6:-6] = 1
                self._pad = cv2.GaussianBlur(m, (0, 0), 9)
            pad = self._pad * pad_a
            roi[:] = roi * (1 - pad[..., None]) + halo_ink * pad[..., None]
        alpha = np.maximum(np.maximum(a_main, a_dim), a_track)
        halo = cv2.GaussianBlur(alpha, (0, 0), 5) * 0.55
        roi[:] = roi * (1 - halo[..., None]) + halo_ink * halo[..., None]
        col = a_main * ink + a_dim * dim_ink + a_track * track_ink
        cov = np.clip(a_main + a_dim + a_track, 0, 1)
        lumc = col / np.maximum(cov, 1e-6)
        roi[:] = roi * (1 - cov[..., None]) + lumc[..., None] * cov[..., None]
        return text


def section_summary(plan, filmed, s0):
    """Per section: mean filmed % over its frames."""
    out = []
    for s in plan["sections"]:
        a = int(round((s["start_s"] - s0) * 30))
        b = int(round((s["end_s"] - s0) * 30))
        a, b = max(0, a), min(len(filmed), b)
        if b > a:
            f, g = split(float(np.nanmean(filmed[a:b])))
            out.append({"section": s.get("index", -1), "role": s["role"], "frames": [a, b - 1],
                        "filmed_pct": f, "generated_pct": g})
    return out


def write_json(path, plan, filmed_by_output, looks_by_output, reveal, counter, s0, write):
    """The receipt: per output, every frame's filmed share and the whole video's, per section too."""
    doc = {"definition": ("filmed = the artist mask in output space (after the reframe) minus any pocket the "
                          "despill cut open, over the 1080x1920 frame; generated = everything else (plate, "
                          "particles, collage paper, overlays). Per frame and whole video, in percent."),
           "fps": 30, "start_s": s0, "receipt_reveal": reveal, "counter_drawn": counter, "outputs": {}}
    for out, filmed in filmed_by_output.items():
        f, g = split(float(np.nanmean(filmed)))
        doc["outputs"][out] = {
            "video": {"filmed_pct": f, "generated_pct": g},
            "sections": section_summary(plan, filmed, s0),
            "frames": {"filmed_pct": [round(float(v) * 100, 2) for v in filmed],
                       "look": looks_by_output[out]},
        }
    return write(path, doc)
