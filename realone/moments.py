"""Lyric moments (`--moments`): two or three words in a section get a visual hit on the frame they land.

James, Nov 2: "get burnt up ... shouldn't this thing be generating fire?", and the bar he set: "what would Kendrick
Lamar do?". Not effects everywhere. A few conceptual moments a section, picked from what the words say. The director
(Muse Wonder) picks the words and the effect in the treatment (`moments`); with no director pick, words about fire
ignite on their own.

- ignite: an ember burst rises from behind the artist, the world takes a warm flash, and a glow climbs from his
  feet. The burst and the glow are drawn into the world layer (a fifth of the embers into the front layer, the one
  the renderer already draws over the artist), so the artist's pixels are never generated and the Real Receipt
  counts every ember as generated.
- slam: the word itself slams in huge in front of the artist on its beat, holds, and fades. It covers part of him
  for under a second, so it is drawn on every look except the Real Receipt, whose filmed share is measured from his
  mask: the receipt stays a true count.

Deterministic: an ember's path is a function of its moment's seed and the frames since the word landed, so a
re-render is the same video. Timing comes from the lyric schedule (realone/lyrictext.py): a moment lands on the
beat its word lands on."""
import numpy as np

from realone.common import H as H_, W as W_

EFFECTS = ("ignite", "slam")
MAX_MOMENTS = 3         # distinct words picked
MAX_HITS = 4            # moments landing in one render, a hook's repeats included (critic round 2, Nov 2: "add
                        # 3-4 of these hits"; one per ~5 s of a 20 s hook)
MIN_GAP = 45            # frames between two landings (1.5 s): a repeat sung sooner is skipped, so hits stay hits
IGNITE_WORDS = frozenset(
    "fire fires burn burns burnt burned burning flame flames blaze blazing lit heat ash ashes inferno torch "
    "lighter matches".split())
IGNITE_FRAMES = 42      # the burst's life, 1.4 s at 30 fps
FLASH_FRAMES = 8        # the warm flash on the world
SLAM_FRAMES = 20        # the slammed word, 0.67 s
SLAM_FADE = 6           # its last frames fade out
SLAM_SCALE = (1.28, 1.1, 1.03)  # its first frames: in from bigger, landing at 1.0
SLAM_Y = (0.72, 0.18)   # where the word may sit (share of frame height): the lower third first, else the top
FACE_SHARE = 0.24       # the top of the artist's box this deep is his head: a slam never covers it
EMBERS = 230
WARM = np.float32([-0.22, 0.06, 0.42])  # BGR gain at full flash: blue down, red up


# Words that never go on screen by default (Meta employees judge the hackathon; Savv's drill verse, Nov 2): the
# director may not pick them as moments, and a lyric carrying any of them keeps lyrics-in-the-world off unless
# --lyrics-in-world forces it.
VIOLENCE = frozenset("gun guns glock glocks shoot shooting shot shots kill killed killing murder blood bullet bullets "
                     "stick sticks opp opps smoke smoked body bodies drill strap draco".split())


class MomentError(ValueError):
    pass


def _norm(w):
    w = str(w).lower().replace("’", "'")
    return "".join(ch for ch in w if ch.isalnum() or ch == "'").strip("'")


def pick(lines, chosen=None, most=MAX_MOMENTS, repeats=True):
    """lines: the render's lyric phrases ([[word, ...], ...], from lyrictext.schedule). chosen: the director's
    [{"word", "effect"}, ...] in order, None when no director chose (then fire words ignite), [] when it chose none.
    -> [(phrase id, word index, effect, word)] for at most `most` distinct picks; with `repeats` a pick lands every
    time its word is sung (a hook that repeats carries its moment each time: the director is told so), and
    `profile` spaces the landings (MIN_GAP) and caps them (MAX_HITS)."""
    out, used, picks = [], set(), 0
    if chosen is not None:
        for ch in chosen:
            eff, want = ch.get("effect"), _norm(ch.get("word", ""))
            if eff not in EFFECTS or not want:
                continue
            spots = [(g, k) for g, ws in enumerate(lines) for k, w in enumerate(ws)
                     if _norm(w) == want and (g, k) not in used]
            if not spots:
                continue  # the word is not sung inside this render
            for g, k in (spots if repeats else spots[:1]):
                used.add((g, k))
                out.append((g, k, eff, lines[g][k]))
            picks += 1
            if picks >= most:
                break
        return out
    for g, ws in enumerate(lines):
        for k, w in enumerate(ws):
            if _norm(w) in IGNITE_WORDS:
                out.append((g, k, "ignite", w))
                if len(out) >= most:
                    return out
    return out


def violent_words(text):
    """The words of `text` on the VIOLENCE list, normalised."""
    return sorted({_norm(w) for w in str(text).split()} & VIOLENCE)


def slam_y(box, word_h):
    """The word's centre row: the first SLAM_Y place whose band misses the artist's head (the top FACE_SHARE of his
    box, output px), else the one that covers it least. The first render put it across his face (Nov 2)."""
    if box is None or np.isnan(box[0]):
        return int(SLAM_Y[0] * H_)
    f0, f1 = box[1], box[1] + FACE_SHARE * (box[3] - box[1])

    def overlap(cy):
        top, bot = cy * H_ - word_h / 2, cy * H_ + word_h / 2
        return max(0.0, min(bot, f1) - max(top, f0))

    best = min(SLAM_Y, key=lambda cy: (overlap(cy) > 0, overlap(cy)))
    return int(best * H_)


def slam_looks(frames, painter, j, age, box=None):
    """Slam word j onto every look but the Real Receipt, whose filmed share is measured from the artist's mask: a
    word covering him there would make the receipt's count untrue. box: the artist in output px (keeps his face
    clear)."""
    return {lk: fr if lk == "receipt" else painter.apply(fr, j, age, box) for lk, fr in frames.items()}


def land_frame(line_id, shown, gid, k):
    """The first frame on which word k of phrase gid is visible, or None."""
    idx = np.nonzero((np.asarray(line_id) == gid) & (np.asarray(shown) >= k + 1))[0]
    return int(idx[0]) if idx.size else None


def profile(moments, line_id, shown, n, boxes=None):
    """-> (arrays, designed frames, report). arrays: mo_ign (frames since an ignite landed, -1 none), mo_ign_id
    (which moment), mo_ign_box (the artist's box when it landed, source px, for where the burst starts), mo_slam
    (index into the slam words, -1 none), mo_slam_age; plus the slam words list. designed: the frames whose whole
    picture a moment changes on purpose (the flash, the slam), for the render's cut check."""
    ign, ign_id = np.full(n, -1, np.int16), np.full(n, -1, np.int16)
    ign_box = np.full((n, 4), np.nan, np.float32)
    slam, slam_age = np.full(n, -1, np.int16), np.zeros(n, np.int16)
    words, designed, report = [], set(), []
    landed = sorted((f0, m) for m in moments if (f0 := land_frame(line_id, shown, m[0], m[1])) is not None)
    kept, last = [], -10 ** 9
    for f0, m in landed:  # spaced and capped, in the order they are sung
        if f0 - last >= MIN_GAP and len(kept) < MAX_HITS:
            kept.append((f0, m))
            last = f0
    for j, (f0, (g, k, eff, w)) in enumerate(kept):
        if eff == "ignite":
            box = boxes[f0] if boxes is not None else np.array([W_ * 0.3, H_ * 0.2, W_ * 0.7, H_ * 0.95])
            if np.isnan(box[0]):
                box = np.array([W_ * 0.3, H_ * 0.2, W_ * 0.7, H_ * 0.95])
            for i in range(f0, min(n, f0 + IGNITE_FRAMES)):
                ign[i], ign_id[i], ign_box[i] = i - f0, j, box
        else:
            words.append(w)
            for i in range(f0, min(n, f0 + SLAM_FRAMES)):
                slam[i], slam_age[i] = len(words) - 1, i - f0
        designed.update(range(f0, min(n, f0 + 2)))
        report.append({"word": w, "effect": eff, "frame": f0})
    arrays = {"mo_ign": ign, "mo_ign_id": ign_id, "mo_ign_box": ign_box, "mo_slam": slam, "mo_slam_age": slam_age}
    return arrays, words, designed, report



def box_out(box, crop):
    """The artist's source-px box mapped into the output frame through the shot's crop (x0, y0, cw, ch)."""
    x0, y0, cw, ch = crop
    z = W_ / cw
    return ((box[0] - x0) * z, (box[1] - y0) * z, (box[2] - x0) * z, (box[3] - y0) * z)


class EmberBurst:
    """One ignite's embers: spawned across the artist's base and up his body, rising fast and slowing, cooling from
    white-yellow to orange to red as they die. A fifth are near the camera (the front layer, drawn over him)."""

    def __init__(self, seed, n=EMBERS):
        r = np.random.default_rng(1000 + seed)
        self.n = n
        self.u = r.normal(0, 0.55, n)               # across, in artist widths from his centre
        self.v = r.uniform(0.45, 1.0, n) ** 0.6     # up the body, 1 = feet (most start low)
        self.vy = r.uniform(420, 1400, n)           # px/s up at the start
        self.vx = r.normal(0, 190, n)
        self.life = r.uniform(0.5, 1.0, n) * IGNITE_FRAMES / 30.0
        self.delay = r.exponential(0.06, n).clip(0, 0.3)  # the burst spreads over a few frames
        self.size = r.uniform(1.6, 3.4, n) * np.where(r.uniform(0, 1, n) > 0.88, 2.4, 1.0)  # a few big ones
        self.ph = r.uniform(0, 6.28, n)
        self.front = r.uniform(0, 1, n) > 0.8

    def draw(self, age, box, fps=30):
        """-> (back, front) float32 (H, W, 3) 0..255 BGR, for `age` frames after the word landed, `box` the artist
        in output px."""
        import cv2
        back = np.zeros((H_, W_, 3), np.float32)
        front = np.zeros((H_, W_, 3), np.float32)
        t = age / fps - self.delay
        alive = (t >= 0) & (t < self.life)
        if not alive.any():
            return back, front
        x0, y0, x1, y1 = box
        bw = max(90.0, x1 - x0)
        drag = 1.7
        rise = self.vy * (1 - np.exp(-drag * np.maximum(t, 0))) / drag
        x = (x0 + x1) / 2 + self.u * bw + self.vx * np.maximum(t, 0) + 16 * np.sin(9 * t + self.ph)
        y = y0 + self.v * (y1 - y0) - rise
        vy_now = self.vy * np.exp(-drag * np.maximum(t, 0))
        heat = np.clip(1 - t / self.life, 0, 1)
        flick = 0.7 + 0.3 * np.sin(17 * t + self.ph)
        for i in np.nonzero(alive)[0]:
            h = heat[i]
            # white-yellow when young, orange, then deep red (BGR)
            col = (np.float32([0.45, 0.85, 1.0]) * h ** 4 + np.float32([0.06, 0.40, 1.0]) * (1 - h ** 4) * h
                   + np.float32([0.02, 0.10, 0.75]) * (1 - h))
            a = 255 * (0.35 + 0.65 * h) * flick[i]
            near = self.front[i]
            s = self.size[i] * (2.2 if near else 1.0)
            tgt = front if near else back
            # a short streak along its motion (the shutter is open while it rises)
            sx, sy = self.vx[i] / fps * 0.9, -vy_now[i] / fps * 0.9
            cv2.line(tgt, (int(x[i]), int(y[i])), (int(x[i] - sx), int(y[i] - sy)),
                     tuple(float(c) * a for c in col), max(1, int(round(s))), cv2.LINE_AA)
        back = cv2.GaussianBlur(back, (0, 0), 1.2)
        back += cv2.GaussianBlur(back, (0, 0), 6) * 2.2 + cv2.GaussianBlur(back, (0, 0), 22) * 3.0  # the heat haze
        front = cv2.GaussianBlur(front, (0, 0), 2.6) * 1.3
        return back, front


def flash_amount(age):
    """The warm flash: a hard hit on the word that settles into a glow for the rest of the burst."""
    hit = (1 - min(age, FLASH_FRAMES) / FLASH_FRAMES) ** 2
    glow = max(0.0, 1 - age / IGNITE_FRAMES) ** 0.5
    return 0.45 * hit + 0.4 * glow


def ignite(world, front, age, box, burst):
    """Frame `age` of an ignite on the world layer (float32 0..255): the warm grade, the floor glow behind the artist,
    the embers. -> (world, front)."""
    g = flash_amount(age)
    world = world * (1 + g * WARM)
    x0, y0, x1, y1 = box
    cx, bw = (x0 + x1) / 2, max(90.0, x1 - x0)
    yy, xx = np.ogrid[:H_, :W_]
    # the glow climbs from his feet: an ellipse wider than him, its height growing over the first frames
    reach = (0.35 + 0.65 * min(1.0, age / 10)) * max(200.0, (y1 - y0) * 0.9)
    glow = np.exp(-(((xx - cx) / (1.3 * bw)) ** 2) - (((yy - y1) / reach) ** 2)).astype(np.float32)
    world = world + glow[..., None] * np.float32([30, 110, 255]) * (0.65 * g + 0.15)
    b, f = burst.draw(age, box)
    world = world + b
    front = front + f
    return np.clip(world, 0, 255), front


class SlamPainter:
    """A word slammed in front of the artist: Helvetica Neue Condensed Black, white with a black edge and a soft
    shadow, as wide as the frame allows, upper third. Rendered once per word, scaled per frame."""

    def __init__(self, words, font_spec=None):
        from realone import lyrictext as LT
        self.words = [str(w).upper().strip(",.!?;:") for w in words]
        self.font_spec = font_spec or LT.DEFAULT_FONT
        self.cache = {}

    def _base(self, j):
        if j in self.cache:
            return self.cache[j]
        from PIL import Image, ImageDraw, ImageFilter
        from realone import lyrictext as LT
        word = self.words[j]
        size = 400
        font = LT._font(self.font_spec, size)
        while size > 60:
            font = LT._font(self.font_spec, size)
            l, t, r, b = font.getbbox(word)
            if r - l <= 0.78 * W_ and b - t <= 0.22 * H_:
                break
            size -= 12
        l, t, r, b = font.getbbox(word)
        stroke = max(2, size // 70)
        pad = stroke * 6
        cw, ch = r - l + 2 * pad, b - t + 2 * pad
        img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        sh = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        pad = max(pad, size // 6)
        cw, ch = r - l + 2 * pad, b - t + 2 * pad
        img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        sh = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((pad - l, pad - t + size // 40), word, font=font, fill=(0, 0, 0, 200),
                                stroke_width=size // 30, stroke_fill=(0, 0, 0, 200))
        sh = sh.filter(ImageFilter.GaussianBlur(size // 18))
        img.alpha_composite(sh)
        ImageDraw.Draw(img).text((pad - l, pad - t), word, font=font, fill=(255, 255, 255, 255),
                                 stroke_width=stroke, stroke_fill=(20, 20, 20, 255))
        arr = np.asarray(img).astype(np.float32) / 255
        self.cache[j] = arr
        return arr

    def apply(self, frame, j, age, box=None):
        """frame: uint8 (H, W, 3) BGR -> the frame with word j slammed in, `age` frames after it landed, clear of
        the artist's face when his box (output px) is given."""
        import cv2
        if j < 0 or age >= SLAM_FRAMES:
            return frame
        base = self._base(j)
        sc = SLAM_SCALE[age] if age < len(SLAM_SCALE) else 1.0
        img = cv2.resize(base, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        fade = 1.0 if age < SLAM_FRAMES - SLAM_FADE else (SLAM_FRAMES - age) / (SLAM_FADE + 1)
        h, w = img.shape[:2]
        cx, cy = W_ // 2, slam_y(box, base.shape[0])
        xa, ya = cx - w // 2, cy - h // 2
        x0, y0, x1, y1 = max(0, xa), max(0, ya), min(W_, xa + w), min(H_, ya + h)
        if x1 <= x0 or y1 <= y0:
            return frame
        part = img[y0 - ya:y1 - ya, x0 - xa:x1 - xa]
        al = part[..., 3:4] * fade
        rgb = part[..., 2::-1] * 255  # RGBA -> BGR
        out = frame.copy()
        reg = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = (reg * (1 - al) + rgb * al).clip(0, 255).astype(np.uint8)
        return out
