"""Lyrics in the world: the section's lyric lines written into the scene behind the artist, a word on each beat.

Why it exists (James, Nov 2): viral music clips carry the lyrics on screen. Real One already has the artist's mask on
every frame and, on the tracked stage, the world filmed through his own camera, so the words can live IN the world:
painted on the backdrop, moving with the camera (the dolly zoom grows them with the background), and he walks in
front of them. They are drawn into the world layer before the artist is composited, so his pixels are never touched
and the Real Receipt counts every letter as generated.

Timing, two ways. With word times (`--word-times`, scripts/align_lyrics.py: Demucs isolates the vocal and MMS forced
alignment places the artist's own words on it), each word lands when it is sung and each phrase stays up until the
next one starts. Without them, the beat grid: Voice Transcribe gives a time per turn, not per word, so each lyric
line gets an equal share of its section's bars (starting on a downbeat) and its words land on successive beats
(half-beats when a line has more words than beats). That reads as "on the beat" for a hook sung a line a bar, and
runs ahead of a fast verse (song B's verse ran out at bar 20 of 24, Nov 4); the report says which timing was used.

The world layer's freeze key is taken before the words are blended in (realone/units.py), so the Hook Freeze still
proves the world itself stood still while the lyrics carry on with the vocal."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

DEFAULT_FONT = ("/System/Library/Fonts/HelveticaNeue.ttc", 9)  # Helvetica Neue Condensed Black
POP_FRAMES = 4          # the newest word slides up and fades in over this many frames
MAX_ROWS = 3


class LyricError(ValueError):
    pass


def words_of(line):
    return [w for w in str(line).replace("’", "'").split() if w.strip()]


PHRASE_WORDS = 4            # a lyric line is shown as phrases of at most this many words, a bar or so each
LEAD_S = 0.05               # with word times: a word lands this much before it is sung (the eye is slower than the ear)
HOLD_S = 1.2                # ... and a phrase stays up at most this long after its last word ends


def phrases(words, most=PHRASE_WORDS):
    """Split a line into near-equal phrases of at most `most` words ("lights on at four / and I'm still holding"):
    shorter-lived signs drift less as the camera moves, and a phrase reads at a glance."""
    if not words:
        return []
    k = -(-len(words) // most)
    size = -(-len(words) // k)
    return [words[j:j + size] for j in range(0, len(words), size)]


def load_word_times(path, section_lines):
    """The aligner's JSON (scripts/align_lyrics.py) -> {section index: [[(start_s, end_s) per word] per line]}.
    Refuses unless every line is the same text the render will draw (a file aligned to another lyric would put
    words on the wrong syllables), with one time per word of the render's own split."""
    import json
    from pathlib import Path
    try:
        rec = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise LyricError(f"--word-times {path}: cannot read it ({e})") from None
    got = {(ln.get("section"), ln.get("line")): ln for ln in rec.get("lines", [])}
    out = {}
    for sec, lines in (section_lines or {}).items():
        rows = []
        for j, text in enumerate(lines):
            ln = got.get((sec, j))
            if ln is None or ln.get("text") != text:
                raise LyricError(f"--word-times {path}: section {sec} line {j} is not the treatment's line "
                                 f"{text!r}; re-run scripts/align_lyrics.py on this treatment")
            ws = words_of(text)
            if len(ln.get("words", [])) != len(ws):
                raise LyricError(f"--word-times {path}: section {sec} line {j} has {len(ln.get('words', []))} "
                                 f"times for {len(ws)} words")
            rows.append([(float(x["start"]), float(x["end"])) for x in ln["words"]])
        if rows:
            out[sec] = rows
    return out


def _sung_phrases(lines, times):
    """One section's lines and their word times -> [(phrase words, starts, ends)] in sung order."""
    items = []
    for text, tt in zip(lines, times):
        k = 0
        for ph in phrases(words_of(text)):
            seg = tt[k:k + len(ph)]
            k += len(ph)
            items.append((ph, [x[0] for x in seg], [x[1] for x in seg]))
    return items


def schedule(section_lines, timeline, s0, n, fps=30, word_times=None, sections_out=None):
    """section_lines: {section index: [lyric line, ...]}; timeline: realone.audio.analyze output (beats, downbeats,
    sections); s0: song time of the render's frame 0; n: frames; word_times: load_word_times() output, or None for
    the beat grid; sections_out: a list that receives each returned phrase's section index (a lyric moment the
    director picked for one section must land only there).
    -> (lines: [[word, ...], ...], line_id int16[n] (-1 none), shown int16[n] words visible, age int16[n] frames
    since the newest visible word landed)."""
    beats = np.asarray(timeline.get("beats") or [], float)
    downs = np.asarray(timeline.get("downbeats") or [], float)
    if beats.size < 2 and not word_times:
        raise LyricError("the timeline has no beat grid; lyrics are placed on beats")
    lines, line_id = [], np.full(n, -1, np.int16)
    shown, age = np.zeros(n, np.int16), np.zeros(n, np.int16)
    t = s0 + np.arange(n) / fps

    def paint(ws, lands, l0, l1):
        inside = (t >= l0) & (t < l1)
        if not inside.any():
            return  # this line falls outside the rendered frames
        gid = len(lines)
        lines.append(ws)
        if sections_out is not None:
            sections_out.append(sec["index"])
        for i in np.nonzero(inside)[0]:
            k_shown = int(np.searchsorted(lands, t[i] + 1e-6, side="right"))
            if k_shown == 0:
                continue
            line_id[i], shown[i] = gid, k_shown
            age[i] = int(round((t[i] - lands[k_shown - 1]) * fps))

    for sec in timeline.get("sections", []):
        a, b = float(sec["start_s"]), float(sec["end_s"])
        if word_times and sec["index"] in word_times:
            items = _sung_phrases(section_lines.get(sec["index"], []), word_times[sec["index"]])
            for k, (ws, st, en) in enumerate(items):
                lands = np.asarray(st, float) - LEAD_S
                nxt = items[k + 1][1][0] - LEAD_S if k + 1 < len(items) else b
                l1 = min(nxt, en[-1] + HOLD_S)
                if l1 > lands[0]:
                    paint(ws, lands, float(lands[0]), float(l1))
            continue
        ls = [ph for x in section_lines.get(sec["index"], []) for ph in phrases(words_of(x))]
        if not ls:
            continue
        if b <= t[0] or a >= t[-1] + 1 / fps:
            continue  # not in this render
        bars = [d for d in downs if a - 1e-3 <= d < b - 1e-3] or [a]
        if len(bars) < len(ls):  # more phrases than bars: share the section out by beats instead
            bars = [x for x in beats if a - 1e-3 <= x < b - 1e-3] or [a]
        per = max(1, len(bars) // len(ls))
        for k, ws in enumerate(ls):
            if k * per >= len(bars):
                break  # more lines than bars: the rest do not fit this section
            l0 = bars[k * per]
            l1 = bars[(k + 1) * per] if (k + 1) * per < len(bars) and k < len(ls) - 1 else b
            bt = beats[(beats >= l0 - 1e-3) & (beats < l1 - 1e-3)]
            if bt.size == 0:
                bt = np.array([l0])
            if len(ws) > bt.size:  # half-beats when the line is busier than its beats
                mids = (bt[:-1] + bt[1:]) / 2 if bt.size > 1 else np.array([])
                bt = np.sort(np.concatenate([bt, mids, [bt[-1] + (bt[-1] - bt[-2]) / 2 if bt.size > 1 else l0]]))
            lands = bt[: len(ws)] if bt.size >= len(ws) else np.linspace(l0, l1, len(ws), endpoint=False)
            paint(ws, lands, l0, l1)
    return lines, line_id, shown, age


BEAT_TIMING = "on the beat grid (a word per beat), not aligned to the vocal"


def for_render(section_lines, timeline, s0, n, fps=30, word_times=None, word_source=None):
    """The render's entry: refuses with a reason instead of rendering a video with no words in it.
    -> (lines, {"lyr_line", "lyr_shown", "lyr_age"} arrays, a report for the shot plan)."""
    if not section_lines or not any(section_lines.values()):
        raise LyricError("--lyrics-in-world needs lyric lines: pass --from-treatment (the director's treatment "
                         "carries each section's lyric)")
    line_section = []
    lines, line, shown, age = schedule(section_lines, timeline, s0, n, fps, word_times, line_section)
    if not (line >= 0).any():
        raise LyricError("--lyrics-in-world: none of the treatment's lyric lines falls inside the rendered range")
    timing = (f"each word when it is sung ({word_source or 'word times'}), {LEAD_S:.2f} s early"
              if word_times else BEAT_TIMING)
    report = {"lines": len(lines), "words": sum(len(x) for x in lines), "frames_with_words": int((line >= 0).sum()),
              "timing": timing, "line_section": line_section}
    return lines, {"lyr_line": line, "lyr_shown": shown, "lyr_age": age}, report


def _font(spec, size):
    path, index = spec
    try:
        return ImageFont.truetype(path, size, index=index)
    except OSError:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)


def layout(words, width, height, font_spec=DEFAULT_FONT, size_frac=0.085, center=0.30, max_words=None,
           max_rows=MAX_ROWS):
    """Fit the whole line (all its words, so nothing jumps as words land) into at most max_rows rows (and at most
    max_words a row), 86% of the canvas wide, as big as it fits. -> (font, rows: [[(word, x, y)], ...], row_h)."""
    up = [w.upper() for w in words]
    size = int(height * size_frac)
    while size > 12:
        f = _font(font_spec, size)
        space = f.getlength(" ")
        rows, cur, cur_w = [], [], 0.0
        for w in up:
            ww = f.getlength(w)
            if cur and (cur_w + space + ww > width * 0.86 or (max_words and len(cur) >= max_words)):
                rows.append(cur)
                cur, cur_w = [], 0.0
            cur_w = cur_w + (space if cur else 0) + ww
            cur.append(w)
        rows.append(cur)
        if len(rows) <= max_rows:
            break
        size = int(size * 0.9)
    row_h = int(size * 0.98)
    top = int(height * center) - (len(rows) * row_h) // 2
    placed = []
    for r, ws in enumerate(rows):
        rw = sum(f.getlength(w) for w in ws) + f.getlength(" ") * (len(ws) - 1)
        x = (width - rw) / 2
        row = []
        for w in ws:
            row.append((w, x, top + r * row_h))
            x += f.getlength(w) + f.getlength(" ")
        placed.append(row)
    return f, placed, row_h


class LyricPainter:
    """Draws the visible words of a line as RGBA on a canvas (the world still's size on the stage, the output
    frame's in 2D). Layouts are cached per line; one frame costs a few milliseconds."""

    def __init__(self, lines, width, height, font_spec=DEFAULT_FONT, size_frac=0.085, center=0.30, max_words=None,
                 max_rows=MAX_ROWS):
        self.lines, self.w, self.h, self.font_spec = lines, int(width), int(height), font_spec
        self.size_frac, self.center, self.max_words, self.max_rows = size_frac, center, max_words, max_rows
        self._layouts = {}

    def rgba(self, line, shown, age):
        """-> (rgb float32 (h, w, 3) 0..255, alpha float32 (h, w) 0..1), or None when nothing shows."""
        if line < 0 or shown <= 0:
            return None
        if line not in self._layouts:
            self._layouts[line] = layout(self.lines[line], self.w, self.h, self.font_spec, self.size_frac,
                                         self.center, self.max_words, self.max_rows)
        f, rows, row_h = self._layouts[line]
        flat = [p for row in rows for p in row]
        text = Image.new("L", (self.w, self.h), 0)
        d = ImageDraw.Draw(text)
        for k, (word, x, y) in enumerate(flat[:shown]):
            a = 255
            dy = 0
            if k == shown - 1 and age < POP_FRAMES:  # the newest word lands: slides up and fades in
                u = (age + 1) / POP_FRAMES
                a = int(255 * (0.35 + 0.65 * u))
                dy = int(row_h * 0.18 * (1 - u))
            d.text((x, y + dy), word, font=f, fill=a)
        al = np.asarray(text, np.float32) / 255
        shadow = np.asarray(text.filter(ImageFilter.GaussianBlur(max(2, row_h // 14))), np.float32) / 255
        shadow = np.roll(np.roll(shadow, row_h // 30 + 1, axis=0), row_h // 40 + 1, axis=1)
        # white letters over a soft dark shadow: readable on any world, bright or dark
        alpha = np.clip(al + 0.55 * shadow * (1 - al), 0, 1)
        grey = 255.0 * al / np.maximum(alpha, 1e-6)  # white letters, black shadow: the colour under the coverage
        rgb = np.repeat(grey[..., None], 3, axis=2)  # three channels: cv2 warps keep them (one would be dropped)
        return rgb.astype(np.float32), alpha.astype(np.float32)


# Cards: in a medium or close shot a sign behind the artist lands behind his head and is cut in half by it (the
# Nov 2 blind critic: "text stacked directly over the forehead and headphones"), and there is no free world beside
# him. There the line comes to the front: a bold card in the lower third, or the top when his head is low, word by
# word on the beat. Wide shots keep the sign in the world. A card covers part of him, so like a slam it is never
# drawn on the Real Receipt.
CARD_FRAMINGS = ("medium", "close")
CARD_SIZE = 0.085            # letter height, share of the frame height (0.068 read as captions, Nov 2)
CARD_WORDS = 2               # words a row: a phrase stacks as a card, no stray word on a row of its own
CARD_ROWS = 2


def card_places(line_id, n_lines, framing_at, box_at):
    """-> int8 per frame: 0 the line is a sign in the world, 1 a card in the lower third, 2 a card at the top.
    Decided once per line from the frame it starts on: framing_at(i) is that frame's shot framing, box_at(i) the
    artist's box in output px (or None)."""
    from realone import moments as MO
    from realone.common import H as _H
    line_id = np.asarray(line_id)
    out = np.zeros(line_id.shape[0], np.int8)
    row_h = CARD_SIZE * _H * CARD_ROWS
    for gid in range(n_lines):
        idx = np.nonzero(line_id == gid)[0]
        if not idx.size or framing_at(int(idx[0])) not in CARD_FRAMINGS:
            continue
        y = MO.slam_y(box_at(int(idx[0])), row_h)
        out[idx] = 1 if y == int(MO.SLAM_Y[0] * _H) else 2
    return out


def card_changes(line_id, card):
    """The frames on which a front card appears, changes line or leaves (and the frame after, for the encoder's
    smear): designed whole-band changes for the render's cut check."""
    line_id, card = np.asarray(line_id), np.asarray(card)
    on = np.where((card > 0) & (line_id >= 0), line_id, -1)
    out = set()
    for i in np.nonzero(on[1:] != on[:-1])[0] + 1:
        out.update((int(i), int(i) + 1))
    return {i for i in out if i < len(on)}


class CardPainter:
    """A line as a front card: the frame's width, CARD_ROWS rows at most, each word landing with the usual pop."""

    def __init__(self, lines, font_spec=DEFAULT_FONT):
        from realone import moments as MO
        from realone.common import H as _H, W as _W
        self.p = {place: LyricPainter(lines, _W, _H, font_spec, CARD_SIZE, MO.SLAM_Y[place - 1], CARD_WORDS, CARD_ROWS)
                  for place in (1, 2)}

    def apply(self, frame, place, line, shown, age):
        got = self.p[place].rgba(line, shown, age) if place in self.p else None
        if got is None:
            return frame
        rgb, a = got
        return (frame.astype(np.float32) * (1 - a[..., None]) + rgb * a[..., None]).clip(0, 255).astype(np.uint8)


def card_looks(frames, painter, place, line, shown, age):
    """The card on every look but the Real Receipt (its filmed share is measured from the artist's mask)."""
    return {lk: fr if lk == "receipt" else painter.apply(fr, place, line, shown, age) for lk, fr in frames.items()}


# The sign the words stand on in the world (realone/stage): its size, where it stands, and how a line comes in.
SIGN_CANVAS = (1536, 1536)   # pixels painted on the sign (square: two words a row, up to four rows)
SIGN_WIDTH_M = 1.4           # metres wide in the world (tuned on the Nov 2 stand-in: wider clips)
SIGN_HEIGHT_M = 1.7          # its centre above the floor: around the artist's head, behind it
SIGN_BEHIND_M = 1.4          # how far behind him it stands, along the floor, when its line starts
SIGN_YAW = 28.0              # degrees it is turned, its far end toward the frame's edge
SIGN_SIDE = 0.19             # how far off the centre line a line's sign stands, share of the shot's width
SIGN_TARGET_Y = 0.30         # where its centre lands, mid-line, share of the shot's height
SIGN_SHARE = 0.34            # how much of that shot's width the sign spans


def sign_target(line, body=None):
    """Where a phrase's sign stands in its shot's frame. -> (side, x share, y share, width share); side -1 is the
    frame's left, +1 its right. With the artist's box in that framing (body = (left, right) as shares of the
    width) the sign goes to the side opposite his body's centre, away from his head, at the usual offset and size.
    (Fitting it into the free strip beside the box was tried and lost: with his arms out the box fills the frame
    and the sign shrank to nothing, Nov 2 lab.) Without a box, phrases alternate sides around the centre."""
    if body is None or not all(np.isfinite(body)):
        side = -1 if line % 2 == 0 else 1
    else:
        side = -1 if (float(body[0]) + float(body[1])) / 2 >= 0.5 else 1
    x = float(np.clip(0.5 + side * SIGN_SIDE, SIGN_SHARE / 2 + 0.03, 1 - SIGN_SHARE / 2 - 0.03))
    return side, x, SIGN_TARGET_Y, SIGN_SHARE
SWING_FRAMES = 8             # a new line swings in from a steeper angle over this many frames


def sign_yaw(side, shown, age):
    """The sign's angle for this frame. Its near (bigger) end faces the artist and its far end goes toward the
    frame's edge, so the edge never cuts the big letters: - for a sign on the frame's left (its left end
    farther), + on the right; a new phrase swings in from 2x."""
    base = -SIGN_YAW if side < 0 else SIGN_YAW
    if shown == 1 and age < SWING_FRAMES:
        u = (age + 1) / SWING_FRAMES
        ease = 1 - (1 - u) ** 3
        return base * (2.0 - 1.0 * ease)
    return base


def occlude(alpha, world_zinv, sign_zinv, tol=0.05):
    """Hide the sign wherever the world is nearer than it (a bigger inverse depth): a pillar in front of the words
    covers them. Both depths are the still camera's inverse depth, warped into the frame the same way."""
    return (alpha * (world_zinv <= sign_zinv * (1 + tol))).astype(np.float32)


def sign_painter(lines):
    return LyricPainter(lines, *SIGN_CANVAS, size_frac=0.24, center=0.5, max_words=2, max_rows=4)


def blend(world, rgb, alpha):
    """world: float32 (H, W, 3) 0..255. The words sit IN the world layer (behind the artist, under the front
    particles)."""
    return world * (1 - alpha[..., None]) + rgb * alpha[..., None]
