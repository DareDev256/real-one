"""Tracked effects: name a part of the take in words ("his hands"), pick an effect and a moment, and the effect
follows that part. SAM 3.1 segments the part on every frame of the take (one cached call per phrase), and the effect
is drawn from the part's box frame by frame, so it moves with him.

James, Nov 4: "the ability to place tracked explosions and visual effects down ... like a real video editor, all of
with Meta's abilities". The edit file is the spine an editor writes to, on the song clock:

    {"fx": [{"track": "his hands", "at_s": 26.7, "effect": "fire", "seconds": 1.4}]}

`render --edits FILE`. The effect is generated, never the artist: it is drawn into the world layer (and a fifth of the
embers into the front layer, as the lyric moments do), so the Real Receipt counts it as generated. Refuses when the
part is not found on enough of the frames the effect needs, instead of drawing fire on nothing."""
import json
from pathlib import Path

import numpy as np

from realone.common import Refuse

EFFECTS = {"fire": "embers burst from the part and rise, the part glows, the world takes a warm flash"}
FIRE_EMBERS = 700      # an edit's fire is the subject of its moment: three times a lyric ignite's embers ...
FIRE_SIZE = 1.7        # ... and bigger (the first render read as faint sparks at phone size, Nov 4)
GLOW = np.float32([40, 150, 255])  # BGR: the warm light on and around the part
MIN_FOUND = 0.6        # SAM must find the part on this share of the effect's frames
SECONDS = (0.3, 4.0)   # an effect's length on screen
INTENSITY = (0.25, 2.0)  # the editor's fire slider: ember count and size scale by it (1 = the default fire)


def load_edits(path):
    """The edit file -> its validated fx list. Refuses anything it cannot place exactly."""
    try:
        rec = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise Refuse(f"--edits {path}: cannot read it ({e})") from None
    fx = rec.get("fx")
    cards = load_cards(path, rec)
    if (fx is None or fx == []) and cards:
        return []  # an edit file that only places lyric lines (the editor's text placement)
    if not isinstance(fx, list) or not fx:
        raise Refuse(f"--edits {path}: no 'fx' list")
    out = []
    for k, e in enumerate(fx):
        track, eff = str(e.get("track", "")).strip(), e.get("effect", "fire")
        try:
            at, secs = float(e["at_s"]), float(e.get("seconds", 1.4))
        except (KeyError, TypeError, ValueError):
            raise Refuse(f"--edits {path}: fx {k} needs at_s (song seconds)") from None
        if not track:
            raise Refuse(f"--edits {path}: fx {k} names nothing to track (\"track\": \"his hands\")")
        if eff not in EFFECTS:
            raise Refuse(f"--edits {path}: fx {k} effect {eff!r}; one of {', '.join(EFFECTS)}")
        if not SECONDS[0] <= secs <= SECONDS[1]:
            raise Refuse(f"--edits {path}: fx {k} lasts {secs} s; {SECONDS[0]} to {SECONDS[1]} s")
        try:
            gain = float(e.get("intensity", 1.0))
        except (TypeError, ValueError):
            raise Refuse(f"--edits {path}: fx {k} intensity must be a number") from None
        if not INTENSITY[0] <= gain <= INTENSITY[1]:
            raise Refuse(f"--edits {path}: fx {k} intensity {gain}; {INTENSITY[0]} to {INTENSITY[1]}")
        out.append({"track": track, "at_s": at, "effect": eff, "seconds": secs, "intensity": gain})
    return out


CARD_PLACES = ("world", "bottom", "top")  # lyrictext.card_places' codes 0, 1, 2


def load_cards(path, rec=None):
    """The edit file's lyric placements -> [{"line": g, "place": "world"|"bottom"|"top"}] ([] when it has none).
    g indexes the treatment's lyric lines in order (lyrictext.global_lines). The Real One editor writes them: James,
    Nov 4, "move the text while keeping it still tracked in 3D space" ("world" is a sign on the floor's track).
    Refuses a malformed entry instead of guessing."""
    if rec is None:
        try:
            rec = json.loads(Path(path).read_text())
        except (OSError, ValueError) as e:
            raise Refuse(f"--edits {path}: cannot read it ({e})") from None
    cards = rec.get("cards", [])
    if not isinstance(cards, list):
        raise Refuse(f"--edits {path}: 'cards' must be a list")
    out = []
    for k, c in enumerate(cards):
        if not isinstance(c, dict) or set(c) - {"line", "place", "text"}:
            raise Refuse(f"--edits {path}: cards {k} takes only line, place (and text, a note)")
        line, place = c.get("line"), c.get("place")
        if not isinstance(line, int) or isinstance(line, bool) or line < 0:
            raise Refuse(f"--edits {path}: cards {k} needs line, the lyric line's index (0 or more)")
        if place not in CARD_PLACES:
            raise Refuse(f"--edits {path}: cards {k} place {place!r}; one of {', '.join(CARD_PLACES)}")
        out.append({"line": line, "place": place})
    return out


def _fill(b):
    """NaN rows (the part not found on a frame) take the nearest found box, so a blink of the matte does not throw
    the effect across the frame."""
    b = b.copy()
    ok = ~np.isnan(b[:, 0])
    if not ok.any():
        return b
    idx = np.arange(len(b))
    for j in range(4):
        b[:, j] = np.interp(idx, idx[ok], b[ok, j])
    return b


def plan(fx_list, s0, n, boxes_for, fps=30):
    """fx_list: load_edits(); s0: the render's first frame on the song clock; boxes_for(track) -> (n, 4) source-px
    boxes of that part, NaN where SAM did not find it.
    -> ({"fx_age" int16[n] (-1 none), "fx_id" int16[n], "fx_box" float32[n, 4]}, designed frames, report)."""
    age = np.full(n, -1, np.int16)
    fid = np.full(n, -1, np.int16)
    fbox = np.full((n, 4), np.nan, np.float32)
    designed, report = set(), []
    for j, e in enumerate(fx_list):
        f0 = int(round((e["at_s"] - s0) * fps))
        f1 = min(n, f0 + int(round(e["seconds"] * fps)))
        if f0 < 0 or f0 >= n:
            report.append({**e, "placed": False, "why": "outside this render"})
            continue
        b = np.asarray(boxes_for(e["track"]), float)[f0:f1]
        found = float((~np.isnan(b[:, 0])).mean()) if len(b) else 0.0
        if found < MIN_FOUND:
            raise Refuse(f"fx {j}: SAM 3.1 found '{e['track']}' on {found:.0%} of the frames at {e['at_s']:.2f} s "
                         f"(need {MIN_FOUND:.0%}); pick another moment or another part")
        b = _fill(b)
        age[f0:f1] = np.arange(f1 - f0)
        fid[f0:f1] = j
        fbox[f0:f1] = b
        designed.update(range(f0, min(n, f0 + 2)))   # the warm flash changes the whole frame on purpose
        report.append({**e, "placed": True, "frames": [f0, f1 - 1], "found": round(found, 3)})
    gains = np.float32([float(e.get("intensity", 1.0)) for e in fx_list])  # per effect, indexed by fx_id
    return {"fx_age": age, "fx_id": fid, "fx_box": fbox, "fx_int": gains}, designed, report


def burst(j, intensity=1.0):
    """The ember burst for fx j (deterministic: the same edit renders the same fire). intensity (the editor's fire
    slider, INTENSITY) scales the ember count and their size."""
    from realone.moments import EmberBurst
    g = min(INTENSITY[1], max(INTENSITY[0], float(intensity)))
    return EmberBurst(500 + j, n=max(1, int(round(FIRE_EMBERS * g))), size_mult=FIRE_SIZE * g)


def part_glow(world, box, age, fps=30):
    """A warm glow on and around the tracked part (box in output px), strongest as the fire catches and easing out:
    the light the fire throws, drawn on the world layer behind him (his pixels are composited over it)."""
    from realone.moments import IGNITE_FRAMES
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(140.0, 0.6 * max(x1 - x0, y1 - y0))
    h, w = world.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    t = age / max(1, IGNITE_FRAMES)
    k = min(1.0, age / 4) * max(0.0, 1 - t) ** 0.7
    g = np.exp(-(((xx - cx) / r) ** 2) - (((yy - cy) / r) ** 2)).astype(np.float32)
    return np.clip(world + g[..., None] * GLOW * (0.9 * k), 0, 255)
