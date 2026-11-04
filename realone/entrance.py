"""The entrance: when the take starts before the artist is in frame, the world stays dark until he steps in, then
its lights come up.

Why (Nov 4, gauntlet round 1 on James's first real take): every one of his takes starts with an empty wall while he
walks to his mark, so the music video opened on an empty generated world for 5.5 s and the intro line ("One take,
I'm the real one") played over nobody. A blind critic scored the first three seconds 3/10: "a judge has read 'AI
background loop' before the real one appears". The song answers it itself: "Lights go up when I step in".

Rules: only the world layers are darkened (the artist, when he arrives, is his own exposure); the lyric lines that
play in the dark are front cards (a sign standing in a dark world would be dark too); before he steps in the matte
is empty (the critic saw "a pale sliver at the left edge" on the empty take: a matte on the wall or a few pixels of
him at the frame edge, which a dark world would make glare); nothing changes when the artist is in the first
frame."""
import numpy as np

from realone.common import H, W

DARK_GAIN = 0.06           # the world before he steps in: nearly black, the shapes just there
RAMP_S = 0.7               # the lights coming up, eased
MIN_BOX_W = 0.04           # a body, not a speck: his box at least this share of the frame wide ...
MIN_BOX_H = 0.30           # ... and this share high
HOLD_FRAMES = 3            # ... on this many frames in a row


def entry_frame(boxes):
    """The first frame of HOLD_FRAMES in a row on which the artist's box is a body (boxes: (n, 4) source px, NaN
    when no mask). 0 when he is in the first frame; None when he never appears."""
    b = np.asarray(boxes, float)
    ok = ~np.isnan(b[:, 0]) & ((b[:, 2] - b[:, 0]) >= MIN_BOX_W * W) & ((b[:, 3] - b[:, 1]) >= MIN_BOX_H * H)
    run = 0
    for i, v in enumerate(ok):
        run = run + 1 if v else 0
        if run >= HOLD_FRAMES:
            return i - HOLD_FRAMES + 1
    return None


def gain(n, entry, fps=30):
    """-> float32[n]: DARK_GAIN before the entry frame, an eased ramp to 1 over RAMP_S, then 1. All ones when the
    artist is there from the start (entry 0) or never comes (None: nothing to wait for)."""
    g = np.ones(n, np.float32)
    if not entry:
        return g
    g[:entry] = DARK_GAIN
    r = max(1, int(round(RAMP_S * fps)))
    u = np.arange(1, r + 1) / r
    ease = 1 - (1 - u) ** 3
    end = min(n, entry + r)
    g[entry:end] = DARK_GAIN + (1 - DARK_GAIN) * ease[: end - entry]
    return g


def plan(boxes, n, fps=30):
    """-> (gain array, report)."""
    e = entry_frame(boxes)
    g = gain(n, e, fps)
    if not e:
        why = "the artist is in the first frame" if e == 0 else "the artist never appears"
        return g, {"on": False, "why": why}
    return g, {"on": True, "entry_frame": int(e), "entry_s": round(e / fps, 3), "dark_gain": DARK_GAIN,
               "ramp_s": RAMP_S, "lit_frame": int(min(n, e + round(RAMP_S * fps)))}


def apply(world, front, g):
    """Darken the world layers by g (the artist is composited over them afterwards)."""
    if g >= 1.0:
        return world, front
    return world * g, (front * g if front is not None else None)


def before_entry(g):
    """True on the frames before he steps in (the world at DARK_GAIN, not yet ramping)."""
    return g <= DARK_GAIN + 1e-6


def matte(mask, g):
    """The artist's matte for a frame with world gain g: empty before he steps in, untouched after."""
    return np.zeros_like(mask) if before_entry(g) else mask
