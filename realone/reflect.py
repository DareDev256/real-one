"""Wet floors reflect him: in a world whose prompt makes the ground wet (rain, puddles, wet stone, reflections), the
artist's composited pixels are mirrored at the line his feet stand on and laid into the floor, fading with distance
and smeared like water.

Why (Nov 4, gauntlet round 5, both blind critics): "he is not grounded or lit by his worlds ... the puddle at 4.5 s
reflects the sky but not him". Both hooks are wet worlds (the plaza's "wet granite ... reflections", the last hook's
"still puddles mirroring"), so a floor that mirrors everything but the one person standing on it is the tell.

The reflection is made from his filmed pixels but it is not him: it is drawn into the world layer before he is
composited, so the Real Receipt greys it and counts it as generated, and his own pixels are untouched."""
import numpy as np

WET_WORDS = ("wet", "rain", "puddle", "reflect", "mirror", "flood")
STRENGTH = 0.62      # the reflection right at his feet (it fades to 0 over FADE of the room below); 0.42
#                      read as nothing at phone size on the plaza (Nov 4)
FADE = 0.9
DARKEN = 0.55        # wet stone gives back a darker image than the thing itself
SMEAR_PX = 5         # vertical smear (water), px at 1080 wide


def is_wet(prompt):
    p = str(prompt).lower()
    return any(w in p for w in WET_WORDS)


def layer(fg255, alpha):
    """fg255 (H, W, 3) float 0..255, the artist's composited colour in output space; alpha (H, W) 0..1.
    -> (rgb (H, W, 3), a (H, W)) to lay into the world under him, or None when his feet are not in frame (nothing
    to mirror into)."""
    import cv2
    h, w = alpha.shape
    rows = np.nonzero(alpha.max(axis=1) > 0.5)[0]
    if rows.size == 0:
        return None
    yb = int(rows[-1])                 # the floor line: the lowest row of him
    room = h - 1 - yb
    if room < 8:
        return None                    # his feet are at the frame's bottom edge: the floor is out of frame
    k = min(room, yb + 1)
    rgb = np.zeros((h, w, 3), np.float32)
    a = np.zeros((h, w), np.float32)
    src = slice(yb, yb - k, -1)        # from his feet up, mirrored downward
    rgb[yb + 1:yb + 1 + k] = fg255[src] * DARKEN
    fade = np.clip(1 - np.arange(k) / max(1.0, FADE * room), 0, 1)[:, None] ** 1.5
    a[yb + 1:yb + 1 + k] = alpha[src] * STRENGTH * fade
    if SMEAR_PX > 0:
        kx, ky = 1, 2 * SMEAR_PX + 1
        rgb = cv2.blur(rgb, (kx, ky))
        a = cv2.blur(a, (kx, ky))
    return rgb, a


def apply(world, fg255, alpha):
    """The world with his reflection laid into its floor (a new array), or the world unchanged."""
    got = layer(fg255, alpha)
    if got is None:
        return world
    rgb, a = got
    return world * (1 - a[..., None]) + rgb * a[..., None]
