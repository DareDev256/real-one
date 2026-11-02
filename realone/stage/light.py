"""The take's own light, measured, so every generated world can be lit the same way.

Gauntlet round 7 (blind critic, real footage as the bar): the same take in a world lit like it (a soft blue hour
desert) beat a world lit against it (a low warm sunset), because a filmed artist's light can be graded but never
regenerated. So the world adapts to him: the background around the artist is measured on a few frames (how dark,
the sky's look), named in plain words, and that sentence is appended to every Muse Image world prompt when the
tracked stage is on. Only the KIND of light is named (soft, hard, low): the light's colour was tried and dropped,
because on real footage a pink brick floor or green trees read as warm light (scene colour is not light colour).
Nothing here touches the artist's pixels."""
import cv2
import numpy as np

from realone.common import H, W

LOW_L = 28.0          # mean background lightness (L*, 0..100) under this: night or a dim interior
SKY_BAND = 0.22       # the top of the frame, where the sky is when there is one
SKY_L = 68.0          # sky pixels are bright...
GREY_C = 14.0         # ...and an overcast sky is nearly colourless (chroma under this)
BLUE_B = -9.0         # a clear sky is clearly blue (b* under this): direct sun
SKY_SHARE = 0.55      # share of the top band that must look like sky before the sky decides


def measure(frames, masks):
    """frames: BGR uint8 list; masks: matching bool (H, W) artist masks. -> dict of the background's light.
    Overall darkness from the whole background; the kind of light from the sky band (bright and grey: overcast,
    bright and blue: clear sun)."""
    L, top_sky_grey, top_sky_blue, top_n = [], 0, 0, 0
    for fr, m in zip(frames, masks):
        bg = ~(cv2.dilate(m.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0)
        lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[..., 0] * 100 / 255
        b = lab[..., 2] - 128
        c = np.hypot(lab[..., 1] - 128, b)
        L.append(l[bg])
        band = np.zeros_like(bg)
        band[: int(H * SKY_BAND)] = True
        band &= bg
        top_n += int(band.sum())
        top_sky_grey += int((band & (l > SKY_L) & (c < GREY_C)).sum())
        top_sky_blue += int((band & (l > SKY_L - 15) & (b < BLUE_B)).sum())
    if not L or not sum(len(x) for x in L):
        return None
    lv = np.concatenate(L)
    return dict(l_mean=float(lv.mean()), l_p3=float(np.percentile(lv, 3)), l_p97=float(np.percentile(lv, 97)),
                sky_grey=top_sky_grey / max(1, top_n), sky_blue=top_sky_blue / max(1, top_n))


def classify(m):
    """-> soft | hard | low. With no sky in frame (an interior, a wall) the light reads soft: a phone take without
    a visible sun is almost always diffuse."""
    if m["l_mean"] < LOW_L:
        return "low"
    if m["sky_blue"] >= SKY_SHARE and m["sky_blue"] > m["sky_grey"]:
        return "hard"
    return "soft"


SENTENCES = {
    "soft": "Light: soft, diffuse light with no direct sun in frame and no hard cast shadows (overcast, shade or "
            "blue hour).",
    "hard": "Light: bright direct sunlight from a high angle, crisp cast shadows.",
    "low": "Light: low light, night or a dim interior, lit by a few practical lights; deep shadows.",
}


def clause(m):
    """The sentence appended to a world prompt so Muse Image lights the world the way the take was lit."""
    return " " + SENTENCES[classify(m)]


def from_take(proxy, masks, n, samples=6):
    """Measure the take on `samples` frames spread over it. masks: packed (n, W*H/8) array. -> (dict, clause)."""
    from realone.media import read_frames
    pick = set(np.linspace(0, max(0, n - 1), samples).astype(int).tolist())
    frs, mks = [], []
    for i, fr in enumerate(read_frames(proxy)):
        if i > max(pick):
            break
        if i in pick:
            frs.append(fr)
            mks.append(np.unpackbits(masks[i])[: W * H].reshape(H, W).astype(bool))
    m = measure(frs, mks)
    if m is None:
        return None, ""
    return dict(m, kind=classify(m)), clause(m)
