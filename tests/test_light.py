"""The take's light (realone/stage/light.py): measured off the background, named, appended to world prompts."""
import cv2
import numpy as np

from realone.common import H, W
from realone.stage import light as LT


def _frame(level, sky=None, tint=(0, 0, 0), seed=0, floor=None):
    """A background at `level` (0..255 grey), optionally a sky across the top quarter (BGR) and a coloured floor."""
    rng = np.random.default_rng(seed)
    f = np.full((H, W, 3), level, np.float32)
    f += np.array(tint, np.float32)
    if sky is not None:
        f[: H // 4] = np.array(sky, np.float32)
    if floor is not None:
        f[H // 2:] = np.array(floor, np.float32)
    f += rng.normal(0, 2, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def _mask():
    m = np.zeros((H, W), bool)
    m[700:1700, 450:650] = True
    return m


def test_the_sky_decides_soft_or_hard_and_darkness_decides_low():
    overcast = LT.measure([_frame(110, sky=(235, 235, 232))], [_mask()])           # white-grey sky
    clear = LT.measure([_frame(120, sky=(235, 170, 110))], [_mask()])             # BGR: a blue sky
    interior = LT.measure([_frame(140)], [_mask()])                               # no sky at all
    night = LT.measure([_frame(40)], [_mask()])
    assert LT.classify(overcast) == "soft" and "no direct sun" in LT.clause(overcast)
    assert LT.classify(clear) == "hard" and "direct sunlight" in LT.clause(clear)
    assert LT.classify(interior) == "soft"
    assert LT.classify(night) == "low"
    assert "colour" not in LT.clause(overcast)      # the light's colour is not claimed (scene colour fooled it)


def test_the_artist_is_not_measured():
    f = _frame(150)
    m = np.zeros((H, W), bool)
    m[300:1800, 200:880] = True                    # a big figure in a black outfit
    f[m] = 10
    got = LT.measure([f], [m])
    grey = float(cv2.cvtColor(np.full((1, 1, 3), 150, np.uint8), cv2.COLOR_BGR2LAB)[0, 0, 0]) * 100 / 255
    assert abs(got["l_mean"] - grey) < 1.5           # the background's lightness, untouched by his clothes
