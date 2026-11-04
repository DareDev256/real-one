"""The lyric type (Nov 4, James: "the text should have some effects or font style so it seems more creative"): Anton,
a dark edge, and the word being sung in its section's accent from the director's palette."""
import numpy as np

from realone import lyrictext as LT
from realone import moments as MO
from realone.common import H, W

HOOK = ['#0F4C5C', '#F2A541', '#D64550', '#F5E6C8', '#1A1A1E']   # the plaza hook's palette (Wonder)
GREY = ['#6B7280', '#9CA3AF', '#D1D5DB', '#374151']             # the last hook's grey world


def test_the_accent_is_the_palettes_boldest_bright_colour_and_none_for_grey():
    b, g, r = LT.accent_of(HOOK)
    assert (round(r), round(g), round(b)) == (242, 165, 65)       # #F2A541, amber
    assert LT.accent_of(GREY) is None and LT.accent_of([]) is None


def _coloured(rgb, alpha):
    on = alpha > 0.9
    px = rgb[on]
    return px, (np.abs(px[:, 0] - px[:, 2]) > 60).mean()           # share of letter pixels that are not grey


def test_the_word_being_sung_takes_the_accent_and_the_rest_stay_white():
    acc = LT.accent_of(HOOK)
    p = LT.LyricPainter([["FIRE", "ON", "STAGE"]], W, H, line_colors=[acc])
    rgb, al = p.rgba(0, 3, 10)
    _, share = _coloured(rgb, al)
    assert 0.15 < share < 0.85                                     # STAGE coloured, FIRE ON white
    rgb1, al1 = p.rgba(0, 1, 10)
    assert _coloured(rgb1, al1)[1] > 0.6                           # one word shown: it is the one being sung
    plain = LT.LyricPainter([["FIRE", "ON", "STAGE"]], W, H)
    assert _coloured(*plain.rgba(0, 3, 10))[1] < 0.02              # no palette: all white, as before


def test_a_slam_takes_its_sections_accent():
    acc = LT.accent_of(HOOK)
    sp = MO.SlamPainter(["real"], colors=[acc])
    bg = np.full((H, W, 3), 128, np.uint8)
    out = sp.apply(bg.copy(), 0, 6, None).astype(int)
    warm = (out[..., 2] - out[..., 0]) > 80
    assert warm.mean() > 0.01
    white = MO.SlamPainter(["real"]).apply(bg.copy(), 0, 6, None).astype(int)
    assert ((white[..., 2] - white[..., 0]) > 80).mean() < 0.001
