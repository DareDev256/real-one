"""Lyrics in the world (realone/lyrictext.py): words land on beats inside their line's bars, nothing is invented
outside the render, the painter writes only where the words are, and the world keeps every pixel the words do not
cover. The artist is composited over the world afterwards, so his pixels are never touched (units.py)."""
import numpy as np
import pytest

from realone import lyrictext as LT
from realone.common import H, W


def timeline(bpm=120, bars=4, start=0.0):
    beat = 60 / bpm
    beats = [start + k * beat for k in range(bars * 4 + 1)]
    return {"beats": beats, "downbeats": beats[::4][:bars],
            "sections": [{"index": 0, "start_s": start, "end_s": start + bars * 4 * beat}]}


def test_each_word_lands_on_a_beat_and_each_line_owns_its_bars():
    tl = timeline()  # 4 bars of 4 beats, a beat every 0.5 s
    lines, line, shown, age = LT.schedule({0: ["one two three four", "five six seven eight"]}, tl, 0.0, 240)
    assert lines == [["one", "two", "three", "four"], ["five", "six", "seven", "eight"]]  # four words: one phrase
    f = lambda s: int(round(s * 30))
    assert (line[f(0):f(4)] == 0).all() and (line[f(4):f(8) - 1] == 1).all()     # 2 bars each
    assert [int(shown[f(t)]) for t in (0, 0.5, 1.0, 1.5, 3.9)] == [1, 2, 3, 4, 4]  # a word per beat, then hold
    assert int(age[f(0.5)]) == 0 and int(age[f(0.5) + 6]) == 6                    # age restarts as a word lands
    assert int(shown[f(4)]) == 1                                                  # the next line starts over


def test_a_busy_line_uses_half_beats_and_nothing_shows_outside_its_section():
    tl = timeline(bars=1, start=2.0)  # one bar starting at 2.0 s: 4 beats
    lines, line, shown, age = LT.schedule({0: ["a b c d e f g"]}, tl, 0.0, 150)
    assert (line[:60] == -1).all()                                                # before the section: nothing
    assert int(shown[int(round(2.25 * 30))]) == 2                                 # half a beat in: two words
    assert (line[int(round(4.0 * 30)):] == -1).all()                              # after it: nothing


def test_lines_outside_the_render_are_not_counted_and_the_render_refuses_without_words():
    tl = timeline(bars=4, start=10.0)
    with pytest.raises(LT.LyricError, match="none of the treatment's lyric lines"):
        LT.for_render({0: ["late line"]}, tl, 0.0, 90)
    with pytest.raises(LT.LyricError, match="needs lyric lines"):
        LT.for_render({0: []}, tl, 0.0, 90)
    with pytest.raises(LT.LyricError, match="no beat grid"):
        LT.schedule({0: ["x"]}, {"beats": [], "sections": []}, 0.0, 10)
    lines, arr, rep = LT.for_render({0: ["seen here"]}, tl, 10.0, 30)
    assert rep["lines"] == 1 and rep["words"] == 2 and "not aligned to the vocal" in rep["timing"]


def test_the_painter_writes_only_the_words_and_more_words_mean_more_ink():
    p = LT.LyricPainter([["I'M", "GOOD,", "DON'T", "CALL", "IT"]], W, H)
    assert p.rgba(-1, 0, 0) is None and p.rgba(0, 0, 0) is None
    one = p.rgba(0, 1, 9)[1]
    three = p.rgba(0, 3, 9)[1]
    assert one.sum() > 0 and three.sum() > one.sum() * 1.8
    rows = np.nonzero(three.max(axis=1) > 0.05)[0]
    assert rows.min() > H * 0.12 and rows.max() < H * 0.5                         # the upper part of the frame
    world = np.full((H, W, 3), 40.0, np.float32)
    rgb, al = p.rgba(0, 3, 9)
    out = LT.blend(world, rgb, al)
    assert np.array_equal(out[al == 0], world[al == 0])                           # untouched where no word is
    assert out[al > 0.99].mean() > 200                                            # the letters are white


def test_the_newest_word_lands_with_a_short_pop_then_settles():
    p = LT.LyricPainter([["ONE", "TWO"]], W, H)
    landing = p.rgba(0, 2, 0)[1]
    settled = p.rgba(0, 2, LT.POP_FRAMES + 2)[1]
    assert landing.sum() < settled.sum()                                          # fading in on the beat
    assert np.array_equal(p.rgba(0, 2, LT.POP_FRAMES + 2)[1], settled)            # then stable


def test_a_section_that_starts_where_the_render_ends_adds_no_lines():
    """The Nov 2 hook render ends a frame before the bridge starts: the bridge's lines must not be counted."""
    tl = timeline(bars=2)
    tl["sections"].append({"index": 1, "start_s": 3.99, "end_s": 8.0})  # its first downbeat is 4.0 s
    lines, arr, rep = LT.for_render({0: ["in the hook"], 1: ["in the bridge"]}, tl, 0.0, 120)  # last frame 3.97 s
    assert lines == [["in", "the", "hook"]] and rep["lines"] == 1



def test_the_words_survive_the_stage_warp_onto_the_frame():
    """Nov 2: the painter returned (h, w, 1) colour; cv2.warpPerspective drops a single channel axis, and the first
    stage render died on the blend. The stage path: paint on the still's canvas, warp by a homography, blend."""
    import cv2
    from realone.common import H as FH, W as FW
    p = LT.LyricPainter([["WARP", "ME"]], 1152, 2048)
    rgb, al = p.rgba(0, 2, 9)
    assert rgb.shape == (2048, 1152, 3) and al.shape == (2048, 1152)
    Hm = np.array([[FW / 1152 * 1.1, 0.02, -30], [0.01, FH / 2048 * 1.1, -40], [0, 0, 1]], float)
    rgb_w = cv2.warpPerspective(rgb, Hm, (FW, FH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    al_w = cv2.warpPerspective(al, Hm, (FW, FH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    out = LT.blend(np.zeros((FH, FW, 3), np.float32), rgb_w, al_w)
    assert out.shape == (FH, FW, 3) and out.max() > 200



def test_a_nearer_world_hides_the_sign_and_a_farther_one_does_not():
    al = np.ones((4, 4), np.float32)
    sign = np.full((4, 4), 1 / 3.5, np.float32)                 # the sign 3.5 m away
    world = np.full((4, 4), 1 / 10.0, np.float32)               # the world behind it, 10 m
    world[:, :2] = 1 / 2.0                                      # a pillar 2 m away on the left
    out = LT.occlude(al, world, sign)
    assert (out[:, :2] == 0).all() and (out[:, 2:] == 1).all()


def test_a_new_line_swings_in_then_holds_its_angle_and_lines_alternate_sides():
    swing = [abs(LT.sign_yaw(-1, 1, a)) for a in range(LT.SWING_FRAMES + 2)]
    assert swing[0] > 1.5 * LT.SIGN_YAW and swing[-1] == LT.SIGN_YAW
    assert all(x >= y for x, y in zip(swing, swing[1:]))         # it settles, never overshoots back out
    assert LT.sign_yaw(1, 3, 20) == LT.SIGN_YAW and LT.sign_yaw(-1, 3, 20) == -LT.SIGN_YAW  # far end to the edge
    assert LT.sign_target(0)[1] < 0.5 < LT.sign_target(1)[1]                                # no box: alternate


def test_the_sign_sets_two_words_a_row_so_the_block_stays_narrow():
    f, rows, _ = LT.layout(["LIGHTS", "ON", "AT", "FOUR", "AND", "I'M"], *LT.SIGN_CANVAS, size_frac=0.24, center=0.5,
                           max_words=2, max_rows=4)
    assert max(len(r) for r in rows) <= 2 and len(rows) <= 4 and sum(len(r) for r in rows) == 6


def test_short_words_still_break_two_a_row():
    f, rows, _ = LT.layout(["I", "AM", "SO", "OK", "GO", "ON"], *LT.SIGN_CANVAS, size_frac=0.24, center=0.5,
                           max_words=2, max_rows=4)
    assert [len(r) for r in rows] == [2, 2, 2]



def test_a_long_line_is_split_into_short_phrases_each_on_its_own_bars():
    assert LT.phrases("lights on at four and I'm still holding".split()) == [
        ["lights", "on", "at", "four"], ["and", "I'm", "still", "holding"]]
    assert LT.phrases("I'm good, I'm good, don't call it".split()) == [["I'm", "good,", "I'm", "good,"],
                                                                     ["don't", "call", "it"]]
    tl = timeline(bars=4)
    lines, line, shown, age = LT.schedule({0: ["a b c d e f g h"]}, tl, 0.0, 240)
    assert lines == [list("abcd"), list("efgh")]
    assert (line[: 2 * 60] == 0).all() and (line[2 * 60: 4 * 60 - 1] == 1).all()   # two bars each



def test_a_phrase_goes_to_the_side_with_more_room_away_from_his_body():
    side, x, _, share = LT.sign_target(1, (0.40, 0.95))   # he is right of centre
    assert side == -1 and x < 0.5 and x - share / 2 >= 0  # the sign goes left, inside the frame
    side, x, _, share = LT.sign_target(0, (0.05, 0.55))   # he is left of centre
    assert side == 1 and x > 0.5 and x + share / 2 <= 1
    side, x, _, share = LT.sign_target(0, (0.10, 0.98))   # arms out across the frame: still a full-size sign
    assert share == LT.SIGN_SHARE
