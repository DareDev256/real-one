"""Several takes, one edit (python -m realone.director takes, Nov 4): Wonder's per-section picks are checked locally,
and the cut points are computed on the song clock from each take's own sync offset."""
import pytest

from realone.common import Refuse
from realone.director import takes as TK

SECS = [{"index": 0, "start_s": 0.037, "end_s": 6.899}, {"index": 1, "start_s": 6.899, "end_s": 20.633},
        {"index": 2, "start_s": 20.633, "end_s": 41.171}, {"index": 3, "start_s": 41.171, "end_s": 54.917}]
# James's three run-throughs (song time = clip time + offset): take 1 started 6 s into the song
SPANS = [{"path": "t1.mov", "offset_s": 5.98, "song_start_s": 5.98, "song_end_s": 57.0},
         {"path": "t2.mov", "offset_s": -0.571, "song_start_s": -0.571, "song_end_s": 57.74},
         {"path": "t3.mov", "offset_s": -0.585, "song_start_s": -0.585, "song_end_s": 58.39}]


def reply(*takes):
    return {"picks": [{"section": i, "take": t, "why": "x" * 5} for i, t in enumerate(takes)], "why": ""}


def test_a_take_covers_a_section_only_if_it_runs_the_whole_section():
    assert not TK.covers(SPANS[0], SECS[0])  # take 1 missed the intro
    assert TK.covers(SPANS[0], SECS[1]) and TK.covers(SPANS[1], SECS[0])


def test_picks_are_checked_before_anything_is_cut():
    assert TK.check(reply(2, 1, 3, 2), SECS, SPANS) == {0: 1, 1: 0, 2: 2, 3: 1}
    with pytest.raises(Refuse, match="take 1 does not cover section 0"):
        TK.check(reply(1, 2, 2, 2), SECS, SPANS)
    with pytest.raises(Refuse, match="there are 3"):
        TK.check(reply(2, 4, 2, 2), SECS, SPANS)
    with pytest.raises(Refuse, match="no pick for section"):
        TK.check(reply(2, 2, 2), SECS, SPANS)
    bad = reply(2, 2, 2, 2)
    bad["picks"].append({"section": 9, "take": 2, "why": "nope!"})
    with pytest.raises(Refuse, match="does not have"):
        TK.check(bad, SECS, SPANS)


def test_segments_cut_each_take_on_the_song_clock():
    segs = TK.segments(SECS, {0: 1, 1: 0, 2: 2, 3: 1}, SPANS, 57.05)
    assert [k for k, *_ in segs] == [1, 0, 2, 1]
    # contiguous on the song clock: each segment starts where the last one ended
    song_starts = [s0 for *_, s0 in segs]
    lengths = [c1 - c0 for _, c0, c1, _ in segs]
    for j in range(1, len(segs)):
        assert song_starts[j] == pytest.approx(song_starts[j - 1] + lengths[j - 1], abs=1e-3)
    # the clip time of a section start is its song time minus that take's offset
    assert segs[1][1] == pytest.approx(6.899 - 5.98, abs=1e-3)
    assert segs[2][1] == pytest.approx(20.633 + 0.585, abs=1e-3)
    # the first segment starts at the song's start; the last runs to the song's end, inside its take
    assert segs[0][3] == 0.0 and segs[0][1] == pytest.approx(0.571, abs=1e-3)
    assert segs[-1][2] == pytest.approx(57.05 + 0.571, abs=1e-3)


def test_the_last_segment_stops_where_its_take_stops():
    short = [dict(SPANS[0]), dict(SPANS[1], song_end_s=55.5), dict(SPANS[2])]
    segs = TK.segments(SECS, {0: 1, 1: 1, 2: 1, 3: 1}, short, 57.05)
    assert segs[-1][2] == pytest.approx(55.5 + 0.571, abs=1e-3)


def _wall(seed=3, h=480, w=270):
    import numpy as np
    rng = np.random.default_rng(seed)
    import cv2
    g = cv2.GaussianBlur((rng.random((h, w)) * 255).astype("uint8"), (0, 0), 1.2)
    return cv2.equalizeHist(g)


def test_the_camera_check_ignores_the_artist_and_sees_the_phone_move():
    """The first probe (phase correlation over the whole frame) read 299 px between two takes from an unmoved
    phone: it measured the artist. Now the artist moving reads ~0 and the wall moving reads its shift."""
    import numpy as np
    import cv2
    wall = _wall()
    a, b = wall.copy(), wall.copy()
    a[150:440, 40:120] = 20    # the artist on the left in one take ...
    b[120:440, 150:240] = 20   # ... on the right in the other, standing taller
    still, n_in = TK.camera_shift(a, b)
    assert still < 2.0 and n_in >= 30
    moved = cv2.warpAffine(b, np.float32([[1, 0, 18], [0, 1, 0]]), (b.shape[1], b.shape[0]),
                           borderMode=cv2.BORDER_REPLICATE)
    shift, _ = TK.camera_shift(a, moved)
    assert 15 < shift < 22
