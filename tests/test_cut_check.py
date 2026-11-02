"""realone.encode.planned_cut_found: a planned cut in a busy handheld take is confirmed as a clear local peak of the
frame-difference curve, even where it never reaches the global 6x rule (Savv's reel, Nov 2: median 8.4, cuts 2.0
to 3.0x); a frame that is not the sharpest change around it is not a cut."""
import numpy as np

from realone.encode import planned_cut_found


def busy(n=200, base=9.0, seed=0):
    rng = np.random.default_rng(seed)
    d = base + rng.normal(0, 1.5, n)
    d[0] = 0.0
    return np.clip(d, 0.5, None)


def test_a_cut_three_times_the_local_busyness_is_found_even_below_the_global_rule():
    d = busy()
    d[100] = 28.0                                     # 3.1x the median: the global 6x rule misses it
    assert 28.0 < 6 * np.median(d[1:])
    assert planned_cut_found(d, 100) and planned_cut_found(d, 101)  # one frame of slack
    assert not planned_cut_found(d, 104)              # three frames away: not this cut


def test_no_peak_means_no_cut_and_a_neighbouring_bigger_change_wins():
    d = busy()
    assert not planned_cut_found(d, 100)              # nothing there
    d[100], d[102] = 25.0, 40.0                       # a bigger change two frames later
    assert not planned_cut_found(d, 100)              # 100 is not the peak of its neighbourhood


def test_the_bar_is_lower_only_where_a_designed_change_shares_the_frame():
    d = busy()
    d[100] = 19.0                                     # about 2.1x
    assert not planned_cut_found(d, 100)              # 2.5x is the normal bar
    assert planned_cut_found(d, 100, ratio=2.0)       # a crash snap on the same downbeat: 2x
