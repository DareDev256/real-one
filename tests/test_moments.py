"""Lyric moments (realone/moments.py): the director's picked words get an ignite or a slam on the frame they land;
fire words ignite on their own only when no director chose; the Real Receipt never carries a slam."""
import numpy as np

from realone import moments as MO

LINES = [["I'm", "good,", "I'm", "good"], ["get", "burnt", "up"], ["lights", "on", "at", "four"]]


def _sched(n=120):
    # phrase 0 words land at 0, 10, 20, 30; phrase 1 at 40, 50, 60; phrase 2 at 70, 80, 90, 100
    line, shown = np.full(n, -1, np.int16), np.zeros(n, np.int16)
    t = 0
    for g, ws in enumerate(LINES):
        for k in range(len(ws)):
            for i in range(t, t + 10):
                line[i], shown[i] = g, k + 1
            t += 10
    return line, shown


def test_director_picks_win_in_order_and_cap():
    got = MO.pick(LINES, [{"word": "GOOD", "effect": "slam"}, {"word": "lights", "effect": "ignite"},
                          {"word": "good", "effect": "slam"}, {"word": "four", "effect": "slam"}])
    assert [(g, k, e) for g, k, e, _ in got] == [(0, 1, "slam"), (2, 0, "ignite"), (0, 3, "slam")]


def test_bad_picks_are_skipped():
    got = MO.pick(LINES, [{"word": "nowhere", "effect": "slam"}, {"word": "up", "effect": "explode"},
                          {"word": "up", "effect": "ignite"}])
    assert [(g, k, e) for g, k, e, _ in got] == [(1, 2, "ignite")]


def test_fire_words_ignite_only_without_a_director():
    assert [(g, k, e) for g, k, e, _ in MO.pick(LINES, None)] == [(1, 1, "ignite")]
    assert MO.pick(LINES, []) == []  # the director chose no moments: none


def test_land_frame():
    line, shown = _sched()
    assert MO.land_frame(line, shown, 1, 1) == 50
    assert MO.land_frame(line, shown, 0, 0) == 0
    assert MO.land_frame(line, shown, 5, 0) is None


def test_profile_spans_and_designed():
    line, shown = _sched()
    mo = [(1, 1, "ignite", "burnt"), (0, 1, "slam", "good,")]
    arr, words, designed, report = MO.profile(mo, line, shown, 120)
    assert arr["mo_ign"][50] == 0 and arr["mo_ign"][49] == -1 and arr["mo_ign"][50 + MO.IGNITE_FRAMES - 1] > 0
    assert words == ["good,"] and arr["mo_slam"][10] == 0 and arr["mo_slam_age"][15] == 5
    assert arr["mo_slam"][10 + MO.SLAM_FRAMES] == -1
    assert {50, 51, 10, 11} <= designed and 12 not in designed
    assert [r["frame"] for r in report] == [50, 10]


def test_ignite_box_comes_from_the_artist():
    line, shown = _sched()
    boxes = np.full((120, 4), np.nan)
    boxes[50] = (100, 200, 300, 900)
    arr, *_ = MO.profile([(1, 1, "ignite", "burnt")], line, shown, 120, boxes)
    assert tuple(arr["mo_ign_box"][60]) == (100, 200, 300, 900)


def test_flash_hits_then_settles():
    assert MO.flash_amount(0) > MO.flash_amount(4) > MO.flash_amount(MO.IGNITE_FRAMES - 1) > 0
    assert MO.flash_amount(MO.IGNITE_FRAMES) == 0


def test_burst_is_deterministic_and_rises():
    box = (400, 600, 700, 1800)
    a, fa = MO.EmberBurst(3).draw(10, box)
    b, fb = MO.EmberBurst(3).draw(10, box)
    assert np.array_equal(a, b) and np.array_equal(fa, fb)
    early, _ = MO.EmberBurst(3).draw(3, box)
    late, _ = MO.EmberBurst(3).draw(20, box)
    ys = lambda img: np.average(np.arange(img.shape[0]), weights=img.sum(axis=(1, 2)) + 1e-9)
    assert ys(late) < ys(early)  # the embers climb
    assert a.sum() > 0 and MO.EmberBurst(3).draw(MO.IGNITE_FRAMES + 20, box)[0].sum() == 0


def test_ignite_warms_the_world():
    world = np.full((MO.H_, MO.W_, 3), 100, np.float32)
    w, f = MO.ignite(world, np.zeros_like(world), 2, (400, 600, 700, 1800), MO.EmberBurst(0))
    assert w[..., 2].mean() > 100 > w[..., 0].mean() - 1  # red up, blue not up
    assert w.max() <= 255


def test_slam_draws_and_ends():
    fr = np.zeros((MO.H_, MO.W_, 3), np.uint8)
    sp = MO.SlamPainter(["good,"])
    on = sp.apply(fr, 0, 5)
    assert (on > 200).sum() > 10000  # a big white word
    assert sp.apply(fr, 0, MO.SLAM_FRAMES) is fr and sp.apply(fr, -1, 0) is fr
    assert (sp.apply(fr, 0, MO.SLAM_FRAMES - 1) > 200).sum() < (on > 200).sum()  # fading
    assert sp.words == ["GOOD"]


def test_box_out_follows_the_crop():
    assert MO.box_out((100, 200, 300, 400), (0, 0, MO.W_, MO.H_)) == (100, 200, 300, 400)
    x0, y0, x1, y1 = MO.box_out((540, 960, 640, 1060), (540, 960, MO.W_ / 2, MO.H_ / 2))
    assert (x0, y0) == (0, 0) and abs(x1 - 200) < 1e-6
