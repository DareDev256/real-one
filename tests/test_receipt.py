"""Real Receipt: the filmed share is the artist's final matte over the frame; the counter draws exactly that number
(within 0.5 percentage points of the mask area), only on receipt frames; the receipt json keeps every frame;
--receipt-reveal flips a section to the receipt look."""
import re
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from realone import composite as C
from realone import receipt as R
from realone.common import H, W
from realone.freeze import select_sections


def args(**kw):
    d = dict(erode=2, feather=1.4, wrap=0.6, match=0.35, grain=0.0, despill=True)
    d.update(kw)
    return SimpleNamespace(**d)


@pytest.fixture(scope="module")
def scene():
    src = np.full((H, W, 3), (70, 56, 46), np.uint8)
    mask = np.zeros((H, W), np.uint8)
    cv2.rectangle(mask, (330, 700), (750, 1500), 1, -1)
    cv2.circle(mask, (540, 520), 150, 1, -1)
    src[mask > 0] = (40, 40, 160)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    world = np.stack([120 + 100 * np.sin(xx / 90), 60 + 50 * np.cos(yy / 130), 200 - 60 * np.sin((xx + yy) / 200)], -1)
    return {"src": src, "mask": mask > 0, "world": world.astype(np.float32), "front": np.zeros((H, W, 3), np.float32)}


def render(scene, look, counter, crop=(0, 0, W, H)):
    comp = C.Compositor(args())
    small = cv2.resize(scene["world"], (270, 480)).astype(np.float32) / 255
    outs, info = comp.frame(scene["src"], scene["mask"], crop, scene["world"], scene["front"], C.lab_stats(small),
                            0, 0, [look], counter=counter)
    return outs[look], info[look]


def drawn_value(text):
    m = re.match(r"FILMED (\d+\.\d)%  GENERATED (\d+\.\d)%", text)
    assert m, text
    return float(m.group(1)), float(m.group(2))


@pytest.mark.parametrize("crop", [(0, 0, W, H), (300, 400, W / 1.4, H / 1.4)])
def test_counter_matches_the_mask_area_within_half_a_point(scene, crop):
    out, info = render(scene, "receipt", counter=True, crop=crop)
    f, g = drawn_value(info["counter_text"])
    assert abs(f + g - 100.0) < 1e-9
    z = W / crop[2]
    M = np.float32([[z, 0, -z * crop[0]], [0, z, -z * crop[1]]])
    hard = cv2.warpAffine(scene["mask"].astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR) > 0.5
    area = hard.mean() * 100
    assert abs(f - area) <= 0.5, (f, area)
    assert abs(f - info["filmed"] * 100) <= 0.05  # the counter shows the measured number, rounded to 0.1


def test_counter_is_drawn_only_on_receipt_frames(scene):
    with_c, _ = render(scene, "receipt", counter=True)
    without, _ = render(scene, "receipt", counter=False)
    box = (slice(R.Y0 - 10, R.Y0 + 140), slice(R.X0 - 10, R.X0 + R.BLOCK_W + 10))
    assert np.abs(with_c[box].astype(int) - without[box].astype(int)).mean() > 8
    assert np.array_equal(with_c[1000:], without[1000:])  # nothing else on the frame changed
    clean_c, info = render(scene, "clean", counter=True)
    clean, _ = render(scene, "clean", counter=False)
    assert np.array_equal(clean_c, clean) and "counter_text" not in info


def test_every_look_reports_its_filmed_share(scene):
    for look in C.LOOKS:
        _, info = render(scene, look, counter=False)
        assert 0.05 < info["filmed"] < 0.4, (look, info["filmed"])


def test_split_sums_to_100():
    for v in (0.0, 0.23449, 0.2345, 0.99999, 1.0, 0.5):
        f, g = R.split(v)
        assert abs(f + g - 100.0) < 1e-9 and 0 <= f <= 100


def test_select_sections(timeline):
    secs = timeline["sections"]
    assert [s["start_bar"] for s in select_sections(timeline, "hook", secs)] == [12, 28]
    assert [s["start_bar"] for s in select_sections(timeline, "hook:last", secs)] == [28]
    assert [s["start_bar"] for s in select_sections(timeline, "#1", secs)] == [4]


def test_the_counter_goes_where_the_artist_is_not():
    """A close-up whose head fills the top left: the counter moves; a wide shot with the artist centred low keeps
    the top left; the place never overlaps the artist when a free place exists."""
    head_top_left = [(0.0, 0.0, 560.0, 900.0)] * 10       # a punch-in: head and shoulders over the top left
    assert R.place_for(head_top_left) == 2
    centred = [(330.0, 500.0, 750.0, 1700.0)] * 10       # a wide shot
    assert R.place_for(centred) == 1
    tall_both_top = [(0.0, 0.0, 1080.0, 1000.0)] * 10      # covers the whole top
    assert R.place_for(tall_both_top) == 3


def test_counter_drawn_at_its_place(scene):
    comp = C.Compositor(args())
    small = cv2.resize(scene["world"], (270, 480)).astype(np.float32) / 255
    outs = {}
    for place in (0, 2):
        o, info = comp.frame(scene["src"], scene["mask"], (0, 0, W, H), scene["world"], scene["front"],
                             C.lab_stats(small), 0, 0, ["receipt"], counter=place)
        outs[place] = o["receipt"]
    x, y = R.PLACES[2]
    box = (slice(y, y + R.BLOCK_H), slice(x, x + R.BLOCK_W))
    assert np.abs(outs[2][box].astype(int) - outs[0][box].astype(int)).mean() > 8
    tl = (slice(R.Y0, R.Y0 + R.BLOCK_H), slice(R.X0, R.X0 + 300))
    assert np.array_equal(outs[2][tl], outs[0][tl])  # nothing drawn top left


def test_a_free_place_beats_a_slightly_covered_one():
    assert R.place_for([(0.0, 0.0, 600.0, 900.0)] * 4) == 3  # the top right block would clip 30 px of the head


def test_in_a_close_shot_the_counter_keeps_off_his_face():
    """Gauntlet round 1 (Nov 4): in a close shot his body covers every place, and the counter sat over his mouth.
    He fills the frame from y=120 down: every place overlaps his body by the same area, so the old rule tied and
    took the first (top left, on his face). His head is the top quarter of his box; the counter goes lower left."""
    from realone import receipt as R
    assert R.place_for([(0.0, 120.0, 1080.0, 1920.0)] * 4) == 3
