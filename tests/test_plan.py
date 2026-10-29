"""Shot plan: every cut on a downbeat (the detected grid AND the song's true bar lines)."""
import numpy as np
import pytest

from realone.audio import pick_section
from realone.common import FPS
from realone.plan import camera_tracks, plan_shots

FRAME = 1 / FPS


@pytest.mark.parametrize("cut_bars", [1, 2, 4])
def test_every_cut_within_one_frame_of_a_downbeat(timeline, song, cut_bars):
    truth = song[2]
    true_db = truth["first_downbeat_s"] + np.arange(truth["n_bars"] + 1) * truth["bar_s"]
    sec = pick_section(timeline, "hook")
    plan = plan_shots(timeline, sec, cut_bars=cut_bars, n_worlds=3)
    db = np.array(timeline["downbeats"])
    assert len(plan["shots"]) == sec["bars"] // cut_bars
    for sh in plan["shots"]:
        t = plan["start_s"] + sh["start_frame"] / FPS  # the time the cut is actually rendered at
        assert np.abs(db - t).min() <= FRAME, (sh, t)
        assert np.abs(true_db - t).min() <= FRAME, (sh, t)
    gaps = np.diff([s["start_frame"] for s in plan["shots"]])
    bar_frames = truth["bar_s"] * FPS * cut_bars
    assert np.all(np.abs(gaps - bar_frames) <= 1.0), gaps


def test_shots_tile_the_section_and_every_cut_changes_something(timeline):
    plan = plan_shots(timeline, pick_section(timeline, "hook"), cut_bars=2, n_worlds=3)
    sh = plan["shots"]
    assert sh[0]["start_frame"] == 0 and sh[-1]["end_frame"] == plan["n_frames"]
    for a, b in zip(sh, sh[1:]):
        assert a["end_frame"] == b["start_frame"]
        assert a["world"] != b["world"] and a["framing"] != b["framing"]


def test_camera_crops_stay_inside_the_frame_and_follow_the_artist(timeline):
    from realone.common import H, W
    plan = plan_shots(timeline, pick_section(timeline, "hook"), cut_bars=2, n_worlds=3)
    n = plan["n_frames"]
    boxes = np.tile(np.array([700.0, 300.0, 1000.0, 1700.0]), (n, 1))  # artist standing right of centre
    boxes[5:9] = np.nan  # a few empty masks must not throw the camera
    crops = camera_tracks(plan, boxes)
    assert np.all(crops[:, 0] >= 0) and np.all(crops[:, 1] >= 0)
    assert np.all(crops[:, 0] + crops[:, 2] <= W + 1e-6) and np.all(crops[:, 1] + crops[:, 3] <= H + 1e-6)
    close = next(s for s in plan["shots"] if s["framing"] == "close")
    c = crops[close["start_frame"]]
    assert c[0] + c[2] / 2 > W / 2  # centred toward the artist, not the frame
