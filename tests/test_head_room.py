"""Close and medium shots keep his head in frame (gauntlet round 4, Nov 4): the smoothed track lagged him standing
up, and a close shot of "Every pixel's real" cut his head off at the top."""
import numpy as np

from realone.common import H, W
from realone.plan import HEAD_ROOM, camera_tracks

N = 120


def _plan(framing, zoom):
    return {"n_frames": N, "shots": [{"start_frame": 0, "end_frame": N, "zoom": zoom, "framing": framing}]}


def _standing_up():
    """He crouches (box top at 900 px), then stands up fast over 6 frames (top at 120 px) and stays up."""
    b = np.zeros((N, 4))
    top = np.where(np.arange(N) < 40, 900.0, 120.0)
    top[40:46] = np.linspace(900, 120, 6)
    b[:, 0], b[:, 2], b[:, 1], b[:, 3] = 300, 800, top, 1880
    return b, top


def test_a_close_shot_never_starts_below_his_head():
    boxes, top = _standing_up()
    crops = camera_tracks(_plan("close", 1.9), boxes)
    ch = H / 1.9
    room = (top - crops[:, 1]) / ch
    assert room.min() >= HEAD_ROOM / 2 - 1e-6          # his head stays in on every frame, even mid stand-up
    assert room[60:].min() >= HEAD_ROOM - 1e-6         # and the full headroom once he is up
    assert np.abs(np.diff(crops[:, 1])).max() < 200    # a fast tilt with him, not a snap of the whole 780 px


def test_a_wide_shot_is_untouched():
    boxes, _ = _standing_up()
    crops = camera_tracks(_plan("wide", 1.0), boxes)
    assert np.all(crops == (0, 0, W, H))
