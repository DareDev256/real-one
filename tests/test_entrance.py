"""The entrance (Nov 4, gauntlet round 1): every one of James's takes starts with an empty wall while he walks to his
mark, so the video opened on an empty generated world. Now the world waits in the dark and its lights come up when he
steps in."""
import numpy as np

from realone import entrance as EN
from realone.common import H, W

N, FPS = 300, 30


def boxes_with_entry(at, n=N):
    b = np.full((n, 4), np.nan)
    b[at:] = (0.1 * W, 0.2 * H, 0.4 * W, 0.95 * H)   # a body
    return b


def test_he_steps_in_when_a_body_holds_for_a_few_frames():
    b = boxes_with_entry(120)
    b[40] = (0, 0.5 * H, 0.01 * W, 0.52 * H)          # a speck at the edge is not him
    b[80:90] = (0, 0.2 * H, 0.02 * W, 0.9 * H)        # nor a thin sliver of matte down the frame edge
    b[60:62] = (0.1 * W, 0.2 * H, 0.4 * W, 0.95 * H)  # two frames of a body is not yet him
    assert EN.entry_frame(b) == 120
    assert EN.entry_frame(boxes_with_entry(0)) == 0
    assert EN.entry_frame(np.full((N, 4), np.nan)) is None


def test_the_world_is_dark_until_he_steps_in_then_the_lights_come_up():
    g = EN.gain(N, 120, FPS)
    assert np.all(g[:120] == np.float32(EN.DARK_GAIN))
    r = int(round(EN.RAMP_S * FPS))
    ramp = g[120:120 + r]
    assert np.all(np.diff(ramp) > 0) and ramp[-1] == np.float32(1.0)
    assert np.all(g[120 + r:] == 1.0)


def test_nothing_changes_when_he_is_there_from_the_start_or_never_comes():
    for e in (0, None):
        assert np.all(EN.gain(N, e, FPS) == 1.0)
    g, rep = EN.plan(boxes_with_entry(0), N, FPS)
    assert not rep["on"] and np.all(g == 1.0)
    g, rep = EN.plan(boxes_with_entry(90), N, FPS)
    assert rep["on"] and rep["entry_frame"] == 90 and rep["lit_frame"] == 90 + int(round(EN.RAMP_S * FPS))


def test_only_the_world_layers_are_darkened():
    world = np.full((4, 4, 3), 200.0, np.float32)
    front = np.full((4, 4, 3), 50.0, np.float32)
    w2, f2 = EN.apply(world, front, 0.25)
    assert np.allclose(w2, 50.0) and np.allclose(f2, 12.5)
    w3, f3 = EN.apply(world, front, 1.0)
    assert w3 is world and f3 is front


def test_before_he_steps_in_the_matte_is_empty():
    g = EN.gain(N, 120, FPS)
    m = np.ones((8, 8), bool)
    assert not EN.matte(m, float(g[50])).any()      # a sliver on the wall before he arrives: not him
    assert EN.matte(m, float(g[120])).all()         # from the entry frame on, his matte as filmed
    assert EN.matte(m, 1.0) is m


def test_a_cut_in_the_dark_is_reported_not_required():
    g = EN.gain(N, 183, FPS)                       # take 2: he steps in at 6.1 s
    assert EN.dark_cuts([103, 206, 400], g) == [103]   # only the cut while it is still dark
    assert EN.dark_cuts([103], np.ones(N)) == [] and EN.dark_cuts([103], None) == []
