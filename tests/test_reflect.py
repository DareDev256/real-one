"""Wet floors reflect him (realone/reflect.py, Nov 4, gauntlet round 5: "the puddle reflects the sky but not him")."""
import numpy as np

from realone import composite as C
from realone import reflect as RF
from realone.common import H, W

import test_looks as TL


def test_wet_worlds_are_read_from_the_prompt():
    assert RF.is_wet("vast brutalist plaza at golden hour, long shadows across wet granite")
    assert RF.is_wet("rain-washed grey boulevard, still puddles mirroring pale towers")
    assert not RF.is_wet("dawn glow through tall loft windows, dust suspended across worn concrete")


def test_the_reflection_starts_at_his_feet_and_fades_down():
    fg = np.full((H, W, 3), 200.0, np.float32)
    al = np.zeros((H, W), np.float32)
    al[600:1500, 400:700] = 1.0                       # he stands, feet on row 1499
    rgb, a = RF.layer(fg, al)
    assert a[:1490].max() == 0                         # nothing above the floor line (but the smear's few px)
    assert a[1520, 550] > 0.3 and a[1700, 550] < a[1520, 550] and a[1900, 550] < 0.1
    assert rgb[1520, 550, 0] < 200                     # wet stone gives back a darker image
    feet_out = np.zeros((H, W), np.float32)
    feet_out[600:H, 400:700] = 1.0                     # feet cut by the frame bottom: nothing to mirror into
    assert RF.layer(fg, feet_out) is None


def test_the_compositor_mirrors_him_only_in_a_wet_world_and_greys_it_on_the_receipt():
    import inspect
    import cv2
    sc = inspect.unwrap(TL.scene)()  # the looks tests' synthetic figure, built directly (not as a fixture)
    small = cv2.resize(sc["world"], (270, 480)).astype(np.float32) / 255
    outs = {}
    for wet in (False, True):
        comp = C.Compositor(TL.args(grain=0.0))
        comp.reflect = wet
        o, _ = comp.frame(sc["src"], sc["mask"], (0, 0, W, H), sc["world"], sc["front"], C.lab_stats(small), 0, 0,
                          ["clean", "receipt"])
        outs[wet] = o
    below = (slice(1520, 1600), slice(380, 700))       # just under his feet (the figure ends at row 1500)
    above = (slice(100, 300), slice(0, 200))           # far from him
    d_below = np.abs(outs[True]["clean"][below].astype(float) - outs[False]["clean"][below]).mean()
    d_above = np.abs(outs[True]["clean"][above].astype(float) - outs[False]["clean"][above]).mean()
    assert d_below > 5 and d_above < 0.5
    rec = outs[True]["receipt"][below].astype(float)
    assert np.abs(rec - outs[False]["receipt"][below]).mean() > 3   # the Receipt shows the reflection too ...
    assert np.abs(rec[..., 0] - rec[..., 2]).max() <= 2              # ... grey, as generated
