"""The editor's fire intensity slider: an fx entry's optional "intensity" (0.25 to 2, default 1) scales the fire's
ember count and size (fx.burst), travels through fx.plan as fx_int (per effect), and anything outside refuses."""
import json

import numpy as np
import pytest

from realone import fx as FX
from realone.common import Refuse

FIRE = {"track": "his hands", "at_s": 32.817, "effect": "fire", "seconds": 1.4}


def _load(tmp_path, **extra):
    p = tmp_path / "edits.json"
    p.write_text(json.dumps({"fx": [{**FIRE, **extra}]}))
    return FX.load_edits(p)


def test_default_intensity_is_one_and_the_slider_value_is_kept(tmp_path):
    assert _load(tmp_path)[0]["intensity"] == 1.0
    assert _load(tmp_path, intensity=1.8)[0]["intensity"] == 1.8


@pytest.mark.parametrize("bad", [0.1, 2.5, "hot", None])
def test_intensity_out_of_range_refuses(tmp_path, bad):
    with pytest.raises(Refuse):
        _load(tmp_path, intensity=bad)


def test_burst_scales_ember_count_and_size():
    lo, one, hi = FX.burst(0, 0.25), FX.burst(0), FX.burst(0, 2.0)
    assert lo.n < one.n < hi.n
    assert one.n == FX.FIRE_EMBERS and hi.n == 2 * FX.FIRE_EMBERS and lo.n == round(0.25 * FX.FIRE_EMBERS)


def test_plan_carries_the_intensity_per_effect():
    boxes = np.tile(np.float32([100, 200, 300, 400]), (300, 1))
    arr, _, _ = FX.plan([{**FIRE, "at_s": 1.0, "intensity": 1.6}, {**FIRE, "at_s": 5.0, "intensity": 0.5}],
                        0.0, 300, lambda tr: boxes, 30)
    assert arr["fx_int"].tolist() == pytest.approx([1.6, 0.5])
