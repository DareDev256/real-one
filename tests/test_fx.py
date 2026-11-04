"""Tracked effects (realone/fx.py, --edits, Nov 4): name a part in words, SAM 3.1 tracks it, the effect follows it."""
import json

import numpy as np
import pytest

from realone import fx as FX
from realone.common import Refuse

N, FPS, S0 = 300, 30, 20.0


def _edits(tmp_path, fx):
    p = tmp_path / "edits.json"
    p.write_text(json.dumps({"fx": fx}))
    return p


def test_the_edit_file_is_checked_before_anything_is_called(tmp_path):
    ok = FX.load_edits(_edits(tmp_path, [{"track": "his hands", "at_s": 26.7, "effect": "fire"}]))
    assert ok == [{"track": "his hands", "at_s": 26.7, "effect": "fire", "seconds": 1.4}]
    for bad, says in ([{"at_s": 26.7}], "names nothing"), ([{"track": "x", "effect": "fire"}], "needs at_s"), \
            ([{"track": "x", "at_s": 1, "effect": "laser"}], "one of fire"), \
            ([{"track": "x", "at_s": 1, "seconds": 9}], "lasts 9"), ([], "no 'fx'"):
        with pytest.raises(Refuse, match=says):
            FX.load_edits(_edits(tmp_path, bad))


def _hands(found=1.0):
    """His hands moving right across the frame; missing on a share of frames."""
    b = np.zeros((N, 4))
    x = np.linspace(200, 700, N)
    b[:, 0], b[:, 1], b[:, 2], b[:, 3] = x, 800, x + 150, 950
    rng = np.random.default_rng(3)
    b[rng.uniform(0, 1, N) > found] = np.nan
    return b


def test_the_effect_follows_the_part_on_the_song_clock():
    fx = [{"track": "his hands", "at_s": 22.0, "effect": "fire", "seconds": 1.4}]
    arr, designed, rep = FX.plan(fx, S0, N, lambda tr: _hands(0.9), FPS)
    f0 = int(round((22.0 - S0) * FPS))
    f1 = f0 + int(round(1.4 * FPS))
    assert arr["fx_age"][f0] == 0 and arr["fx_age"][f1 - 1] == f1 - 1 - f0 and arr["fx_age"][f1] == -1
    assert (arr["fx_id"][f0:f1] == 0).all() and (arr["fx_id"][:f0] == -1).all()
    x = arr["fx_box"][f0:f1, 0]
    assert not np.isnan(x).any() and np.all(np.diff(x) >= 0)   # held through matte blinks, moving with his hands
    assert {f0, f0 + 1} <= designed and rep[0]["placed"] and rep[0]["frames"] == [f0, f1 - 1]


def test_fire_on_nothing_is_refused_and_an_effect_elsewhere_is_reported():
    fx = [{"track": "his hands", "at_s": 22.0, "effect": "fire", "seconds": 1.4}]
    with pytest.raises(Refuse, match="found 'his hands' on"):
        FX.plan(fx, S0, N, lambda tr: _hands(0.3), FPS)
    out = [{"track": "his hands", "at_s": 5.0, "effect": "fire", "seconds": 1.4}]   # before this render starts
    arr, _, rep = FX.plan(out, S0, N, lambda tr: _hands(), FPS)
    assert not rep[0]["placed"] and (arr["fx_age"] == -1).all()


def test_an_edits_fire_is_bigger_than_a_lyric_ignite_and_glows_on_the_part():
    from realone import moments as MO
    b = FX.burst(0)
    assert b.n == FX.FIRE_EMBERS > MO.EMBERS and b.size.mean() > MO.EmberBurst(0).size.mean() * 1.5
    world = np.zeros((400, 300, 3), np.float32)
    lit = FX.part_glow(world, (100, 150, 160, 250), age=6)
    assert lit[200, 130].sum() > lit[20, 10].sum() + 50     # warm at the hands, dark far away
    assert FX.part_glow(world, (100, 150, 160, 250), age=200).max() == 0  # gone once the fire has burnt out
