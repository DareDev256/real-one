"""--from-treatment: the director's treatment drives the render (worlds, moves, freeze, receipt reveal)."""
import json
from types import SimpleNamespace

import pytest

from realone.common import Refuse
from realone.render import _treatment_specs, apply_treatment


def _t(tmp_path):
    t = {"song_subtext": "x" * 20, "sections": [
        {"index": 0, "world_prompt": "an empty hallway before sunrise", "look": "clean", "freeze": False, "move": "none"},
        {"index": 2, "world_prompt": "a deserted parking structure", "look": "clean", "freeze": False, "move": "dolly"},
        {"index": 4, "world_prompt": "the same structure at pale morning", "look": "receipt", "freeze": True,
         "move": "none"}]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"treatment": t}))
    return p


def test_the_treatment_fills_worlds_moves_freeze_and_reveal(tmp_path):
    a = SimpleNamespace(from_treatment=str(_t(tmp_path)), section_world=None, moves_from=None)
    apply_treatment(a)
    assert a.section_world == ["#0=an empty hallway before sunrise", "#2=a deserted parking structure",
                               "#4=the same structure at pale morning"]
    assert a.moves_from.endswith("t.json")
    assert a._treatment_freeze == [4] and a._treatment_reveal == [4]
    # only the sections being rendered take the freeze / reveal
    assert _treatment_specs(a, a._treatment_freeze, [{"index": 2}]) == []
    assert _treatment_specs(a, a._treatment_freeze, [{"index": 2}, {"index": 4}]) == ["#4"]


def test_a_bad_treatment_is_refused(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"sections": [{"index": 1}]}))
    with pytest.raises(Refuse, match="world_prompt"):
        apply_treatment(SimpleNamespace(from_treatment=str(p), section_world=None, moves_from=None))
    with pytest.raises(Refuse, match="not found"):
        apply_treatment(SimpleNamespace(from_treatment=str(tmp_path / "nope.json"), section_world=None, moves_from=None))
    apply_treatment(SimpleNamespace(from_treatment=None))  # no flag: nothing happens
