"""The director's lyric-moment pick (realone/director/moments.py): only words sung in the section they name, at most
2 a section and 8 a song, never a word on the violence list; a reply that does not validate refuses."""
import json

import pytest

from realone.common import Refuse
from realone.director import moments as DM

REC = {"song_subtext": "pretending to be fine", "sections": [
    {"index": 0, "role": "verse", "lyric": ["Phone face down", "Rent due Friday, get burnt up"], "subtext": "a"},
    {"index": 1, "role": "hook", "is_hook": True, "lyric": ["I'm good, I'm good, don't call it"], "subtext": "b"},
    {"index": 2, "role": "verse", "lyric": ["got the glock on me"], "subtext": "c"}]}


class _Backend:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def respond(self, **kw):
        self.calls.append(kw)
        return type("T", (), {"text": self.reply, "source": "fixture", "id": "r1", "usd": 0.001})()


def _m(section, word, effect="slam"):
    return {"section": section, "word": word, "effect": effect, "why": "because"}


def test_keeps_sung_words_and_drops_the_rest():
    kept, dropped = DM.check({"picks": [_m(0, "burnt", "ignite"), _m(1, "good"), _m(1, "Friday"),
                                          _m(2, "glock"), _m(5, "good"), _m(1, "two words")]}, REC)
    assert kept == {0: [{"word": "burnt", "effect": "ignite", "why": "because"}],
                    1: [{"word": "good", "effect": "slam", "why": "because"}]}
    whys = [d["why"] for d in dropped]
    assert any("not sung in section 1" in w for w in whys)
    assert any("violence" in w for w in whys)
    assert any("not in the treatment" in w for w in whys)
    assert any("single word" in w for w in whys)


def test_caps_per_section_and_song():
    kept, dropped = DM.check({"picks": [_m(1, "good"), _m(1, "call"), _m(1, "don't")]}, REC)
    assert len(kept[1]) == DM.MAX_PER_SECTION and "already has" in dropped[0]["why"]
    many = {"sections": [{"index": i, "lyric": ["fire"]} for i in range(12)]}
    kept, dropped = DM.check({"picks": [_m(i, "fire", "ignite") for i in range(12)]}, many)
    assert sum(len(v) for v in kept.values()) == DM.MAX_TOTAL and len(dropped) == 12 - DM.MAX_TOTAL


def test_pick_writes_moments_onto_every_section():
    be = _Backend(json.dumps({"picks": [_m(1, "good")], "why": "the hook is the lie"}))
    out = DM.pick(be, REC)
    assert [s["moments"] for s in out["sections"]] == [[], [{"word": "good", "effect": "slam", "why": "because"}], []]
    assert out["moments_pick"]["kept"] == 1 and out["moments_pick"]["source"] == "fixture"
    assert be.calls[0]["schema_name"] == "lyric_moments"
    assert "Kendrick" in be.calls[0]["instructions"]
    assert REC["sections"][1].get("moments") is None  # the input record is not changed


def test_invalid_reply_refuses():
    with pytest.raises(Refuse):
        DM.pick(_Backend("not json"), REC)
    with pytest.raises(Refuse):
        DM.pick(_Backend(json.dumps({"picks": [{"section": 1, "word": "good"}], "why": ""})), REC)


def test_every_wire_schema_requires_only_defined_keys():
    """Strict structured output refuses a schema whose `required` names a key missing from `properties` (HTTP 400,
    seen live Nov 2 when a LOCAL_ONLY name was stripped from properties but kept in required)."""
    from realone.director import schema as S

    def walk(node, path="$"):
        if isinstance(node, dict):
            if "required" in node:
                missing = set(node["required"]) - set(node.get("properties", {}))
                assert not missing, f"{path}: required {sorted(missing)} not in properties"
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for v in node:
                walk(v, path)

    for name in ("TREATMENT_SCHEMA", "REWRITE_SCHEMA", "NOTES_SCHEMA", "DECISION_SCHEMA"):
        walk(S.wire_schema(getattr(S, name)), name)
    walk(S.wire_schema(DM.MOMENTS_SCHEMA), "MOMENTS_SCHEMA")
    assert "picks" in S.wire_schema(DM.MOMENTS_SCHEMA)["properties"]
