"""The literal-lyrics guard and the treatment schema. Pure functions: no model, no network, no key."""
import copy
import json

import pytest

from conftest import REPO
from realone.director import guard
from realone.director.schema import (TREATMENT_SCHEMA, check_treatment, parse_json, people_in, validate,
                                     wire_schema)

FIX = REPO / "tests" / "fixtures" / "director"

# Lyric lines written for these tests (not from any real song).
HOOK = """I'm good, I'm good, don't call it
Keys in my hand but I ain't going
I'm good, I'm good, don't call it
Lights on at four and I'm still holding"""
WHIP = """Pull up in the whip with the top down
Money on my mind and the city in my palm"""
RAIN = "Dancing in the rain till the morning comes"


def sections():
    mk = lambda i, role, hook: {"index": i, "role": role, "is_hook": hook}  # noqa: E731
    return [mk(0, "intro", False), mk(1, "verse", False), mk(2, "hook", True), mk(3, "bridge", False),
            mk(4, "hook", True)]


def fixture_treatment(name="treatment_2_literal.json"):
    return copy.deepcopy(json.loads((FIX / name).read_text())["response"])


# ------------------------------------------------------------------------------------------------- guard

@pytest.mark.parametrize("prompt, lyric, hit", [
    ("a close-up of a wristwatch ticking on a busy street", "Eyes on me, watch me move", "watch->wristwatch"),
    ("a hand gripping car keys under bright kitchen lights at four in the morning", HOOK, "key"),
    ("a red convertible with the roof down cruising a coastal road at dusk", WHIP, "whip->convertible"),
    ("stacks of cash on a glass table in a penthouse", WHIP, "money->cash"),
    ("a rain-soaked street, a dancer's shoes in a puddle", RAIN, "rain"),
])
def test_literal_prompts_are_rejected(prompt, lyric, hit):
    r = guard.check(prompt, lyric)
    assert r["verdict"] == "reject", r
    assert hit in r["concrete"], r


@pytest.mark.parametrize("prompt, lyric", [
    ("an empty laundromat after midnight, one dryer still turning, fluorescent tubes humming over rows of plastic "
     "chairs, rain streaking the window", HOOK),
    ("a closed amusement park at dawn, rides covered in tarps, puddles reflecting a pale sky", WHIP),
    ("an empty train platform at dusk, rocky terrain beyond the last stop", RAIN),  # t+rain, ter+rain: no compound
    ("a stone hearth in an empty cabin, snow against the glass", "You broke my heart in two"),
])
def test_subtext_prompts_pass(prompt, lyric):
    r = guard.check(prompt, lyric)
    assert r["verdict"] == "pass", r
    assert r["concrete"] == [] and r["score"] <= guard.THRESHOLD


def test_score_is_exposed_and_the_threshold_decides():
    # no concrete nouns here, so only the overlap score can reject
    lyric = "I keep running from the feeling, every promise sounds the same"
    over = guard.check("running promise feeling quiet", lyric)        # 3 of 4 prompt words come from the lyric
    under = guard.check("a quiet gymnasium, bleachers folded, one promise painted on a banner, dust in the light",
                        lyric)                                          # 1 shared word of many
    assert over["verdict"] == "reject" and over["score"] == pytest.approx(0.75) and over["concrete"] == []
    assert under["verdict"] == "pass" and 0 < under["score"] <= guard.THRESHOLD
    assert set(over) >= {"verdict", "score", "threshold", "shared", "concrete"}


def test_instrumental_sections_have_no_lyric_to_copy():
    assert guard.check("a dark stairwell, one exit sign", "")["verdict"] == "no_lyric"


def test_reask_names_the_offending_words():
    tr = fixture_treatment()
    lyr = {2: HOOK.split("\n"), 1: ["Phone face down on the kitchen counter"]}
    flagged = {i: r for i, r in guard.check_treatment(tr, lyr).items() if r["verdict"] == "reject"}
    assert sorted(flagged) == [1, 2]
    msg = guard.reask_message(flagged, tr, lyr)
    for w in ("key", "hand", "phone", "kitchen", "counter"):
        assert w in msg
    assert "section 2" in msg and "section 1" in msg and "rewrites" in msg


# ------------------------------------------------------------------------------------------------- schema

def test_valid_fixture_treatment_passes():
    tr, errs = check_treatment(fixture_treatment(), sections())
    assert errs == [] and tr is not None


def test_every_error_is_reported_at_once():
    obj = json.loads((FIX / "treatment_1_invalid.json").read_text())["response"]
    tr, errs = check_treatment(json.dumps(obj), sections())
    assert tr is None
    text = "\n".join(errs)
    assert "$.sections[0].why: missing" in text
    assert "'grainy' is not one of" in text
    assert "'teal' does not match" in text
    assert "true on section 1, which is not a hook" in text


@pytest.mark.parametrize("mutate, expect", [
    (lambda t: t["sections"].pop(), "indices must be exactly [0, 1, 2, 3, 4]"),
    (lambda t: t["sections"][2].update(freeze=True), "exactly one hook"),
    (lambda t: t["sections"][4].update(freeze=False), "exactly one hook"),
    (lambda t: t["sections"][3].update(world_prompt="a crowd of dancers in a neon warehouse at night"), "people"),
    (lambda t: t["sections"][0].update(extra=1), "not allowed"),
    (lambda t: t["sections"][1].update(palette=["#FFFFFF"]), "at least 3"),
])
def test_schema_and_semantic_rules(mutate, expect):
    t = fixture_treatment()
    mutate(t)
    _, errs = check_treatment(t, sections())
    assert any(expect in e for e in errs), errs


def test_parse_json_tolerates_a_fence_and_reports_garbage():
    assert parse_json('```json\n{"a": 1}\n```')[0] == {"a": 1}
    obj, errs = parse_json("not json at all")
    assert obj is None and "not valid JSON" in errs[0]


def test_wire_schema_keeps_structure_only():
    w = wire_schema(TREATMENT_SCHEMA)
    sec = w["properties"]["sections"]["items"]
    assert sec["additionalProperties"] is False and set(sec["required"]) == set(sec["properties"])
    blob = json.dumps(w)
    assert "minLength" not in blob and "pattern" not in blob and "minimum" not in blob
    assert validate({"a": True}, {"type": "object", "properties": {"a": {"type": "integer"}}})  # bool is not int


def test_people_words():
    assert people_in("an empty hotel lobby nobody checks into") == []
    assert people_in("a woman waiting at a bus stop") == ["woman"]


def test_moves_are_part_of_every_section_and_follow_the_rules():
    t = fixture_treatment()
    secs = sections()
    ok, errs = check_treatment(copy.deepcopy(t), secs)
    base = len(errs)
    # a missing move is a schema error
    t2 = copy.deepcopy(t)
    del t2["sections"][0]["move"]
    _, e2 = check_treatment(t2, secs)
    assert any("move" in e for e in e2) and len(e2) > base
    # at most one dolly in a song
    t3 = copy.deepcopy(t)
    for s in t3["sections"][:2]:
        s["move"] = "dolly"
    _, e3 = check_treatment(t3, secs)
    assert any("dolly at most once" in e for e in e3)
    # never a swap on the frozen hook
    t4 = copy.deepcopy(t)
    frozen = next(i for i, s in enumerate(t4["sections"]) if s.get("freeze") is True)
    t4["sections"][frozen]["move"] = "swap"
    _, e4 = check_treatment(t4, secs)
    assert any("swap on the frozen hook" in e for e in e4)
    # an unknown move is refused by the schema
    t5 = copy.deepcopy(t)
    t5["sections"][0]["move"] = "orbit"
    _, e5 = check_treatment(t5, secs)
    assert any("move" in e for e in e5) and len(e5) > base
