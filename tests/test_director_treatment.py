"""Treatment: schema validation with one repair turn, the guard's one re-ask, budget and key refusals.
Every model response here is a FIXTURE (inline or from tests/fixtures/director/)."""
import copy
import json

import pytest

from conftest import REPO
from realone.common import Refuse
from realone.director import budget, treatment as trt
from realone.director.backend import AudioRejected, FixtureBackend

FIX = REPO / "tests" / "fixtures" / "director"


def timeline():
    rows = [(0, "intro", False, 0.4, 10.2), (1, "verse", False, 10.2, 29.79), (2, "hook", True, 29.79, 49.38),
            (3, "bridge", False, 49.38, 68.97), (4, "hook", True, 68.97, 88.57)]
    return {"duration_s": 91.06, "bpm": 98.0, "beats_per_bar": 4,
            "sections": [{"index": i, "role": r, "is_hook": h, "start_s": a, "end_s": b, "bars": 8}
                         for i, r, h, a, b in rows]}


def lyrics():
    lines = {1: ["Told my mama I'm good, I'm good", "Phone face down on the kitchen counter"],
             2: ["I'm good, I'm good, don't call it", "Keys in my hand but I ain't going",
                 "Lights on at four and I'm still holding"],
             3: ["Bus pass, two jobs, one dream"], 4: ["I'm good, I'm good, don't call it"]}
    return {"sections": [{"index": s["index"], "lines": lines.get(s["index"], []),
                          "source": "artist" if s["index"] in lines else "none"} for s in timeline()["sections"]]}


def fx(name):
    return json.loads((FIX / name).read_text())


def clean_treatment():
    t = copy.deepcopy(fx("treatment_2_literal.json")["response"])
    for rw in fx("guard_rewrites.json")["response"]["rewrites"]:
        t["sections"][rw["index"]]["world_prompt"] = rw["world_prompt"]
    return t


def test_a_valid_treatment_needs_no_repair_and_no_reask():
    fb = FixtureBackend({"treatment": [{"response": clean_treatment()}]})
    res = trt.make(fb, timeline(), lyrics(), audio_file_id="file-1")
    assert len(res["attempts"]) == 1 and res["attempts"][0]["errors"] == []
    assert res["guard"]["reask"] is None and res["guard"]["still_literal"] == []
    req = fb.requests[0]
    assert req["schema_name"] == "real_one_treatment" and req["effort"] == "high"
    parts = req["input"][0]["content"]
    assert {"type": "input_audio", "file_id": "file-1"} in parts       # the song goes in as audio


def test_invalid_json_gets_one_repair_turn_that_names_the_errors():
    fb = FixtureBackend({"treatment": ["treatment_1_invalid.json"], "treatment_repair": ["treatment_2_literal.json"],
                         "guard_reask": ["guard_rewrites.json"]})
    res = trt.make(fb, timeline(), lyrics())
    assert [a["purpose"] for a in res["attempts"]] == ["treatment", "treatment_repair"]
    assert res["attempts"][1]["errors"] == []
    repair = fb.requests[1]
    assert repair["purpose"] == "treatment_repair" and repair["previous_id"] == "resp_FIXTURE_1"
    msg = repair["input"][0]["content"][0]["text"]
    assert "'grainy' is not one of" in msg and "$.sections[0].why: missing" in msg and "not a hook" in msg


def test_a_second_invalid_treatment_refuses():
    bad = fx("treatment_1_invalid.json")
    fb = FixtureBackend({"treatment": [bad], "treatment_repair": [bad]})
    with pytest.raises(Refuse, match="failed validation twice"):
        trt.make(fb, timeline(), lyrics())
    assert len(fb.requests) == 2                                         # exactly one repair, no third call


def test_the_guard_reasks_once_and_only_flagged_sections_change():
    rewrites = copy.deepcopy(fx("guard_rewrites.json")["response"])
    rewrites["rewrites"].append({"index": 3, "world_prompt": "a different break room that nobody flagged at all",
                                 "why": "sneaking in an unrequested change"})
    fb = FixtureBackend({"treatment": ["treatment_2_literal.json"], "guard_reask": [{"response": rewrites}]})
    res = trt.make(fb, timeline(), lyrics())
    g = res["guard"]
    assert sorted(g["reask"]["asked"]) == [1, 2] and sorted(g["reask"]["applied"]) == [1, 2]
    assert g["reask"]["ignored"][0]["index"] == 3
    assert g["first_pass"][2]["verdict"] == "reject" and g["final"][2]["verdict"] == "pass"
    assert "laundromat" in res["treatment"]["sections"][2]["world_prompt"]
    assert "break room, a vending" in res["treatment"]["sections"][3]["world_prompt"]  # unchanged
    msg = fb.requests[1]["input"][0]["content"][0]["text"]
    assert "avoid: hand, key" in msg or "avoid: key, hand" in msg
    assert len([r for r in fb.requests if r["purpose"] == "guard_reask"]) == 1


def test_still_literal_after_the_reask_is_reported_not_hidden():
    still = {"rewrites": [{"index": 2, "world_prompt": "car keys on a hook by the door, kitchen lights on at four",
                           "why": "still literal"}]}
    fb = FixtureBackend({"treatment": ["treatment_2_literal.json"], "guard_reask": [{"response": still}]})
    res = trt.make(fb, timeline(), lyrics())
    assert res["guard"]["still_literal"] == [1, 2]                       # 1 got no rewrite, 2 got a literal one


def test_audio_rejected_by_the_api_falls_back_loudly():
    class Once(FixtureBackend):
        def respond(self, **kw):
            if any(p.get("type") == "input_audio" for p in kw["input"][0]["content"]):
                raise AudioRejected("input_audio is not supported for this model")
            return super().respond(**kw)
    fb = Once({"treatment": [{"response": clean_treatment()}]})
    res = trt.make(fb, timeline(), lyrics(), audio_file_id="file-1")
    assert res["audio_in"].startswith("REJECTED") and res["attempts"][0]["errors"] == []


def test_a_hand_edited_treatment_is_rechecked_without_a_model():
    res = trt.make(FixtureBackend({"treatment": [{"response": clean_treatment()}]}), timeline(), lyrics())
    rec = trt.record(res, timeline(), lyrics(), {"mode": "test"})
    rec["sections"][2]["world_prompt"] = "a wristwatch and car keys on a kitchen table"
    again = trt.from_record(rec, timeline(), lyrics())
    assert again["guard"]["still_literal"] == [2]
    rec["sections"][2]["freeze"] = True                                   # two frozen hooks
    with pytest.raises(Refuse, match="exactly one hook"):
        trt.from_record(rec, timeline(), lyrics())


# ------------------------------------------------------------------------------------------------- budget

def test_quote_over_the_cap_refuses_without_yes_spend():
    q = budget.quote(song_s=180, n_sections=8, brief_chars=7000, direct=True, covered_s=[19.6] * 8)
    assert q["ceiling_usd"] > budget.DEFAULT_CAP_USD
    with pytest.raises(Refuse, match="over the cap"):
        budget.gate(q, budget.DEFAULT_CAP_USD, yes_spend=False)
    assert budget.gate(q, budget.DEFAULT_CAP_USD, yes_spend=True) == q["ceiling_usd"]
    small = budget.quote(song_s=91, n_sections=5, brief_chars=4000)
    assert budget.gate(small, budget.DEFAULT_CAP_USD, yes_spend=False) == budget.DEFAULT_CAP_USD
    assert q["ceiling_cad"] == pytest.approx(q["ceiling_usd"] * budget.USD_TO_CAD, abs=0.01)


def test_the_meter_stops_before_a_call_that_would_pass_a_limit():
    m = budget.Meter(cap_usd=0.50, max_calls=3)
    m.before("a", 0.20)
    m.charge("a", 0.20)
    with pytest.raises(Refuse, match="passes the cap"):
        m.before("b", 0.31)                                               # 0.20 + 0.31 > 0.50
    m.before("c", 0.10)
    m.before("d", 0.0)
    with pytest.raises(Refuse, match="call cap reached"):
        m.before("e", 0.0)
    assert m.calls == 3


def test_documented_prices():
    assert budget.wonder_usd(1_000_000, 0) == pytest.approx(4.00)
    assert budget.wonder_usd(1_000_000, 0, cached_tokens=1_000_000) == pytest.approx(0.20)
    assert budget.wonder_usd(0, 1_000_000) == pytest.approx(20.00)
    assert budget.VOICE_USD_PER_S * 3600 == pytest.approx(0.18)


def test_missing_key_refuses(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(Refuse, match="MODEL_API_KEY is not set.*Nothing was called"):
        budget.require_key()
