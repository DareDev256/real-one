"""Tool dispatch and the rules the code enforces on the agent: one revision pass, the treatment as the single
source of truth, renders capped and only on covered sections, watch only on this run's renders, budget refusals
that end the run. The renderer is a stub that writes a run report shaped like realone/render.py's."""
import copy
import json

import pytest

from conftest import REPO
from realone.common import Refuse
from realone.director import budget, loop
from realone.director.backend import FixtureBackend
from realone.director.tools import TOOL_SPECS, Renderer, State, Tools, detect_freeze

FIX = REPO / "tests" / "fixtures" / "director"
HELP_010 = "usage: realone render [-h] --clip CLIP --song SONG [--section SECTION] [--look LOOK] [--max-usd MAX_USD]"
HELP_FLAG = HELP_010 + " [--freeze]\n  --freeze   freeze the world on the hook"
HELP_VALUE = HELP_010 + " [--freeze FREEZE]\n  --freeze FREEZE   seconds"


def timeline():
    rows = [(0, "intro", False, 0.4035, 10.1993), (1, "verse", False, 10.1993, 29.7909),
            (2, "hook", True, 29.7909, 49.3825), (3, "bridge", False, 49.3825, 68.9742),
            (4, "hook", True, 68.9742, 88.5658)]
    return {"duration_s": 91.06, "bpm": 98.0, "beats_per_bar": 4,
            "sections": [{"index": i, "role": r, "is_hook": h, "start_s": a, "end_s": b} for i, r, h, a, b in rows]}


def treatment():
    t = copy.deepcopy(json.loads((FIX / "treatment_2_literal.json").read_text())["response"])
    for rw in json.loads((FIX / "guard_rewrites.json").read_text())["response"]["rewrites"]:
        t["sections"][rw["index"]]["world_prompt"] = rw["world_prompt"]
    return t


LYRICS = {1: ["Phone face down on the kitchen counter"],
          2: ["I'm good, I'm good, don't call it", "Keys in my hand but I ain't going",
              "Lights on at four and I'm still holding"],
          4: ["I'm good, I'm good, don't call it"]}


class StubRenderer(Renderer):
    def __init__(self, out, help_text=HELP_010, dry_run=True, rc=0):
        super().__init__(out / "take.mp4", out / "song.wav", dry_run=dry_run, help_text=help_text, runner=self.fake)
        self.out, self.rc, self.calls = out, rc, []

    def run_json(self, tag, section):
        return self.out / f"{'TEST_LOCAL_' if self.dry_run else ''}{tag}_s{section}_run.json"

    def fake(self, argv):
        self.calls.append(argv)
        if self.rc:
            return self.rc, "realone: REFUSED: stub refusal"
        tag, sec, look = argv[argv.index("--tag") + 1], int(argv[argv.index("--section") + 1][1:]), \
            argv[argv.index("--look") + 1]
        video = self.out / f"{tag}_s{sec}_{look}.mp4"
        video.write_bytes(b"stub")
        s = timeline()["sections"][sec]
        self.run_json(tag, sec).write_text(json.dumps({
            "section": {"start_s": s["start_s"], "end_s": s["end_s"]}, "engine": "stub", "billed_usd": 0.0,
            "looks": {look: {"video": str(video), "spec_ok": True, "cuts_on_plan": "3/3", "cuts_off_plan": [],
                             "audio_vs_master_error_ms": 0.4, "cuts_detected": [147, 294, 441]}}}))
        return 0, "ok"


def make(tmp_path, script=None, dry_run=True, covered=(2,), help_text=HELP_010, meter=None):
    st = State(timeline=timeline(), lyrics=LYRICS, treatment=treatment(), covered=list(covered), tag="t",
               dry_run=dry_run)
    fb = FixtureBackend(script or {})
    r = StubRenderer(tmp_path, help_text=help_text, dry_run=dry_run)
    return Tools(fb, st, r, meter), st, r, fb


WATCH_REVISE = {"response": {"verdict": "revise", "notes": [{"t_s": 9.8, "against": "look", "severity": "high",
                                                             "note": "the boil fights the stillness"}],
                             "suggestion": {"field": "look", "value": "receipt", "why": "only the artist in colour"}}}


def test_specs_meet_the_strict_subset():
    for t in TOOL_SPECS:
        p = t["parameters"]
        assert p["type"] == "object" and p["additionalProperties"] is False
        assert sorted(p["required"]) == sorted(p["properties"]), t["name"]
        assert t["name"].count(".") <= 1 and t["strict"] is True


def test_unknown_tools_and_bad_arguments_come_back_as_errors(tmp_path):
    tools, st, r, _ = make(tmp_path)
    assert "unknown tool" in tools.dispatch("fly", "{}")["error"]
    assert "look: missing" in tools.dispatch("render_section", '{"section": 2, "freeze": false}')["error"]
    assert "not valid JSON" in tools.dispatch("watch", "{not json")["error"]
    assert "'sepia' is not one of" in tools.dispatch("render_section", {"section": 2, "look": "sepia",
                                                                        "freeze": False})["error"]
    assert r.calls == [] and len(st.log) == 4


def test_generate_world_uses_the_treatment_and_the_guard(tmp_path):
    tools, st, _, _ = make(tmp_path)
    assert "not section 2's world_prompt" in tools.dispatch(
        "generate_world", {"section": 2, "prompt": "an empty parking garage at dawn"})["error"]
    st.plan(2)["world_prompt"] = "car keys in a hand under kitchen lights at four"   # forced past revise on purpose
    err = tools.dispatch("generate_world", {"section": 2, "prompt": st.plan(2)["world_prompt"]})["error"]
    assert "literal-lyrics guard" in err and "key" in err
    st.plan(2)["world_prompt"] = treatment()["sections"][2]["world_prompt"]
    ok = tools.dispatch("generate_world", {"section": 2, "prompt": st.plan(2)["world_prompt"]})
    assert ok["ok"] and ok["world"] is None and ok["note"].startswith("FIXTURE")


def test_dry_run_render_argv_uses_only_test_flags(tmp_path):
    tools, st, r, _ = make(tmp_path)
    out = tools.dispatch("render_section", {"section": 2, "look": "collage", "freeze": False})
    assert out["ok"], out
    argv = r.calls[0]
    assert argv[argv.index("--section") + 1] == "#2" and argv[argv.index("--tag") + 1] == "t_r0"
    assert "--local-plate" in argv and argv[argv.index("--mask") + 1] == "local"
    assert "--world" not in argv and "--freeze" not in argv
    assert out["measured"]["cuts_on_plan"] == "3/3" and out["measured"]["cuts_s"] == [4.9, 9.8, 14.7]


def test_keyed_argv_carries_the_world_the_cap_and_a_detected_freeze(tmp_path):
    r = StubRenderer(tmp_path, help_text=HELP_FLAG, dry_run=False)
    argv, note = r.argv(4, "clean", True, "a rooftop water tower at blue hour", "t_r0", 1.2345)
    assert argv[argv.index("--mask") + 1] == "sam" and argv[argv.index("--world") + 1].startswith("a rooftop")
    assert argv[argv.index("--max-usd") + 1] == "1.2345" and argv[-1] == "--freeze"
    argv, note = StubRenderer(tmp_path, help_text=HELP_010, dry_run=False).argv(4, "clean", True, "x", "t", 1.0)
    assert "--freeze" not in argv and "no freeze option" in note


@pytest.mark.parametrize("text, flag, says", [
    (HELP_010, None, "no freeze option"), (HELP_FLAG, "--freeze", "--freeze"),
    (HELP_VALUE, None, "does not guess"), (HELP_010 + " [--hook-freeze]", "--hook-freeze", "--hook-freeze"),
    (HELP_010 + " [--freeze [SECTION]] [--freeze-bars N]\n  --freeze [SECTION]  bare = the rendered section\n"
                "  --freeze-bars N  length in bars", "--freeze", "--freeze")])
def test_freeze_flag_detection(text, flag, says):
    f, note = detect_freeze(text)
    assert f == flag and says in note


def test_renders_refuse_uncovered_mismatched_and_failed(tmp_path):
    tools, st, r, _ = make(tmp_path)
    assert "does not cover section 1" in tools.dispatch("render_section", {"section": 1, "look": "clean",
                                                                           "freeze": False})["error"]
    assert "change it with revise" in tools.dispatch("render_section", {"section": 2, "look": "clean",
                                                                        "freeze": False})["error"]
    r.rc = 2
    assert "exit 2" in tools.dispatch("render_section", {"section": 2, "look": "collage", "freeze": False})["error"]


def test_one_revision_pass_then_closed(tmp_path):
    tools, st, r, fb = make(tmp_path, {"watch": [WATCH_REVISE]})
    first = tools.dispatch("render_section", {"section": 2, "look": "collage", "freeze": False})
    assert tools.dispatch("watch", {"render_path": first["video"]})["verdict"] == "revise"
    assert "already rendered and nothing changed" in tools.dispatch(
        "render_section", {"section": 2, "look": "collage", "freeze": False})["error"]
    a = tools.dispatch("revise", {"section": 2, "field": "look", "value": "receipt", "because": "note at 9.8 s"})
    b = tools.dispatch("revise", {"section": 2, "field": "freeze", "value": "true", "because": "hold the first hook"})
    assert a["ok"] and b["ok"] and b["changes_left"] == 1
    assert [s["freeze"] for s in st.treatment["sections"]] == [False, False, True, False, False]
    assert sorted(st.revision["changed"]) == [2, 4]
    again = tools.dispatch("render_section", {"section": 2, "look": "receipt", "freeze": True})
    assert again["ok"] and again["tag"] == "t_r1" and st.revision["state"] == "closed"
    assert "pass is spent" in tools.dispatch("revise", {"section": 2, "field": "look", "value": "clean",
                                                        "because": "x"})["error"]
    assert "already rendered" in tools.dispatch("render_section", {"section": 2, "look": "receipt",
                                                                   "freeze": True})["error"]
    assert len(r.calls) == 2                                             # first cut + its revision, no third


def test_the_pass_holds_at_most_three_changes_and_checks_each(tmp_path):
    tools, st, _, _ = make(tmp_path, covered=(1, 2, 3))
    bad = tools.dispatch("revise", {"section": 1, "field": "world_prompt", "value": "a phone glowing on a kitchen "
                                    "counter at night, an empty stool beside it", "because": "x"})
    assert "literal-lyrics guard rejects it" in bad["error"] and "phone" in bad["error"]
    assert st.revision["changes"] == []                                  # a refused change does not count
    assert "describes people" in tools.dispatch("revise", {"section": 3, "field": "world_prompt", "value":
                                                           "a woman alone in an empty office at night, blinds half "
                                                           "open", "because": "x"})["error"]
    assert "not a hook" in tools.dispatch("revise", {"section": 1, "field": "freeze", "value": "true",
                                                     "because": "x"})["error"]
    for i, look in ((1, "collage"), (2, "receipt"), (3, "clean")):
        assert tools.dispatch("revise", {"section": i, "field": "look", "value": look, "because": "x"})["ok"]
    assert "already holds 3 changes" in tools.dispatch("revise", {"section": 0, "field": "look", "value": "receipt",
                                                                  "because": "x"})["error"]


def test_watch_only_takes_this_runs_renders(tmp_path):
    tools, st, _, fb = make(tmp_path, {"watch": [WATCH_REVISE]})
    assert "only takes a video that render_section returned" in tools.dispatch(
        "watch", {"render_path": "/etc/hosts"})["error"]
    assert fb.uploads == []                                              # nothing was sent anywhere
    tools.dispatch("render_section", {"section": 2, "look": "collage", "freeze": False})
    fb.fixture = False                                                   # a real model may not use @render refs
    assert "only takes" in tools.dispatch("watch", {"render_path": "@render:2"})["error"]
    fb.fixture = True
    w = tools.dispatch("watch", {"render_path": "@render:2"})
    assert w["ok"] and w["source"] == "FIXTURE:inline" and w["measured"]["cuts_on_plan"] == "3/3"
    sent = fb.requests[-1]["input"][0]["content"]
    assert sent[1] == {"type": "input_video", "file_id": "file-FIXTURE-1"}


def test_a_budget_refusal_ends_the_run_instead_of_returning_to_the_model(tmp_path):
    meter = budget.Meter(cap_usd=1.0, max_calls=0)
    tools, st, r, _ = make(tmp_path, dry_run=False, meter=meter)
    tools.dispatch("generate_world", {"section": 2, "prompt": st.plan(2)["world_prompt"]})
    st.worlds[2]["prompt"] = " ".join(st.plan(2)["world_prompt"].split())
    with pytest.raises(Refuse, match="call cap reached"):
        tools.dispatch("render_section", {"section": 2, "look": "collage", "freeze": False})
    assert r.calls == []


def test_model_driver_runs_the_fixture_script_to_finish(tmp_path):
    tools, st, r, fb = make(tmp_path, "dry_run_script.json")
    end = loop.run_model(fb, tools, st, max_turns=8)
    assert end == "finished" and st.finished.startswith("Rendered the first hook")
    assert [x["look"] for x in st.renders[2]] == ["collage", "receipt"]
    assert [w["verdict"] for w in st.watches] == ["revise", "keep"]
    assert st.revision["changes"][0]["field"] == "look" and st.revision["state"] == "closed"
    loops = [q for q in fb.requests if q["purpose"] == "loop"]
    assert len(loops) == 7 and all(q["tools"] == [t["name"] for t in TOOL_SPECS] for q in loops)
    assert loops[1]["previous_id"] == "resp_FIXTURE_1"                  # tool results chained to the last turn
    assert len(r.calls) == 2


def test_turn_cap_and_a_silent_model_both_stop_the_loop(tmp_path):
    spin = {"model_turns": [{"calls": [{"name": "finish_not", "arguments": {}}]}] * 3}
    tools, st, _, fb = make(tmp_path, {"loop": [spin]})
    assert loop.run_model(fb, tools, st, max_turns=2) == "turn cap"
    assert len([q for q in fb.requests if q["purpose"] == "loop"]) == 2
    tools, st, _, fb = make(tmp_path, {"loop": [{"text": "I think we are done."}]})
    assert loop.run_model(fb, tools, st) == "model stopped" and "without calling finish" in st.finished


def test_code_driver_does_the_same_job(tmp_path):
    tools, st, r, fb = make(tmp_path, "dry_run_script.json")
    assert loop.run_code(fb, tools, st) == "finished"
    assert [x["look"] for x in st.renders[2]] == ["collage", "receipt"]
    assert [q["purpose"] for q in fb.requests] == ["watch", "decide", "watch"]
