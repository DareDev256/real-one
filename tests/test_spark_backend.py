"""Muse Spark Open on Baseten, keyless (fake chat-completions client): the request maps onto chat completions
(system message, json_schema response_format, nested tools, tool messages), the song goes inline as base64 audio with
no upload, previous_id replays history, the meter refuses before any call, the key never reaches error text, and
`plan --brain spark-open` refuses without its key or with transcription on. Every response here is a FIXTURE."""
import base64
import json
import os
import subprocess
import sys
import types

import openai
import pytest

from conftest import REPO
from realone.common import Refuse
from realone.director import budget, treatment as trt
from realone.director.backend import AudioRejected, SparkOpenBackend, text_part, user
from test_director_treatment import clean_treatment, lyrics, timeline

KEY = "bt-test-" + "Q7" * 16


def reply(content="", tool_calls=None, finish="stop", rid="chatcmpl-1", prompt=1000, completion=500, cached=0):
    msg = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = types.SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                                  prompt_tokens_details=types.SimpleNamespace(cached_tokens=cached),
                                  completion_tokens_details=None)
    return types.SimpleNamespace(id=rid, choices=[types.SimpleNamespace(message=msg, finish_reason=finish)],
                                 usage=usage)


class FakeClient:
    def __init__(self, replies=(), error=None):
        self.replies, self.error, self.calls = list(replies), error, []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(json.loads(json.dumps(kw)))
        if self.error:
            raise self.error
        return self.replies.pop(0)


def backend(client, cap=2.0):
    return SparkOpenBackend(KEY, budget.Meter(cap, 10), client=client)


def bad_request(text):
    req = types.SimpleNamespace(method="POST", url="https://inference.baseten.co/v1/chat/completions", headers={})
    resp = types.SimpleNamespace(status_code=400, request=req, headers={})
    return openai.BadRequestError(text, response=resp, body=None)


def test_schema_and_instructions_map_onto_chat_completions():
    fc = FakeClient([reply(json.dumps({"ok": True}))])
    b = backend(fc)
    turn = b.respond(purpose="treatment", instructions="BRIEF", input=[user(text_part("hello"))],
                     schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
                     schema_name="real_one_treatment", est_input_tokens=1000)
    kw = fc.calls[0]
    assert kw["model"] == budget.SPARK_MODEL and kw["max_tokens"] == budget.MAX_OUT["treatment"]
    assert kw["messages"][0] == {"role": "system", "content": "BRIEF"}
    assert kw["messages"][1] == {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    rf = kw["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["name"] == "real_one_treatment"
    assert rf["json_schema"]["strict"] is True
    assert turn.text == '{"ok": true}' and turn.source == f"{budget.SPARK_MODEL} (Baseten)"
    assert turn.usd == pytest.approx(budget.spark_usd(1000, 500))


def test_song_goes_inline_as_base64_audio_and_upload_sends_nothing(tmp_path):
    song = tmp_path / "song.mp3"
    song.write_bytes(b"ID3fake-mp3-bytes")
    fc = FakeClient([reply("{}")])
    b = backend(fc)
    fid = b.upload(song, "song audio")
    assert fc.calls == []  # no network on upload
    b.respond(purpose="treatment", instructions="I", input=[user(text_part("t"), {"type": "input_audio",
                                                                                  "file_id": fid})])
    part = fc.calls[0]["messages"][1]["content"][1]
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "mp3"
    assert base64.b64decode(part["input_audio"]["data"]) == b"ID3fake-mp3-bytes"


def test_upload_over_the_inline_cap_refuses(tmp_path, monkeypatch):
    import realone.director.backend as be
    monkeypatch.setattr(be, "SPARK_AUDIO_MAX_BYTES", 8)
    big = tmp_path / "big.mp3"
    big.write_bytes(b"x" * 9)
    with pytest.raises(Refuse, match="Nothing was sent"):
        backend(FakeClient()).upload(big, "song audio")


def test_an_audio_part_that_was_never_uploaded_refuses_before_calling():
    fc = FakeClient([reply("{}")])
    with pytest.raises(Refuse, match="never uploaded"):
        backend(fc).respond(purpose="treatment", instructions="I",
                            input=[user({"type": "input_audio", "file_id": "local-9"})])
    assert fc.calls == []


def test_previous_id_replays_the_history_chat_completions_does_not_keep():
    fc = FakeClient([reply("first", rid="r1"), reply("second", rid="r2")])
    b = backend(fc)
    t1 = b.respond(purpose="treatment", instructions="SYS", input=[user(text_part("draft it"))])
    b.respond(purpose="treatment_repair", instructions="SYS", input=[user(text_part("fix it"))], previous_id=t1.id)
    msgs = fc.calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "first" and msgs[3]["content"][0]["text"] == "fix it"
    assert sum(m["role"] == "system" for m in msgs) == 1


def test_an_unknown_previous_id_refuses_before_calling():
    fc = FakeClient([reply("x")])
    with pytest.raises(Refuse, match="not a turn of this Spark Open run"):
        backend(fc).respond(purpose="guard_reask", instructions="I", input=[user(text_part("x"))],
                            previous_id="resp_from_wonder")
    assert fc.calls == []


def test_tools_nest_and_tool_calls_round_trip():
    tc = types.SimpleNamespace(id="call_1", function=types.SimpleNamespace(name="finish", arguments='{"a": 1}'))
    fc = FakeClient([reply("", tool_calls=[tc], rid="r1"), reply("done", rid="r2")])
    b = backend(fc)
    spec = {"type": "function", "name": "finish", "description": "end", "strict": True,
            "parameters": {"type": "object", "properties": {}, "required": []}}
    t1 = b.respond(purpose="loop", instructions="I", input=[user(text_part("go"))], tools=[spec])
    assert fc.calls[0]["tools"] == [{"type": "function", "function": {"name": "finish", "description": "end",
                                                                       "parameters": spec["parameters"],
                                                                       "strict": True}}]
    assert t1.calls == [{"call_id": "call_1", "name": "finish", "arguments": '{"a": 1}'}]
    b.respond(purpose="loop", instructions="I", previous_id=t1.id,
              input=[{"type": "function_call_output", "call_id": "call_1", "output": '{"ok": true}'}])
    msgs = fc.calls[1]["messages"]
    assert msgs[2]["tool_calls"][0]["id"] == "call_1"
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}


def test_the_meter_refuses_before_any_call_over_the_cap():
    fc = FakeClient([reply("x")])
    with pytest.raises(Refuse, match="Stopped before calling"):
        backend(fc, cap=0.0001).respond(purpose="treatment", instructions="I", input=[user(text_part("x"))],
                                        est_input_tokens=50000)
    assert fc.calls == []


def test_errors_never_carry_the_key_and_audio_refusal_is_its_own_signal():
    b = backend(FakeClient(error=bad_request(f"bad key {KEY} rejected")))
    with pytest.raises(Refuse) as e:
        b.respond(purpose="treatment", instructions="I", input=[user(text_part("x"))])
    assert KEY not in str(e.value) and "***" in str(e.value)
    b = backend(FakeClient(error=bad_request("input_audio is not supported for this model")))
    with pytest.raises(AudioRejected):
        b.respond(purpose="treatment", instructions="I", input=[user(text_part("x"))])


def test_video_transcribe_and_image_are_not_wired_and_say_so():
    b = backend(FakeClient([reply("x")]))
    with pytest.raises(Refuse, match="stays on Muse Wonder"):
        b.respond(purpose="watch", instructions="I", input=[user({"type": "input_video", "file_id": "f"})])
    with pytest.raises(Refuse, match="--no-transcribe"):
        b.transcribe("song.wav")
    with pytest.raises(Refuse, match="Muse Image"):
        b.image("a world")


def test_a_full_treatment_runs_through_spark_open():
    fc = FakeClient([reply(json.dumps(clean_treatment()), rid="r1", prompt=4000, completion=3000)])
    b = backend(fc)
    res = trt.make(b, timeline(), lyrics())
    assert res["treatment"]["sections"] and res["attempts"][0]["source"] == f"{budget.SPARK_MODEL} (Baseten)"
    assert b.meter.calls == 1 and b.meter.spent == pytest.approx(budget.spark_usd(4000, 3000))


def test_the_quote_prices_spark_open_and_refuses_direct():
    kw = dict(song_s=180, n_sections=8, brief_chars=4000, audio_in=True, transcribe=True)
    w, s = budget.quote(**kw), budget.quote(**kw, brain="spark-open")
    assert [x["what"].split(" (")[0] for x in s["lines"]][1:] == ["Spark Open treatment", "Spark Open repair turn",
                                                                  "Spark Open guard re-ask"]
    assert 0 < s["ceiling_usd"] < w["ceiling_usd"]
    with pytest.raises(Refuse, match="plan"):
        budget.quote(**kw, brain="spark-open", direct=True, covered_s=[20.0])


def run(args, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k not in ("MODEL_API_KEY", "BASETEN_API_KEY")}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "realone.director"] + args, cwd=REPO, env=env,
                          capture_output=True, text=True)


@pytest.fixture
def junk_song(tmp_path):
    s = tmp_path / "song.wav"
    s.write_bytes(b"not audio")
    return str(s)


def test_plan_spark_open_refuses_without_its_key_before_any_work(junk_song):
    p = run(["plan", "--brain", "spark-open", "--no-transcribe", "--song", junk_song],
            {"MODEL_API_KEY": "sk-meta-not-enough"})
    assert p.returncode == 2, p.stderr
    assert "BASETEN_API_KEY is not set" in p.stderr and "Nothing" in p.stderr


def test_plan_spark_open_refuses_with_transcription_on(junk_song):
    p = run(["plan", "--brain", "spark-open", "--song", junk_song], {"BASETEN_API_KEY": KEY})
    assert p.returncode == 2, p.stderr
    assert "--no-transcribe" in p.stderr and KEY not in p.stderr + p.stdout
