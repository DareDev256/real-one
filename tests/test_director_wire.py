"""What MetaBackend actually puts on the wire, through the real OpenAI SDK, against a mock api.meta.ai on
127.0.0.1. No network, no key, no spend: this pins our request shapes to the documented ones
(docs/meta-api-director.md) before the first keyed run, and checks usage parsing, metering and key scrubbing."""
import http.server
import json
import threading

import pytest

import realone.meta_api as meta_api
from realone.director import budget
from realone.director.backend import AudioRejected, MetaBackend, text_part, user
from realone.director.schema import TREATMENT_SCHEMA
from realone.director.tools import TOOL_SPECS

KEY = "sk-test-" + "W4" * 16


def response(output, usage=None):
    return {"id": "resp_mock_1", "object": "response", "created_at": 1, "model": "muse-wonder-1.0",
            "status": "completed", "output": output, "parallel_tool_calls": True, "tool_choice": "auto",
            "tools": [], "error": None, "incomplete_details": None, "instructions": None, "metadata": {},
            "temperature": 1.0, "top_p": 1.0,
            "usage": usage or {"input_tokens": 9000, "input_tokens_details": {"cached_tokens": 1000},
                               "output_tokens": 5000, "output_tokens_details": {"reasoning_tokens": 4000},
                               "total_tokens": 14000}}


class Mock(http.server.BaseHTTPRequestHandler):
    captured, queue = [], []

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        Mock.captured.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                              "body": body})
        kind, payload = Mock.queue.pop(0)
        if kind == "error":
            return self._send(400, {"error": {"message": payload + " (auth was " +
                                              self.headers.get("Authorization", "") + ")",
                                              "type": "invalid_request_error"}})
        if self.path.startswith("/v1/responses"):
            created = {"type": "response.created", "sequence_number": 0,
                       "response": {**payload, "status": "in_progress", "output": []}}
            done = {"type": "response.completed", "sequence_number": 1, "response": payload}
            sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in (created, done))
            return self._send(200, sse.encode(), "text/event-stream")
        return self._send(200, payload)


@pytest.fixture
def api(monkeypatch):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(meta_api, "BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    Mock.captured, Mock.queue = [], []
    meter = budget.Meter(cap_usd=2.0, max_calls=10)
    yield MetaBackend(KEY, meter), meter
    srv.shutdown()


def test_treatment_request_shape_usage_and_metering(api):
    be, meter = api
    tr = {"song_subtext": "x" * 20, "sections": []}
    Mock.queue = [("ok", response([{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                                    "content": [{"type": "output_text", "text": json.dumps(tr),
                                                 "annotations": []}]}]))]
    turn = be.respond(purpose="treatment", instructions="direct it",
                      input=[user(text_part("timeline"), {"type": "input_audio", "file_id": "file-7"})],
                      schema=TREATMENT_SCHEMA, schema_name="real_one_treatment", effort="high", est_input_tokens=9000)
    req = Mock.captured[0]
    body = json.loads(req["body"])
    assert req["path"] == "/v1/responses" and req["headers"]["authorization"] == f"Bearer {KEY}"
    assert body["model"] == "muse-wonder-1.0" and body["stream"] is True and body["store"] is True
    assert body["reasoning"] == {"effort": "high"} and body["max_output_tokens"] == budget.MAX_OUT["treatment"]
    fmt = body["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["name"] == "real_one_treatment" and fmt["strict"] is True
    assert "minLength" not in json.dumps(fmt["schema"]) and fmt["schema"]["additionalProperties"] is False
    assert body["input"][0]["content"][1] == {"type": "input_audio", "file_id": "file-7"}
    assert json.loads(turn.text) == tr and turn.calls == []
    assert turn.usage == {"input_tokens": 9000, "cached_tokens": 1000, "output_tokens": 5000, "reasoning_tokens": 4000}
    assert turn.usd == pytest.approx(budget.wonder_usd(9000, 5000, 1000)) == pytest.approx(meter.spent)
    assert meter.calls == 1


def test_tool_turns_send_flat_function_tools_and_parse_calls(api):
    be, _ = api
    call = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "render_section",
            "arguments": json.dumps({"section": 2, "look": "collage", "freeze": False}), "status": "completed"}
    Mock.queue = [("ok", response([call]))]
    turn = be.respond(purpose="loop", instructions="run it",
                      input=[{"type": "function_call_output", "call_id": "call_0", "output": "{}"}],
                      tools=TOOL_SPECS, previous_id="resp_prev", effort="medium", est_input_tokens=5000)
    body = json.loads(Mock.captured[0]["body"])
    assert body["previous_response_id"] == "resp_prev" and body["parallel_tool_calls"] is True
    assert "tool_choice" not in body                                    # the default "auto", the only one allowed
    assert [t["name"] for t in body["tools"]] == [t["name"] for t in TOOL_SPECS]
    assert all(t["type"] == "function" and "parameters" in t and "function" not in t for t in body["tools"])
    assert turn.calls == [{"call_id": "call_1", "name": "render_section", "arguments": call["arguments"]}]


def test_upload_sends_user_data_with_an_expiry(api, tmp_path):
    be, meter = api
    f = tmp_path / "render.mp4"
    f.write_bytes(b"\x00" * 64)
    Mock.queue = [("ok", {"id": "file-123", "object": "file", "bytes": 64, "created_at": 1, "filename": "render.mp4",
                          "purpose": "user_data", "status": "processed", "expires_at": 86401})]
    assert be.upload(f, "render") == "file-123"
    body = Mock.captured[0]["body"]
    assert Mock.captured[0]["path"] == "/v1/files"
    assert b'name="purpose"\r\n\r\nuser_data' in body and b"86400" in body and b"created_at" in body
    assert meter.calls == 1 and meter.spent == 0.0


def test_transcribe_is_multipart_with_the_documented_request_part(api, tmp_path):
    be, meter = api
    wav = tmp_path / "asr_00.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40 + b"\x00\x01" * 48000 * 3)   # ~6 s of 24 kHz mono s16 after the header
    Mock.queue = [("ok", {"sessionId": "s1", "transcript": "hi", "audioDurationMs": 6400,
                          "turns": [{"turnId": 1, "startMs": 100, "endMs": 900, "transcript": "hi"}]})]
    out = be.transcribe(wav, keywords=["mama", "overtime"])
    req = Mock.captured[0]
    assert req["path"].startswith("/v1/asr/transcribe?sessionId=realone-")
    assert req["headers"]["authorization"] == f"Bearer {KEY}" and req["headers"]["accept"] == "application/json"
    assert req["headers"]["content-type"].startswith("multipart/form-data; boundary=")
    part = json.loads(req["body"].split(b"Content-Type: application/json\r\n\r\n")[1].split(b"\r\n")[0])
    assert part == {"model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV", "mode": "ENDPOINTING",
                    "keywords": ["mama", "overtime"]}
    assert b'name="audio"; filename="asr_00.wav"' in req["body"]
    assert out["turns"][0]["transcript"] == "hi" and out["_source"] == "muse-voice-transcribe-1.0"
    assert meter.spent == pytest.approx(6 * budget.VOICE_USD_PER_S)       # whole seconds, rounded down


def test_an_audio_rejection_is_typed_and_the_key_is_scrubbed(api):
    be, meter = api
    Mock.queue = [("error", "input_audio is not supported for this model")]
    with pytest.raises(AudioRejected) as e:
        be.respond(purpose="treatment", instructions="x", input=[user(text_part("y"))], schema=TREATMENT_SCHEMA,
                   schema_name="t", est_input_tokens=100)
    assert KEY not in str(e.value) and "***" in str(e.value)
    assert meter.spent == 0.0
