"""Model backends behind one small interface, so the director never talks to an SDK directly.

    respond(purpose, instructions, input, schema=None, tools=None, previous_id=None, effort, ...) -> Turn
    upload(path, purpose) -> file id
    transcribe(wav_path, keywords) -> TranscribeResponse dict
    image(prompt) -> (path or None, note)

- MetaBackend: Meta Model API. Muse Wonder on the Responses API (streamed, so a long reasoning pass cannot hit the
  non-streaming time limit, docs/protocols/responses), Files API uploads that expire after a day, Voice Transcribe
  file mode over plain HTTPS (stdlib, no new dependency), Muse Image through realone/worlds.py's cached call.
  Every paid call goes through the Meter first. The key is never printed: error text is scrubbed.
- FixtureBackend: hand-written responses from tests/fixtures/director/, for --dry-run and the tests. Every Turn it
  returns carries source "FIXTURE:<file>"; it is never presented as model output.
- Later, not built: a chat-completions backend for Muse Spark Open on Baseten (`meta-models/muse-spark-open-1026`)
  implements the same four methods; `respond` maps `schema` to `response_format` and `tools` to the nested
  chat-completions tool format (docs/tool-calling).
"""
import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from realone.common import REPO, Refuse
from realone.director.budget import (MAX_OUT, VOICE_MODEL, VOICE_USD_PER_S, WONDER_MODEL, wonder_usd)
from realone.director.schema import wire_schema

FIXTURE_DIR = REPO / "tests" / "fixtures" / "director"
FIXTURE_LABEL = "FIXTURE (hand-written test response from tests/fixtures/director/, not model output)"


@dataclass
class Turn:
    id: str
    text: str
    calls: list = field(default_factory=list)   # [{"call_id", "name", "arguments" (JSON string)}]
    usage: dict = field(default_factory=dict)
    usd: float = 0.0
    source: str = WONDER_MODEL
    status: str = "completed"


class AudioRejected(Exception):
    """The API refused the song as input_audio (UNCONFIRMED for Wonder). The caller decides what to do."""


def text_part(s):
    return {"type": "input_text", "text": s}


def user(*parts):
    return {"type": "message", "role": "user", "content": list(parts)}


def _scrub(msg, key):
    msg = str(msg)
    return msg.replace(key, "***") if key else msg


# --------------------------------------------------------------------------------------------- Meta Model API

class MetaBackend:
    label = WONDER_MODEL
    fixture = False

    def __init__(self, key, meter):
        from realone.meta_api import make_client
        self._key = key
        self.meter = meter
        self.client = make_client(key)

    def respond(self, *, purpose, instructions, input, schema=None, schema_name=None, tools=None, previous_id=None,
                effort="medium", max_output_tokens=None, est_input_tokens=0):
        import openai
        kind = purpose if purpose in MAX_OUT else "loop"
        max_out = max_output_tokens or MAX_OUT[kind]
        self.meter.before(f"Muse Wonder ({purpose})", wonder_usd(est_input_tokens, max_out))
        kw = {"model": WONDER_MODEL, "input": input, "instructions": instructions,
              "reasoning": {"effort": effort}, "max_output_tokens": max_out, "store": True}
        if schema is not None:
            kw["text"] = {"format": {"type": "json_schema", "name": schema_name or purpose,
                                     "schema": wire_schema(schema), "strict": True}}
        if tools:
            kw["tools"] = tools
            kw["parallel_tool_calls"] = True
        if previous_id:
            kw["previous_response_id"] = previous_id
        try:
            with self.client.responses.stream(**kw) as st:
                resp = st.get_final_response()
        except openai.BadRequestError as e:
            msg = _scrub(e, self._key)
            if "audio" in msg.lower():
                raise AudioRejected(msg[:400]) from None
            raise Refuse(f"Muse Wonder rejected the {purpose} request (HTTP 400): {msg[:400]}") from None
        except openai.APIError as e:
            raise Refuse(f"Muse Wonder {purpose} failed: {type(e).__name__}: {_scrub(e, self._key)[:400]}") from None
        u = resp.usage
        usage = {"input_tokens": getattr(u, "input_tokens", 0) or 0,
                 "cached_tokens": getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0,
                 "output_tokens": getattr(u, "output_tokens", 0) or 0,
                 "reasoning_tokens": getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0}
        usd = wonder_usd(usage["input_tokens"], usage["output_tokens"], usage["cached_tokens"])
        self.meter.charge(f"wonder:{purpose}", usd, usage)
        calls = [{"call_id": it.call_id, "name": it.name, "arguments": it.arguments}
                 for it in resp.output if getattr(it, "type", "") == "function_call"]
        return Turn(id=resp.id, text=resp.output_text or "", calls=calls, usage=usage, usd=usd,
                    source=WONDER_MODEL, status=resp.status or "completed")

    def upload(self, path, purpose):
        self.meter.before(f"Files API upload ({purpose})", 0.0)
        try:
            with open(path, "rb") as f:
                fo = self.client.files.create(file=f, purpose="user_data",
                                              expires_after={"anchor": "created_at", "seconds": 86400})
        except Exception as e:  # noqa: BLE001 (any SDK or network error ends the run the same way)
            raise Refuse(f"Files API upload of {Path(path).name} failed: {_scrub(e, self._key)[:300]}") from None
        self.meter.charge("files:upload", 0.0, {"file": Path(path).name, "id": fo.id, "bytes": fo.bytes})
        return fo.id

    def transcribe(self, wav_path, keywords=None):
        """POST /v1/asr/transcribe, multipart (`request` JSON + `audio` WAV), mode ENDPOINTING (turns)."""
        import urllib.error
        import urllib.request
        from realone.meta_api import BASE_URL
        wav = Path(wav_path).read_bytes()
        secs = max(0.0, (len(wav) - 44) / 48000.0)  # 24 kHz mono s16
        self.meter.before("Voice Transcribe", math.floor(secs) * VOICE_USD_PER_S)
        req = {"model": VOICE_MODEL, "audioEncoding": "WAV", "mode": "ENDPOINTING"}
        if keywords:
            req["keywords"] = list(keywords)[:40]
        boundary = "realone" + uuid.uuid4().hex
        body = b"".join([
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\n"
            f"Content-Type: application/json\r\n\r\n".encode(), json.dumps(req).encode(), b"\r\n",
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"{Path(wav_path).name}\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n".encode(), wav, f"\r\n--{boundary}--\r\n".encode()])
        url = f"{BASE_URL}/asr/transcribe?sessionId=realone-{uuid.uuid4().hex[:12]}"
        headers = {"Authorization": f"Bearer {self._key}", "Accept": "application/json",
                   "Content-Type": f"multipart/form-data; boundary={boundary}"}
        data = None
        for attempt in (1, 2):  # docs: a 500 means the server budget ran out; retry once
            try:
                with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=headers, method="POST"),
                                            timeout=900) as r:
                    data = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                detail = _scrub(e.read()[:300].decode(errors="replace"), self._key)
                if e.code == 500 and attempt == 1:
                    continue
                raise Refuse(f"Voice Transcribe HTTP {e.code}: {detail}") from None
            except urllib.error.URLError as e:
                raise Refuse(f"Voice Transcribe unreachable: {_scrub(e.reason, self._key)}") from None
        billed = math.floor(data.get("audioDurationMs", 0) / 1000) * VOICE_USD_PER_S
        self.meter.charge("voice:transcribe", billed, {"audio_ms": data.get("audioDurationMs"),
                                                      "session": data.get("sessionId")})
        data["_source"] = VOICE_MODEL
        return data

    def image(self, prompt):
        from realone.meta_api import IMAGE_USD
        from realone.worlds import gen_world, muse_world_cached
        if not muse_world_cached(prompt):
            self.meter.before("Muse Image", IMAGE_USD)
        path, usd = gen_world(self.client, prompt)
        if usd:
            self.meter.charge("image:generate", usd, {"world": Path(path).name})
        return path, ("cache hit, no call" if not usd else "generated")


# --------------------------------------------------------------------------------------------- fixtures

class FixtureBackend:
    """Replays hand-written responses, queued per purpose. Runs out -> refuses (it never invents one)."""
    label = "FIXTURE"
    fixture = True

    def __init__(self, script, base=FIXTURE_DIR):
        self.base = Path(base)
        if isinstance(script, (str, Path)):
            script = json.loads((self.base / script).read_text())
        self.queues = {}
        for purpose, entries in script.items():
            if purpose.startswith("_"):
                continue
            q = []
            for e in entries:
                data, src = (json.loads((self.base / e).read_text()), f"FIXTURE:{e}") if isinstance(e, str) \
                    else (e, "FIXTURE:inline")
                if "model_turns" in data:  # a scripted multi-turn file expands into its turns
                    q += [(t, f"{src}#turn{k + 1}") for k, t in enumerate(data["model_turns"])]
                else:
                    q.append((data, src))
            self.queues[purpose] = q
        self.requests = []
        self.uploads = []
        self.n = 0

    def _next(self, purpose):
        q = self.queues.get(purpose)
        if not q:
            raise Refuse(f"the FIXTURE script has no response left for '{purpose}'; a dry run stops here rather "
                         f"than invent one")
        return q.pop(0)

    def respond(self, *, purpose, instructions, input, schema=None, schema_name=None, tools=None, previous_id=None,
                effort="medium", max_output_tokens=None, est_input_tokens=0):
        self.requests.append({"purpose": purpose, "instructions": instructions, "input": input,
                              "schema_name": schema_name, "tools": [t["name"] for t in tools or []],
                              "previous_id": previous_id, "effort": effort})
        data, src = self._next(purpose)
        self.n += 1
        if "text" in data:
            text = data["text"]
        elif "response" in data:
            text = data["response"] if isinstance(data["response"], str) else json.dumps(data["response"])
        else:
            text = ""
        calls = [{"call_id": f"fx_{self.n}_{k}", "name": c["name"],
                  "arguments": c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"])}
                 for k, c in enumerate(data.get("calls", []))]
        return Turn(id=f"resp_FIXTURE_{self.n}", text=text, calls=calls,
                    usage={"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
                    usd=0.0, source=src)

    def upload(self, path, purpose):
        self.uploads.append(str(path))
        return f"file-FIXTURE-{len(self.uploads)}"

    def transcribe(self, wav_path, keywords=None):
        self.requests.append({"purpose": "transcribe", "keywords": list(keywords or [])})
        data, src = self._next("transcribe")
        out = {k: v for k, v in data.items() if not k.startswith("_")}
        out["_source"] = src
        return out

    def image(self, prompt):
        return None, "FIXTURE: Muse Image not called (dry run); the test render uses --local-plate plates"
