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
- SparkOpenBackend: Muse Spark Open on Baseten (OpenAI-compatible chat completions), for `plan --brain spark-open`.
  `respond` maps `instructions` to a system message, `schema` to `response_format`, `tools` to the nested
  chat-completions format and `function_call_output` to tool messages; chat completions are stateless, so
  `previous_id` replays the stored history. There is no Files API: `upload` keeps the path and `respond` inlines the
  song as base64 `input_audio` (the part shape is OpenAI's; UNCONFIRMED for Baseten until the first keyed call).
"""
import base64
import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from realone.common import REPO, Refuse
from realone.director.budget import (MAX_OUT, SPARK_BASE_URL, SPARK_MODEL, VOICE_MODEL, VOICE_USD_PER_S,
                                     WONDER_MODEL, spark_usd, wonder_usd)
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


# --------------------------------------------------------------------------------------------- Spark Open (Baseten)

SPARK_AUDIO_MAX_BYTES = 20 << 20  # inlined as base64 in one request; a 3 min mono 96k MP3 is about 2 MB


class SparkOpenBackend:
    fixture = False

    name, kind, host, base_url, audio = "Spark Open", "spark", "Baseten", SPARK_BASE_URL, True

    def _usd(self, *a):
        return spark_usd(*a)

    def __init__(self, key, meter, model=SPARK_MODEL, client=None, base_url=None):
        self._key = key
        self.meter = meter
        self.model = model
        self.label = f"{model} ({self.host})"
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=base_url or self.base_url, api_key=key, timeout=1800.0)
        self.client = client
        self._files = {}
        self._history = {}

    def _part(self, p):
        t = p.get("type")
        if t == "input_text":
            return {"type": "text", "text": p["text"]}
        if t == "input_audio":
            if not self.audio:
                raise Refuse(f"{self.name} takes no audio input; run with --no-audio (it reads the lyrics and the beat "
                             f"grid). Nothing was sent.")
            path = self._files.get(p.get("file_id"))
            if path is None:
                raise Refuse(f"input_audio names {p.get('file_id')!r}, which this Spark Open run never uploaded")
            data = base64.b64encode(Path(path).read_bytes()).decode()
            return {"type": "input_audio", "input_audio": {"data": data, "format": Path(path).suffix.lstrip(".")}}
        raise Refuse(f"Spark Open drafts treatments only; a {t!r} input part is not wired (watching a render stays "
                     f"on Muse Wonder)")

    def _messages(self, items):
        out = []
        for it in items:
            if it.get("type") == "function_call_output":
                out.append({"role": "tool", "tool_call_id": it["call_id"], "content": it["output"]})
            else:
                out.append({"role": it.get("role", "user"), "content": [self._part(p) for p in it["content"]]})
        return out

    def respond(self, *, purpose, instructions, input, schema=None, schema_name=None, tools=None, previous_id=None,
                effort="medium", max_output_tokens=None, est_input_tokens=0):
        import openai
        kind = purpose if purpose in MAX_OUT else "loop"
        max_out = max_output_tokens or MAX_OUT[kind]
        if previous_id is not None and previous_id not in self._history:
            raise Refuse(f"previous_id {previous_id!r} is not a turn of this Spark Open run; nothing was called")
        messages = list(self._history[previous_id]) if previous_id else [{"role": "system", "content": instructions}]
        messages += self._messages(input)
        self.meter.before(f"Muse {self.name} ({purpose})", self._usd(est_input_tokens, max_out))
        kw = {"model": self.model, "messages": messages, "max_tokens": max_out}
        if schema is not None:
            kw["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema_name or purpose, "schema": wire_schema(schema), "strict": True}}
        if tools:
            kw["tools"] = [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters",
                                                                                "strict") if k in t}}
                           for t in tools]
        try:
            resp = self.client.chat.completions.create(**kw)
        except openai.BadRequestError as e:
            msg = _scrub(e, self._key)
            if "audio" in msg.lower():
                raise AudioRejected(msg[:400]) from None
            raise Refuse(f"Spark Open rejected the {purpose} request (HTTP 400): {msg[:400]}") from None
        except openai.APIError as e:
            raise Refuse(f"Spark Open {purpose} failed: {type(e).__name__}: {_scrub(e, self._key)[:400]}") from None
        u = resp.usage
        usage = {"input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                 "cached_tokens": getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
                 "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                 "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0)
                 or 0}
        usd = self._usd(usage["input_tokens"], usage["output_tokens"], usage["cached_tokens"])
        self.meter.charge(f"{self.kind}:{purpose}", usd, usage)
        choice = resp.choices[0]
        msg = choice.message
        calls = [{"call_id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                 for tc in (getattr(msg, "tool_calls", None) or [])]
        assistant = {"role": "assistant", "content": msg.content or ""}
        if calls:
            assistant["tool_calls"] = [{"id": c["call_id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
        tid = resp.id or f"spark_{uuid.uuid4().hex[:12]}"
        self._history[tid] = messages + [assistant]
        return Turn(id=tid, text=msg.content or "", calls=calls, usage=usage, usd=usd, source=self.label,
                    status="incomplete" if choice.finish_reason == "length" else "completed")

    def upload(self, path, purpose):
        """No Files API on Baseten: keep the path (nothing leaves the Mac here) and inline it in `respond`."""
        size = Path(path).stat().st_size
        if size > SPARK_AUDIO_MAX_BYTES:
            raise Refuse(f"{Path(path).name} is {size} bytes; Spark Open gets audio inline and this run caps it at "
                         f"{SPARK_AUDIO_MAX_BYTES} bytes. Nothing was sent.")
        fid = f"local-{len(self._files) + 1}"
        self._files[fid] = str(path)
        return fid

    def transcribe(self, wav_path, keywords=None):
        raise Refuse("Spark Open has no transcription endpoint. Run plan --brain spark-open with --no-transcribe "
                     "(and --lyrics for the words), or transcribe on Muse Voice Transcribe first.")

    def image(self, prompt):
        raise Refuse("Spark Open drafts treatments only; worlds are built by Muse Image on the Meta Model API")


class GlimmerBackend(SparkOpenBackend):
    """Muse Glimmer 30B, open weights (Apache-2.0), on a llama.cpp server we host (scripts/glimmer_pod.py). Same
    chat-completions shape as Spark Open, so the treatment goes through the same schema, guard and validation.
    No audio input (Glimmer reads images and text), no per-token bill (the GPU is billed by the hour)."""
    name, kind, host, audio = "Glimmer", "glimmer", "self-hosted, llama.cpp", False

    def _usd(self, *a):
        from realone.director.budget import glimmer_usd
        return glimmer_usd(*a)

    def __init__(self, meter, base_url, model=None, client=None):
        from realone.director.budget import GLIMMER_MODEL
        import os  # the one-run key scripts/glimmer_pod.py gave the server (its proxy URL is public)
        super().__init__(os.environ.get("GLIMMER_API_KEY") or "none", meter, model or GLIMMER_MODEL, client=client,
                         base_url=base_url)


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
