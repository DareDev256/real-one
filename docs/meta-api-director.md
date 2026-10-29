# Meta Model API: what the director can rely on

Read 2026-10-29 from the public docs at `https://dev.meta.ai/docs/` with `curl` (no login, no browser), stripped to
text with a small HTML-to-text script. Every item is paraphrased. **CONFIRMED** means the docs state it; the page is
cited as `/docs/<slug>`. **UNCONFIRMED** means the docs are silent or only show it for a different model; the first
keyed run must check it (the list at the end is the checklist).

Pages read: `models`, `quickstart`, `protocols`, `protocols/responses`, `speech-to-text`,
`api-reference/responses/create-response`, `api-reference/files/upload-file`, `image-generation`,
`pricing-rate-limits`, `muse-spark-open`; and, because they settle the exact shapes, `tool-calling`,
`structured-output`, `video-understanding` (titled "Video and audio understanding"), `file-handling`, `reasoning`,
`token-counting`, `image-understanding`, `api-reference/responses/schemas`, `api-reference/files/schemas`,
`api-reference/voice/transcribe`, `api-reference/voice/schemas`.

## Muse Wonder (`muse-wonder-1.0`)

### What it is and where it runs

| Item | Status | Source |
| --- | --- | --- |
| Model id `muse-wonder-1.0`, Meta's most capable line, in preview | CONFIRMED | `/docs/models` |
| Served on Responses (`POST /v1/responses`), Chat Completions and Messages | CONFIRMED | `/docs/models`, `/docs/protocols` |
| Wonder and Spark accept the same requests on the same endpoints, so switching is a model id change | CONFIRMED | `/docs/models` ("Choose Muse Wonder or Muse Spark") |
| Input: text, image, video, audio, PDF. Output: text. Context 1,048,576 tokens | CONFIRMED | `/docs/models` |
| Standard mode only on the API (no Contributor discount); "Wonder Full" is Muse Code only, and `effort: "max"` does not select it | CONFIRMED | `/docs/models`, `/docs/reasoning` |
| Not available in South Korea | CONFIRMED | `/docs/models` |
| Base URL `https://api.meta.ai/v1`, Bearer key from `MODEL_API_KEY`, OpenAI SDK compatible | CONFIRMED | `/docs/quickstart` |

### Request shape (Responses API, the one the director uses)

| Field | Meaning | Status | Source |
| --- | --- | --- | --- |
| `model`, `input` (a string or a list of typed items) | required core | CONFIRMED | `/docs/protocols/responses` |
| `instructions` | developer-level text for this request only; not carried by `previous_response_id`, so resend it each turn | CONFIRMED | `/docs/protocols/responses` |
| content parts `input_text`, `input_image`, `input_file`, `input_video`, `input_audio` | typed user content | CONFIRMED | `/docs/protocols/responses`, `/docs/api-reference/responses/schemas` |
| `previous_response_id` | server keeps history; send only new input | CONFIRMED | `/docs/protocols/responses` |
| `include: ["reasoning.encrypted_content"]` with `store: false` | stateless reasoning replay; cannot be combined with `previous_response_id` (HTTP 400) | CONFIRMED | `/docs/protocols/responses` |
| `reasoning.effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; `none` is rejected by Wonder; `max` is supported on Wonder | CONFIRMED | `/docs/reasoning`, `/docs/api-reference/responses/schemas` |
| default effort when omitted | the model "reasons by default"; the level is not named | UNCONFIRMED | `/docs/protocols/responses` |
| `max_output_tokens` | upper bound on generated tokens INCLUDING reasoning tokens; minimum 16 | CONFIRMED | `/docs/api-reference/responses/schemas` |
| `text.format` | structured output (see below) | CONFIRMED | `/docs/structured-output` |
| `tools`, `tool_choice`, `parallel_tool_calls` | function calling (see below) | CONFIRMED | `/docs/tool-calling` |
| `stream: true` | typed SSE events; long non-streamed calls can hit a server time limit and return HTTP 504 | CONFIRMED | `/docs/protocols/responses` |
| `truncation: "auto"` | rejected; the caller compacts history | CONFIRMED | `/docs/protocols/responses` |
| `temperature`, `top_p` | accepted, but the model is tuned for the defaults | CONFIRMED | `/docs/protocols/responses` |
| logprobs, `frequency_penalty`, `presence_penalty` | not supported | CONFIRMED | `/docs/protocols/responses` |

### Response shape

| Field | Status | Source |
| --- | --- | --- |
| `id`, `status`, `output[]`; message items carry `output_text` parts; `response.output_text` joins them in the SDK | CONFIRMED | `/docs/protocols/responses` (example response) |
| `function_call` output items with `name`, `arguments` (a JSON string), `call_id`, `status` | CONFIRMED | `/docs/tool-calling` |
| `reasoning` output items (opaque; raw chain of thought is never returned) | CONFIRMED | `/docs/protocols/responses` |
| assistant text before a tool call is tagged `phase: "commentary"` | CONFIRMED | `/docs/protocols/responses` |
| `usage.input_tokens`, `usage.input_tokens_details.cached_tokens`, `usage.output_tokens`, `usage.output_tokens_details.reasoning_tokens` | CONFIRMED | `/docs/protocols/responses` (retrieval section) |
| a failed stream ends with `response.failed` carrying an `error.code` | CONFIRMED | `/docs/protocols/responses` |

### Input modalities in detail

| Item | Status | Source |
| --- | --- | --- |
| **Video in**: upload the MP4 with the Files API (`purpose="user_data"`), then reference it with `input_file` + `file_id`, or `input_video` with `file_id` or `video_url` (public URL or base64 data URL) | CONFIRMED | `/docs/video-understanding`, `/docs/file-handling` |
| video reads the frames and the embedded audio together; a video without audio is valid | CONFIRMED | `/docs/video-understanding` |
| video format: MP4 only | CONFIRMED | `/docs/video-understanding` ("Supported formats") |
| `input_video.fps` sets the sampling rate; default is backend-chosen | CONFIRMED | `/docs/api-reference/responses/schemas` |
| **Audio in**: `input_audio` with inline `{data, format}` (`wav` or `mp3`), or `audio_url`, or `file_id` of an uploaded file | CONFIRMED | `/docs/api-reference/responses/schemas`, `/docs/video-understanding` |
| the audio and video examples are written for Muse Spark (1.2 for audio, because Spark 1.3 audio is "not fully supported"); Wonder lists audio and video input on the models page, and the docs say Wonder takes the same requests | CONFIRMED (shape) | `/docs/models`, `/docs/video-understanding` |
| how well **Wonder** hears music (structure, energy, mood) and how well it watches a 9:16 render | UNCONFIRMED | nothing on any page |
| tokens per second of audio or video | UNCONFIRMED | `/docs/token-counting` says media token counts vary by model and preprocessing and are not portable |

### Tool or function calling

| Item | Status | Source |
| --- | --- | --- |
| Wonder supports tool calling | CONFIRMED | `/docs/models` (Wonder paragraph) |
| function tools are flat in Responses: `{type: "function", name, description, parameters, strict?}`; `custom` tools take free text | CONFIRMED | `/docs/tool-calling` |
| results go back as `function_call_output` items with the matching `call_id` (1 to 64 chars), with `previous_response_id` | CONFIRMED | `/docs/tool-calling` |
| `tool_choice` must be `"auto"`; `"required"`, `"none"` and named choices return HTTP 400 | CONFIRMED | `/docs/tool-calling` |
| `parallel_tool_calls` defaults to true; set false for one call per turn | CONFIRMED | `/docs/tool-calling` |
| `max_tool_calls` caps built-in tools only; client tools must be capped by the caller's own loop | CONFIRMED | `/docs/tool-calling` |
| arguments are guided by the schema but NOT guaranteed to validate; parse and validate them | CONFIRMED | `/docs/tool-calling` |
| function names `^[a-zA-Z0-9_.-]+$`, at most one dot | CONFIRMED | `/docs/tool-calling` |
| every tool-calling example on the page uses `muse-spark-1.3`, not Wonder | CONFIRMED (fact about the docs) | `/docs/tool-calling` |
| Wonder's real behaviour driving a multi-step loop of our own tools | UNCONFIRMED | the agent guides point Wonder at long agentic work (`/docs/models`), but no Wonder tool-call example exists |

### Structured output (JSON)

| Item | Status | Source |
| --- | --- | --- |
| Responses: `text.format = {type: "json_schema", name, schema, strict?, description?}`; Chat Completions uses `response_format` instead, and each endpoint ignores the other's parameter | CONFIRMED | `/docs/structured-output`, `/docs/api-reference/responses/schemas` |
| decoding is constrained to the schema ("guaranteed to conform"), whether or not `strict` is set; `strict: true` additionally checks the schema against a subset (root object, `additionalProperties: false`, every property required, no `allOf`/`oneOf`) | CONFIRMED | `/docs/structured-output` |
| schema limits: depth 10, 5,000 properties, 1,000 enum values, no recursive schemas | CONFIRMED | `/docs/structured-output` |
| `json_object` mode also exists | CONFIRMED | `/docs/api-reference/responses/schemas` |
| constrained decoding holds for Wonder with reasoning on, on a 10-section schema | UNCONFIRMED | examples use Spark; the director validates locally anyway |

### Limits and prices

| Item | Status | Source |
| --- | --- | --- |
| US$4.00 per 1M input tokens, US$0.20 per 1M cached input, US$20.00 per 1M output; reasoning tokens bill as output | CONFIRMED | `/docs/pricing-rate-limits`, `/docs/reasoning` |
| no long-context premium; Meta's injected steering tokens are not billed | CONFIRMED | `/docs/pricing-rate-limits` |
| rate limits per team: 3,000 RPM, 4,000,000 TPM | CONFIRMED | `/docs/pricing-rate-limits` |
| inline files up to 50 MB; Files API up to 1 GiB per file, 100 GiB per team; uploads never expire unless `expires_after` (1 hour to 30 days) | CONFIRMED | `/docs/file-handling`, `/docs/api-reference/files/schemas` |
| `POST /v1/responses/input_tokens` counts input tokens without generating; the docs say use it for context fit, not billing | CONFIRMED | `/docs/token-counting` |
| whether `input_tokens` calls are free | UNCONFIRMED | not stated |

## Muse Voice Transcribe (`muse-voice-transcribe-1.0`)

| Item | Status | Source |
| --- | --- | --- |
| speech to text only; no TTS, no speech-to-speech | CONFIRMED | `/docs/models` |
| two endpoints, same model: realtime `wss://api.meta.ai/v1/asr/realtime`, file `POST /v1/asr/transcribe` | CONFIRMED | `/docs/speech-to-text` |
| **file request**: `multipart/form-data` with a `request` part (JSON) and an `audio` part (the WAV); `Authorization: Bearer` header; optional `?sessionId=` | CONFIRMED | `/docs/api-reference/voice/transcribe` |
| `request` JSON: `model` and `audioEncoding: "WAV"` required; `mode` (`PUSH_TO_TALK` default, `ENDPOINTING`, `DIARIZATION`); `keywords`; `languageBias`; `partialMode` and `emitAudioProgress` only affect SSE | CONFIRMED | `/docs/api-reference/voice/schemas` |
| audio: RIFF/WAVE, mono, 16-bit integer PCM at 16 kHz or 24 kHz (24 kHz is native) | CONFIRMED | `/docs/speech-to-text` |
| **file response** (JSON): `sessionId`, `transcript`, `audioDurationMs`, `turns[]` of `{turnId, startMs, endMs, transcript, speaker?}`; `turns` is empty in `PUSH_TO_TALK` | CONFIRMED | `/docs/api-reference/voice/schemas` |
| other response forms: `Accept: text/event-stream` (realtime events) or `text/plain` | CONFIRMED | `/docs/speech-to-text` |
| **timestamps are turn level only**, and a turn's `startMs`/`endMs` are "audio processed when the model detected" the edge, not a precise acoustic boundary; no word timestamps, no confidence | CONFIRMED | `/docs/speech-to-text`, `/docs/api-reference/voice/schemas` |
| caps: 10 minutes of audio (400), 32 MB body (413), 429 at the concurrency or hourly limit, 500 when processing passes the server budget (retry once) | CONFIRMED | `/docs/api-reference/voice/transcribe` |
| limits: 256 concurrent streams, 16,000 streams per hour, shared by both endpoints | CONFIRMED | `/docs/pricing-rate-limits` |
| price: US$0.18 per hour of audio, metered in whole seconds rounded down; failed and 429 requests are not billed | CONFIRMED | `/docs/pricing-rate-limits`, `/docs/speech-to-text` |
| 25 languages with code-switching; `keywords` bias recognition but do not guarantee spelling | CONFIRMED | `/docs/speech-to-text` |
| tool calling, JSON mode, reasoning effort | not applicable (not a text model) | `/docs/models` |
| quality on **sung** vocals over a full mix | UNCONFIRMED | the docs only describe speech products (agents, meetings, captions) |

## Muse Image (`muse-image-1.0`), as the director's `generate_world` tool uses it

US$0.01 per returned image; failed or filtered images are not billed; `size` sets the aspect ratio only, not the pixel
size; 150 RPM (CONFIRMED, `/docs/image-generation`, `/docs/pricing-rate-limits`). The director reuses
`realone/worlds.py::gen_world`, which already carries its own `# UNCONFIRMED` markers (`size="1080x1920"`).

## Muse Spark Open via Baseten (not built)

Spark Open is an 816B-total, 42B-active MoE with text, image, audio and video input, not served on Model API; Baseten
is a listed inference partner (CONFIRMED, `/docs/muse-spark-open`). The model id `meta-models/muse-spark-open-1026`
and its OpenAI-compatible chat completions endpoint come from the hackathon guide, not from dev.meta.ai
(UNCONFIRMED here). The director's backend interface (`realone/director/backend.py`) is shaped so a chat-completions
backend can be added without touching the loop.

## What the director does with this

- **Treatment**: one Wonder call on the Responses API with `text.format` json_schema (strict), the timeline and
  lyrics as text, and the song itself as `input_audio` (`file_id` of an upload, or inline mp3). Validated again
  locally, one repair turn on failure (`previous_response_id`).
- **Agentic loop**: Wonder drives the run through four function tools plus `finish`. Tool calling is CONFIRMED for
  Wonder on the models page, so the model is the driver; `tool_choice` stays `"auto"` and every cap (turns, calls,
  spend, one revision pass) is enforced in our loop because `max_tool_calls` does not cover client tools. A code
  orchestrator (`--driver code`) runs the same tools with Wonder making each decision in JSON, as the fallback if
  Wonder's tool calling misbehaves on the first keyed run.
- **Watch**: renders are uploaded with the Files API (`expires_after` 1 day) and sent as `input_video` + `file_id`.
- **Transcribe**: file mode, `ENDPOINTING`, 24 kHz mono WAV, chunked under 10 minutes.

## Checklist for the first keyed run (every UNCONFIRMED item above)

1. Wonder accepts `input_audio` by `file_id` and says something true about the song's structure and energy.
2. Wonder accepts `input_video` by `file_id` on a 1080x1920 MP4 and its timestamps line up with the render.
3. Wonder emits `function_call` items for our tools and continues correctly from `function_call_output` with
   `previous_response_id`.
4. `text.format` json_schema with `strict: true` is accepted for the treatment schema, and the output validates.
5. Real token counts for audio and video seconds (read `usage` from the first calls; the budget's per-second
   assumptions are in `realone/director/budget.py`).
6. Whether `POST /v1/responses/input_tokens` is free.
7. Voice Transcribe on a sung vocal over the full mix: how many turns, how wrong (compare with the artist's lyrics).
8. Default reasoning effort when omitted (the director always sets it, so this only matters for comparisons).
