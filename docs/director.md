# Director (the Subtext Cut)

The brain of Real One: the model that directs. Muse Wonder reads what the song MEANS (not what the lyrics say),
writes a treatment per section, and then runs the production itself through tools: it builds worlds with Muse
Image, renders sections with the real renderer, watches its own cut with video in, and revises once.
Built 2026-10-29 with no Meta API key: every model response so far is a FIXTURE, and zero paid calls were made.
What the docs confirm and what they do not: `docs/meta-api-director.md`.

## Run it

```
# offline, no key, no spend (FIXTURE responses, renders with --mask local --local-plate, outputs TEST_LOCAL_*)
.venv/bin/python -m realone.director plan   --song footage/test_song.wav \
    --lyrics tests/fixtures/director/test_song_lyrics.txt --dry-run --tag dry
.venv/bin/python -m realone.director direct --clip footage/perf_test.mp4 --song footage/test_song.wav \
    --lyrics tests/fixtures/director/test_song_lyrics.txt --dry-run --tag dry            # --driver code also works
.venv/bin/python -m realone.director review --render out/TEST_LOCAL_director_dry_r0_s2_collage.mp4 \
    --treatment out/TEST_LOCAL_director_dry_treatment.json --section 2 --dry-run

# keyed (MODEL_API_KEY in the environment): quotes US$ and CA$ first, refuses a ceiling over US$2 without --yes-spend
.venv/bin/python -m realone.director direct --clip footage/take1.mov --song footage/master.wav --lyrics lyrics.txt

# the same treatment from Muse Spark Open on Baseten (BASETEN_API_KEY): open weights, plan only, no transcription
.venv/bin/python -m realone.director plan --brain spark-open --no-transcribe --song footage/master.wav \
    --lyrics lyrics.txt --tag spark

# the same treatment from Muse Glimmer 30B (open weights, Apache-2.0) on a GPU rented for the length of the command:
# the pod is created, the model loads, the plan runs, the pod is deleted (refuses a RunPod balance under US$1)
scripts/glimmer_pod.py run -- .venv/bin/python -m realone.director plan --brain glimmer --no-transcribe --no-audio \
    --song footage/master.wav --lyrics lyrics.txt --tag glimmer

.venv/bin/python -m pytest -q tests/test_director*.py
```

If the test song or take is missing: the commands in `docs/implementation-notes.md` ("Run it") regenerate them.

Outputs (`<p>` is `TEST_LOCAL_` on a dry run): `out/<p>director_<tag>_timeline.json`, `_lyrics.json`,
`_treatment.json` (per section: timing, lyric and its source, subtext, world_prompt, palette, camera_energy, look,
freeze, why, the guard's score, and the first-pass verdict if the guard rejected it), `_log.json` (every model turn,
tool call, tool result, render, watch, revision, and the spend ledger), `_review.json`. Renders are the renderer's
own files: `out/<p>director_<tag>_r<n>_s<section>_<look>.mp4` plus its `_run.json` and contact sheet.

## Pipeline

| Step | Module | Model | Output |
| --- | --- | --- | --- |
| timeline | `realone/audio.py` (unchanged, imported) | none | beats, downbeats, sections, hook |
| lyrics | `director/lyrics.py` | Muse Voice Transcribe, file mode | lines per section, source per section |
| treatment | `director/treatment.py` | Muse Wonder, json_schema, song as `input_audio` | treatment per section |
| guard | `director/guard.py` | none (deterministic) | score, shared words, concrete hits |
| loop | `director/loop.py`, `director/tools.py` | Muse Wonder with function tools | renders, notes, one revision |
| budget | `director/budget.py` | none | quote, meter, refusals |
| backends | `director/backend.py` | Meta Model API, Muse Spark Open on Baseten (plan), or FIXTURE | swappable |

## Decisions (made without James, 2026-10-29; each one is reversible)

1. **The model drives.** Tool calling is CONFIRMED for `muse-wonder-1.0` on the models page, so by default Wonder
   picks the next tool, its arguments and when to stop (`--driver model`). Every tool-calling example in the docs
   uses `muse-spark-1.3`, so `--driver code` keeps the same tools with our code orchestrating and Wonder making the
   one revision decision in JSON. That is the fallback if Wonder's tool calls misbehave on the first keyed run.
   Both drivers run in the dry run and the tests.
2. **The treatment is the single source of truth.** `generate_world` and `render_section` must use the section's
   current world_prompt, look and freeze; the only way to change them is `revise`. Otherwise the model could
   re-render with a new look and dodge the one-pass limit.
3. **One revision pass** = up to 3 `revise` changes made before the first re-render; it closes the moment a
   re-render starts. A section renders at most twice (first cut plus its revision), and only if the phone take
   covers it (checked by syncing the take before the loop, so the model is told what it can render).
4. **Code owns every limit.** `max_tool_calls` only caps built-in tools (docs/tool-calling), so turns (8), paid calls
   (40), dollars, renders and the revision pass are all enforced in `tools.py` and `budget.py`. A tool error goes
   back to the model as `{"ok": false, "error": ...}`; a budget or key refusal ends the run.
5. **The guard runs in three places**: on the treatment (one re-ask naming the words; only flagged sections may
   change), in `generate_world`, and in `revise`. The model cannot talk its way past it.
6. **Guard design.** Overlap = share of the prompt's content words (stopwords out, simple symmetric stemming) that
   are in the section's lyric; over 0.20 rejects. Any concrete lyric noun in the prompt rejects, whatever the
   score: same stem, a compound (`watch` inside `wristwatch`, `club` inside `nightclub`) or a literal stand-in
   (`whip` to `convertible`, `money` to `cash`). A compound needs a word-sized other half, so `train`, `terrain` and
   `hearth` do not match `rain` or `heart`. Light words (`light`, `night`, `sky`) are not concrete: lighting is
   cinematography, not illustration. Known blind spot: a literal picture made of words neither the lyric nor the
   alias table knows. Wonder is told the rule; the guard is the part that cannot be argued with.
7. **Still literal after the re-ask**: `plan` writes the treatment, says which sections, and exits 3; `direct`
   refuses. The fix is an edit to the treatment file and `--treatment <file>`, which re-validates and re-guards with
   no model call. Never a silent pass, never an invented prompt.
8. **No people in a world.** A semantic rule rejects person words in any world_prompt: the filmed artist is the only
   person in frame (no generated faces: fans call AI-faced videos "lazy and cheap", r/eurovision, Mar 2026,
   reddit.com/comments/1rr2z9a).
9. **Freeze** is true on exactly one hook. The director reads `python -m realone render --help` at
   runtime and passes the freeze flag it finds there. A flag whose value is optional is passed bare; one that requires
   a value is not guessed at, and the log says freeze was planned, not rendered. The 0.2.0 renderer's
   `--freeze [SECTION]` (bare = the rendered section, `--freeze-bars` default 2) is detected as `--freeze`, which is
   what the director passes for the frozen hook.
10. **Lyrics.** The artist's sheet outranks the transcript, section by section; the transcript is kept beside it as
    `heard`. The sheet is in song order (the brief's rule): `[#N]` binds a block, other bracket lines such as
    `[Verse 1]` are labels only (the analyzer's non-hook role names are cosmetic, e.g. the test song's second verse
    is labelled `bridge`). With fewer blocks than sections, leading intros and trailing outros are dropped first;
    any other mismatch refuses and asks for `[#N]`.
11. **Transcription**: `ENDPOINTING` mode (gives turns), 24 kHz mono 16-bit WAV, 540 s chunks (under the 10-minute
    and 32 MB caps), chunk offsets added back, words from the artist's sheet sent as `keywords`. A turn is placed by
    its overlap with each section, not by its edges, because the docs say the edges are detection points.
12. **Song audio to Wonder** as a 96 kb/s mono MP3 uploaded with the Files API (expires after one day) and sent as
    `input_audio` by `file_id`. If the API rejects it (HTTP 400 naming audio), the treatment is made once more from
    the timeline and lyrics only, and the treatment file says so in `audio_in`. That is the one admitted fallback.
13. **Structured output**: `text.format` json_schema with `strict: true`, but the copy sent to the API keeps
    structure only (type, properties, required, additionalProperties, items, enum, description). Lengths, ranges
    and hex patterns are checked locally, because the docs do not list which validation keywords strict mode
    accepts and an unknown one would be a 400 on the first call. Every error (schema and semantic) is collected
    so the one repair turn can fix them all.
14. **Budget**: two numbers per quote. The ceiling puts every call at its `max_output_tokens` (which counts
    reasoning tokens, so it is a true bound on output) with conditional calls included; the cap is checked against
    it. The expected figure uses typical outputs. Audio (50 tokens/s) and video (300 tokens/s) input rates are
    ASSUMED; the meter reads real `usage` from every response and refuses before any call that could pass the cap.
    `--yes-spend` raises the runtime cap to the quoted ceiling. Each render gets `--max-usd` set to what is left.
15. **Effort**: `high` for the treatment (the creative act), `medium` for repair, re-ask, loop turns, watch and decide.
    Every Wonder call is streamed (long non-streamed calls can hit a 504, docs/protocols/responses).
16. **Watch only sees this run's renders**, and `review` only uploads files under `out/`: nothing else can reach the
    Files API.
17. **No new dependency.** The validator is our own (no `jsonschema` in the venv); Voice Transcribe goes over stdlib
    `urllib` (`requests` is only a transitive dependency of librosa's pooch).
18. **Dry run** replays hand-written responses queued per purpose (`tests/fixtures/director/dry_run_script.json`);
    every one is labelled FIXTURE in its file, its source field and the outputs. When a queue runs out the run
    refuses rather than invent a response. The fixtures are written to exercise the machinery: the first treatment
    is invalid (repair turn), the repaired one is literal in two sections (guard re-ask), and the first watch asks
    for a revision (one pass). `@render:N` in a fixture means "the latest render of section N" and only resolves on
    the FIXTURE backend. The renders themselves are real (`--mask local --local-plate`).
19. **Backends are swappable**: `respond`, `upload`, `transcribe`, `image`. `SparkOpenBackend` (2026-10-30) runs
    Muse Spark Open on Baseten behind the same four methods for `plan --brain spark-open`, so one song gets a Wonder
    treatment and a Spark Open treatment through the same guard and validation. It drafts only: watching a render,
    transcription and worlds stay on the Meta Model API. Chat completions keep no state, so `previous_id` replays
    stored history; there is no Files API, so the song goes inline as base64 `input_audio` (capped at 20 MB).
    Prices: US$1.25 in, US$0.15 cached, US$4.25 out per 1M tokens (baseten.co/library/muse-spark-open, read
    2026-10-30). UNCONFIRMED until the first keyed call: the model slug (`-1026` from the hackathon rules vs the
    library page's bare `muse-spark-open`; `--spark-model` switches), the audio part shape, and strict
    `json_schema` support (a failed parse still gets the one repair turn).
20. **Muse Glimmer, self-hosted** (2026-11-01, wired, not yet run). `GlimmerBackend` is the Spark Open backend pointed
    at a llama.cpp server (`GLIMMER_BASE_URL`, the pod's `/v1`), so the treatment goes through the same schema,
    guard and validation. Glimmer is image-text-to-text: no audio input (the backend refuses one before sending),
    so `--no-audio --no-transcribe` and the lyric sheet; Wonder hears the song, Glimmer reads it. The weights are
    Meta's official GGUF (`meta-models/Muse-Glimmer-30B-GGUF`, Q4_K_M, 16.8 GB; a dense 30B, so it cannot load on a
    16 GB Mac), served by llama.cpp's CUDA image on one 24 GB GPU; llama.cpp's source has the architecture, the
    vision projector and the chat template. Strict `json_schema` becomes a llama.cpp grammar. A call meters US$0:
    the GPU bills by the hour, and `scripts/glimmer_pod.py` reports that cost when it deletes the pod. Submission
    wording: "self-hosted on a rented GPU", never "own hardware".
21. **The watch is told what it cannot see** (2026-11-01). The first keyed watch called a working dolly zoom "not
    visible" and asked a world prompt to remove our dust overlay. `watch` now sends the move with its seconds in the
    clip and its on-screen signature (a dolly's background magnification, (z + dz) / z, from the solved subject
    distance), and names the particles as our overlay; `revise` can change them (`particles`: dust, rain, embers,
    none). `particles` is local-only in the section schema (never in the strict schema sent to Wonder).

## Dry run on the synthetic test song (2026-10-29)

Every model response below is a FIXTURE; every render, check and file is real. Command: the `direct --dry-run`
line under "Run it". Exit 0, US$0.00.

- **Timeline**: 98.0 BPM, 5 sections (#0 intro, #1 verse, #2 hook, #3 "bridge" (the second verse), #4 hook). The take
  synced at +26,314 ms and covers section 2 only, so the director could render the first hook and nothing else.
- **Lyrics**: the FIXTURE sheet's 4 blocks landed on sections 1 to 4 (the intro dropped as instrumental); the FIXTURE
  transcript's 6 non-empty turns were placed by overlap and kept as `heard` (it mishears "rent due" as "rent to";
  the artist's text won).
- **Treatment**: the first response failed validation with 5 errors (a missing `why`, a named colour instead of hex,
  an unknown look, a freeze on a verse, no frozen hook); the one repair turn named all five and the repaired
  treatment validated. The guard then rejected two world_prompts: section 1 (score 0.50; phone, kitchen, counter)
  and section 2 (score 0.44; hand, key). The one re-ask named those words; both rewrites (an empty hotel lobby, an
  empty laundromat after midnight) passed at 0.00. Freeze on the last hook.
- **Loop (model driver, 7 FIXTURE turns)**: generate_world for section 2, render_section collage (192 s, 3.06 fps,
  cuts 3/3 on plan, audio 0.12 ms off the master), watch (revise: the collage boil fights the subtext at 9.8 s),
  revise look collage to receipt (the one pass), render_section receipt (161 s, cuts 3/3, audio 0.12 ms), watch
  (keep), finish. The two contact sheets show the change: paper-cut border on r0, grey world with the figure in
  colour on r1.
- **Review**: `review` on the first render replayed the first FIXTURE watch (4 timestamped notes, suggestion: look to
  receipt).
- Files: `out/TEST_LOCAL_director_dry_{timeline,lyrics,treatment,log,review}.json`,
  `out/TEST_LOCAL_director_dry_r0_s2_collage.mp4` (+ `_contact.jpg`, `_run.json`),
  `out/TEST_LOCAL_director_dry_r1_s2_receipt.mp4` (+ `_contact.jpg`, `_run.json`).
- What this does NOT show: anything about Wonder's taste, its tool calling, its hearing or its watching. Those are
  the keyed run's checklist (`docs/meta-api-director.md`, last section).

The same machinery on a keyed run, quoted by `budget.py` with today's assumptions: this test song, hook only,
ceiling US$1.89 (about CA$2.65), expected US$0.71 (about CA$0.99), under the US$2 cap (the dry run printed this
quote). A 3-minute song with 8 sections all covered by the take: ceiling US$3.49 (about CA$4.89), expected US$1.99
(about CA$2.78), so it needs `--yes-spend`. `plan` alone on that song: ceiling US$0.77, expected US$0.18.

## Tests (`pytest -q tests/test_director*.py`, 72 tests, no network, no key)

- `test_director_guard.py`: literal prompts rejected on lyric lines written for the tests (wristwatch for
  "watch me move", keys and hands, a convertible for "whip", cash for "money", rain), subtext prompts passed (including
  `train`/`terrain`/`hearth` near misses), the score and threshold, the re-ask naming its words; the schema and the
  semantic rules, every error reported at once, the structure-only wire schema.
- `test_director_lyrics.py`: turns placed by overlap (a straddling turn goes where most of it is), chunk offsets
  back in song time, the sheet aligned in order with the intro dropped, `[#N]` binding, ambiguous sheets refused,
  the artist's text outranking the transcript.
- `test_director_treatment.py`: no repair when valid, exactly one repair naming the errors, refusal after a second
  failure, one guard re-ask that may only change flagged sections, still-literal reported, the audio fallback, a
  hand-edited treatment re-checked; the quote over the cap refused without `--yes-spend`, the meter stopping before
  a call that would pass the cap or the call cap, the documented prices, the key gate.
- `test_director_tools.py`: strict tool specs, dispatch errors returned to the model, the treatment as the source of
  truth, dry-run and keyed renderer argv, freeze flag detection, uncovered and failed renders, the one revision pass
  closing on the re-render, three changes at most, watch only on this run's renders, a budget refusal ending the run,
  the model driver running the FIXTURE script to finish, the turn cap, a silent model, the code driver.
- `test_director_cli.py`: keyed runs refuse in under 10 s without a key; over the cap refuses before a client exists
  and never prints the key; the dry-run plan writes FIXTURE-labelled outputs; review never uploads outside `out/`.
- `test_director_wire.py`: the real OpenAI SDK against a mock api.meta.ai on 127.0.0.1: the treatment request (json
  schema strict, `input_audio` by file id, effort, max_output_tokens, stream), flat function tools with
  `previous_response_id`, parsed function calls, usage and metering, the Files upload with its expiry, the Voice
  Transcribe multipart, an audio rejection typed and the key scrubbed.

Mutation checks (2026-10-29: each guard broken on purpose, its tests went red, the file restored byte-identical by
sha256): M1 guard concrete-noun check off; M2 treatment validation skipped (no repair); M3 budget cap gate off;
M4 revision pass never closes; M5 transcript turns placed by start time instead of overlap; M6 missing-key gate
removed from the CLI; M7 guard re-ask allowed to change unflagged sections; M8 no-people rule off; M9
`generate_world` accepts a prompt that is not the treatment's; M10 the meter's spend check off.

## Not built, on purpose

- A "literal" toggle (render the wristwatch version for laughs, side by side). Cheap to add on
  top of the guard's first-pass prompts, which are already kept in the treatment file.
- More than one take per run (the renderer takes one `--clip`).
- Spark Open or Glimmer in `direct` (the loop and the watch stay on Muse Wonder). Glimmer could watch a
  contact sheet as an image with `--driver code`; not built.

## Camera moves (added 2026-11-01)

Every section of the treatment carries a `move`: none, dolly (at most once in a song), crash or swap (never on the
frozen hook). The director picks it from what the section means, the same way it picks the world. Renders pass it
to `realone render --move` (see docs/stage.md); a dolly needs the tracked stage (`direct --stage`).
`realone render --moves-from <treatment.json>` takes the moves for a whole-song render.

