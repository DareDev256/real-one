# Changelog

## Unreleased
- **The tracked stage, `--stage tracked`** (docs/stage.md). The world is rebuilt in depth and filmed through the
  phone's own solved camera, instead of a still sliding behind a cut-out.
  - **Camera:** a planar track of the floor the artist stood on, with him masked out; a camera solve from the floor
    homographies (normal held fixed, Procrustes per frame, low-parallax frames skipped); the horizon from the floor's
    vanishing line; the camera height from the figure standing on the floor.
  - **World:** Depth Anything V2 Small on the Neural Engine through Core ML (Apache-2.0, sha256-pinned). Each still
    splits into a backdrop at infinity, a fitted floor plane and up to 8 depth cards with inpainted backs. Every layer
    is warped by its plane homography, and the floor is pinned to the tracked floor every frame.
  - **Integration:**
    - a contact shadow, on the floor only, cast from the sun the world still shows;
    - depth of field calibrated to the take: the take's background/performer focus ratio is measured (0.75 on
      the test plate) and the world's thin-lens blur is chosen to match. A median-based probe read every focus as
      equal (8-bit steps); the trimmed mean replaced it and was checked against a background blurred on purpose;
    - motion blur along the floor's own motion between frames;
    - the take's grain given to the world, after any punch-in;
    - a ring grade split into light level and colour lean (a dark warm world can still warm him), plus a key
      ramp toward the world's sun (filmed + graded in the Receipt, never generated);
    - clean-plate feet: the real floor, rebuilt from 6 tracked neighbours, takes floor pixels out of the matte's
      outer 10 px at his feet, only in regions that touch the edge (never adds, never touches the interior); the
      Receipt counts the refined matte.
  - **Refusals:** a floor with too few features, too little parallax to see the floor's slope, an artist never seen
    head to feet, or a still that no lens can stretch over the camera's view.
  - **Measured (lab, moving-camera plate):** floor slip near the feet falls from 6.31 to 0.38 px/frame median, and
    from 814 to 59 px over 5 s. The probe reads rendered pixels only and was checked against the plate itself (0.00)
    and a known shift.
  - **End to end:** `TEST_LOCAL_stage1` and a stage Hook Freeze render pass every existing render check (frames
    220/220, cuts on plan, audio +0.5 ms, freeze state identical on all 74 frozen frames). The freeze release
    downbeat is now counted as a designed whole-frame change: the cool grade snapping off read as an off-plan cut
    when a freeze ended inside a shot (also true without the stage).
  - **Blind gauntlet (a fresh critic each round, the real plate as the bar):**
    - Round 1: the 0.2.0 composite 2/10, the stage 6/10.
    - Round 2: focus calibrated; the next gap was test-matte leaks at the feet.
    - Round 3: SAM 3.1 mattes plus the clean-plate feet beat round 2 in both worlds (6 vs 4.5, 4.5 vs 3.5). Its gap
      was the refiner eating a floor-coloured stripe on the shoes. Fixed: a 10 px band, regions must touch the
      matte's edge; the critic's own pass test went from 12 frames to 1 (by our reproduction).
  - **Tests:** `tests/test_stage.py` has 12 tests. They include a camera solve against a synthetic floor filmed by a
    KNOWN dolly-back camera, which catches the sign twin that reads a dolly back as a dolly in. 12 guards were broken
    on purpose and all went red. The mutation pass also caught a real bug: the drop shadow was cast toward the lens.
  - **Credits:** techniques harvested from the #OpenSourceHiggsfield entries pickups, platelock, hero-frame-rig,
    ReAngle, hack-comfy, swapjutsu and shotcheck; listed in docs/stage.md.
- **Muse Spark Open on Baseten as a second director brain** (`plan --brain spark-open`, for the open-weight bonus).
  `SparkOpenBackend` in `realone/director/backend.py` implements the same four methods as the Wonder backend over
  OpenAI-compatible chat completions: instructions as a system message, the treatment schema as a strict
  `json_schema` response_format, tools nested, function outputs as tool messages, `previous_id` replayed from stored
  history, the song inline as base64 audio (no Files API; refused over 20 MB, nothing sent). The quote and meter use
  Baseten's published prices (US$1.25 / 0.15 / 4.25 per 1M tokens, read 2026-10-30). Refuses before any work
  without `BASETEN_API_KEY`, with transcription on, or for `direct`. Keyless: 14 tests on a fake client, including
  a full treatment through `treatment.make`; 9 guards broken on purpose, all red, files restored byte-identical.
  Not run with a key yet; the slug, audio part and strict schema are UNCONFIRMED until the first call.
- README rewritten for a judge with two minutes, after a blind critic ranked it last of four (no visible proof, and a
  `[~]` checklist that read as "not built"). It now leads with the concept line, a Hook Freeze GIF and a Real Receipt
  still (stamped TEST RENDER until the first keyed render), a model table, what it does and the engineering. The
  status checklist moved to `docs/status.md`; the local key path and diary phrasing are gone.
  `scripts/readme_media.py --tag --section` regenerates both images from any render; it stamps TEST RENDER on a
  TEST_LOCAL tag, a local mask mode or a test mask engine, and refuses footage/own/ and missing looks. 7 tests,
  3 mutations red.
- **Watch page (`web/`, `scripts/build_watch_page.py`)**: the link a judge opens. One static page (index.html,
  watch.css, watch.js; no framework, no build step, Google Fonts only) that loads a `data/` folder written by the
  build script from one render's own files. Two `<video>`s of the same cut (the finished look with the sound, the
  receipt look muted on top); the Real Receipt toggle (button or R) cross-fades between them at the same frame
  (instant under prefers-reduced-motion). The follower is held on the master's clock: it waits while the master's
  clock is stuck (audio start, a stall after a seek), jumps when it is off by 2.5 frames or more (aimed ahead by a
  learned seek lead), and otherwise takes a rare rate nudge (WebKit drops about a frame on every rate change).
  A till-receipt counter (FILMED / GENERATED for the frame on screen, from `_receipt.json` at currentTime x fps, plus
  the whole video), a section strip from the song's timeline (click a section for its start, click or drag the
  track below it to scrub; cuts on downbeats and the Hook Freeze marked, bar numbers on the events), the director's
  treatment beside the lyric (the current line lit from the lyrics file's timings; the guard's rejected first
  pass struck through; hidden when no treatment is given), and "How it was made" from `data/credits.json`.
  Space plays and pauses.
- The build script copies and re-encodes the two looks for the web (H.264 high, yuv420p, faststart, 1 s GOP, at
  most 1080x1920, crf 22 capped at 6 Mb/s; the receipt look without audio), probes them after encoding and refuses
  when they are not fit for the web or their frame counts disagree with each other or with the receipt. It also
  refuses a missing file, a render with no receipt look, a tag that matches both a real and a TEST_LOCAL_ render,
  any input or output under `footage/own/` (as given, through symlinks, in any case), a treatment for another song
  or other section timings, a real render with no `--credits` file, a credits file that claims a Meta model ran on
  a test render, and an `--out` folder it did not make. A TEST banner is written into index.html itself whenever
  the tag starts with TEST_LOCAL_, the masks were local, the worlds are the test plates or the treatment is a
  FIXTURE; a test render's credits are generated with every model `ran: false` and US$0.00. Local paths are
  stripped from every json it copies. `scripts/serve_watch_page.py` serves the build on 127.0.0.1 with byte ranges
  (the stock http.server has none, and WebKit will not play a video without them).
- 27 tests (`tests/test_build_watch_page.py`, fake encoder, no ffmpeg); 22 mutations of the script's guards, each
  red, the file restored byte-identical by sha256.
- Sample build from a keyless render (`--section bars:11-20 --look collage,receipt --mask local --local-plate
  --freeze hook --tag web_sample`; 661 frames, cuts 3/3, freeze OK) with the director dry-run treatment
  (`TEST_LOCAL_director_integ02`, sections 1 and 2 line up). Driven with Playwright on Chromium 1440, WebKit 1440 and
  WebKit iPhone 13 (play, toggle mid-play, seek by section and by the track): the two videos within 0.2 to 0.9
  frames in play and through the toggle, up to 3 frames for about half a second after a seek in WebKit, then under
  1; the counter changes with the frame; no horizontal scroll at 390 px. Screenshots:
  `docs/screenshots/2026-10-29_watch-page-{desktop,mobile}.png`. `web/dist/` is build output and is ignored.
- **Hook Freeze v2, `--freeze-style bullet` (the new default; `still` keeps the 0.2.0 freeze for A/B).** The owner
  watched the 0.2.0 freeze on his phone and called it "pretty meh": with the camera stopped too, the world is a still
  image behind a moving person (green screen), and nothing in the frame proves time stopped. Now, in a render with a
  freeze: the particles sit on three depth layers (far, mid, near; size, blur, brightness, fall speed and parallax all
  rise with nearness; near drawn over the artist) and fall as motion-blur streaks; on the freeze downbeat they stop
  dead in world space and redraw as short crisp marks, brighter (the streak's light collapses into a point), while a
  sideways camera truck on the real frame clock keeps moving (the plate 3.5% of the frame width over the freeze,
  far 2.5x that, mid 4.5x, near 9x), so the hanging layers slide past each other. The hit: a 3-frame punch
  (1.04 -> 1.0) and a white flash (30%, 15%, 5%); the world (not the artist) grades to saturation x0.45 and cooler
  over 3 frames. The release: the grade snaps off, the particles fall again with a 3-frame streak burst, the camera
  carries on at the same speed. Pure functions of the world clock and the frame index, as before. A render with no
  freeze, and `still`, are unchanged (still: same sheet, same checks).
- Bullet checks (exit 2 and `freeze` in the run json on a failure, as before): the world-space state the units drew
  from (every particle position, the plate's world clock, the push) is identical on every frozen frame; the camera
  path moves (every step, at least 1% of the width) and the rendered world layer changes on every frozen frame (the
  move is in the pixels); the artist moves; the unfrozen control's particles move; the state changes outside a
  freeze; no jump on the release: clock step 1, particle step, camera step and plate step each within 3x the
  median after. Measured on the hook (collage, local plates): state identical on 147/147 frozen frames, camera
  37.5 px, all checks OK; `still` on the same hook: unchanged checks OK.
- Cut detection median-filters its thumbnails (5 px): the denser rain raised the frame-to-frame baseline until a
  real world swap measured 5.7x and was refused; filtered, cuts measure 9.2x and up and other frames 3x or less.
  A detection on the 3 frames after a bullet freeze's downbeat is the hit (punch + flash), reported as
  `detections_on_the_freeze_hit`, not as a cut off the grid.
- 13 tests at first (`tests/test_freeze_bullet.py`); mutation checks M17 to M28 on a scratch copy of the repo, each red,
  each file restored byte-identical by sha256 (see docs/implementation-notes.md).
- **Bullet plan rule: the freeze downbeat is not a cut** (`plan_range(freeze_style=)`; `still` keeps its rule).
  The shot running into the freeze keeps its world, framing and camera through the downbeat, even on a section
  boundary, so the drops the viewer watched fall are the ones that hang; cuts inside the freeze and on its release
  are dropped too (unless the release is a section boundary), so the same world resumes. A freeze on the render's
  first frame (`--section hook --freeze`, the director's call) gets a 2-bar lead-in in the same world
  (`freeze.lead_in`); when the song, the take or `--max-seconds` cannot cover it, the render opens on the downbeat as
  before and the run json says why (`freeze.lead_in`). New checks: no cut on the downbeat, one world from the frame
  before the downbeat through the release. Measured on bars 11-20 (lead-in pair): bullet cuts 3/3 (bars 11, 15, 17,
  19), frames 70-75 one world, every check OK; still unchanged (cuts 4/4, the freeze on a cut).
- The camera truck eases in before the downbeat and out after the release (smootherstep; it still moves on every
  frozen frame, 37.8 px over the freeze); the release plate-step check now measures against the camera's own motion.
  Rain dashes vary in length and brightness per particle (widest far away); rain is drawn as thin lines, so near
  drops are thin streaks, not pills.
- Director: nothing depends on a freeze starting on a cut (it reads `cuts_on_plan`, `spec_ok` and the audio error,
  and finds bare `--freeze` in `--help`); director tests and the `integ03` dry run pass (its fixture freezes no
  covered hook, so the director's own freeze argv, `--section #2 --freeze`, was run too: exit 0, every check OK).
  On the test take that hook gets no lead-in: the take starts 1.42 s after bar 10, so the run json says so and the
  render opens on the downbeat (1 bar would fit; the rule is 2 bars or none).
- 17 bullet tests; mutations M29 to M34 red (downbeat cut restored, release cut allowed, both new checks forced
  true, lead-in never applied, camera ease removed).
- Everything is still on the drawn test figure and code-drawn plates.
- `scripts/first_keyed_run.sh`: the first keyed runs in two capped stages, `probe` (synthetic take, US$0.50) and
  `real CLIP SONG` (US$1). Reads the key from `~/.secrets/meta-model-api` without printing it; refuses the gym
  clip by real path, symlinks included (guard mutation-checked).
- Sections under 4 bars fold into the neighbour they sound most like (`MIN_SECTION_BARS`). On three real songs
  (analysed locally, nothing sent anywhere): 14 -> 12, 12 -> 10 and 17 -> 15 sections, none under 4 bars, the hook
  label unchanged on each; the synthetic test song is unchanged. Labels that never repeat remain (not tuned: no
  ground truth yet).
- Rights: the commercial lyric used as the "too literal" example (in the guard, the treatment prompt sent to Muse
  Wonder, a test and the director doc) is replaced with an original line, "watch me move".
- README: "with depth layers" and "one tap" were wrong (only the particles have depth; the reveal is the
  `--receipt-reveal` flag); Subtext Cut status now matches the director's.
- SAM upload: the take is uploaded once per run however many `--sam-prompt` phrases (was once per phrase, about
  600 MB each for a 3-minute real take at crf 12), with a one-day expiry on Meta's side (was kept until deleted).
  Refuses before any byte or call when the take is over the Files API's 1 GiB or SAM's 15,000 frames. 3 tests,
  4 mutations red.
- Director checked against the 0.2.0 renderer (no code change needed): the dry run end to end on the test song
  (2 renders, cuts 3/3, audio 0.14 ms off) and the director's own freeze handoff (bare `--freeze` found in `--help`,
  frames 0-146 identical, the world released after).
- `pytest.ini`: a bare `pytest` at the root collects `tests/` only (two files shared a basename).

## 0.2.0 (2026-10-29): the engine
Hook Freeze in the renderer, native-resolution punch-ins, the whole song in parallel, the Real Receipt number,
collage legibility. Every 0.1.0 flag works unchanged; every new flag is optional. Keyless proven; the SAM 3.1 and
Muse Image paths stay unproven until the key lands. Measurements, decisions and limits: `docs/implementation-notes.md`.
- **Hook Freeze** (`--freeze [SECTION]`, `--freeze-bars N`): on a section's first downbeat the world clock stops
  (plate push-in, drift, particles, camera) while the artist and the grain keep moving; it resumes one frame step
  on, and the camera eases back over 0.5 s. Proven on every render by hashing the world layer: identical through
  the freeze, moving outside it, no jump on the release, the artist moving, an unfrozen control moving; a failed
  check exits 2. A freeze starts on a cut; cuts inside it and on its release are dropped so the frozen world is
  the one that resumes. Freeze contact sheet. A bare `--freeze` is what the director's `--help` probe passes.
- **Native punch-ins**: the take is normalized once into a 1080x1920 proxy and a frame-aligned native 9:16 crop (one
  ffmpeg graph); medium and close shots take their pixels from the native frame, and close shots edge-refine the
  mask against them (colour guided filter): close-shot Laplacian variance 26.5 -> 860.6 on the synthetic 4K take
  (mostly native-only texture; drawn edges alone 1.36x), wrong edge pixels 0.55% -> 0.38%. HDR (HLG/PQ) phone clips
  tone-map through VideoToolbox, with the rotation applied by hand and fps=30. Fixed: a phone clip's rotation was
  never read (sideways output); frame counting crashed on phone side data. `--no-native` for A/B;
  `spike/make_test_clip.py --size WxH`.
- **Full song** (`--section all`, `--worlds-json`, `--section-world KEY=PROMPT`, `--workers N`): one world list per
  section; render units in worker processes (default CPU cores minus 2, capped by free memory after 10 workers
  thrashed swap on a shared 16 GB Mac); raw H.264 segments (no B-frames) joined byte for byte and muxed with the
  master, BT.709 now written into the stream. The synthetic full song (2645 frames, both hooks frozen) in 370 s
  wall with masks cached, under load 11 to 18; a 3-minute song extrapolates to about 15 minutes at that load.
  Long takes now sync (judged on all their windows; a whole-song take was refused before). Local masks use person
  segmentation for real people, written as a memmap by several processes.
- **Real Receipt**: every render writes `out/<tag>_<section>_receipt.json`, the filmed share of every frame, of
  each section and of the whole video (the artist mask over the frame, the rest generated). `--receipt-reveal
  SECTION` flips a section (e.g. `hook:last`) to the receipt look, on cuts. `--receipt-counter` draws
  `FILMED 24.7%  GENERATED 75.3%` in DIN Condensed, set like a till receipt, placed per shot where the artist is
  not, inked against the ground; it matches the mask area within 0.5 points (tested).
- **Collage legibility**: when more than 35% of the band under the paper border is lighter than luma 0.72, the
  border becomes a dark outline for that shot.
- **SAM prompt union** (`--sam-prompt`, repeatable): one call per phrase, OR-ed, so a held prop stays in the cut-out.
- 75 engine tests (147 with the director's); mutation checks M9 to M16, each guard broken on a scratch copy of the
  repo, its test red, restored byte-identical: freeze hold removed, release snapping to real time, native frame
  ignored, a unit dropping its last frame, the long-take t-test dropped, the counter under-reporting by 10%, the
  collage border never switching, the SAM union keeping one phrase.
- Pinned: pillow 13.0.0.
- **Director** (separate agent; commits cfce1e3, defdf5b): `python -m realone.director plan|direct|review`: Muse
  Voice Transcribe lyrics (the artist's sheet outranks it), a Muse Wonder treatment per section (subtext,
  world_prompt, palette, camera_energy, look, freeze, why) with strict validation, a literal-lyrics guard, and a
  Wonder-driven tool loop (generate_world, render_section, watch with video in, revise) with one revision pass.
  Spend is quoted in US$ and CA$, with a US$2 cap unless --yes-spend. --dry-run uses FIXTURE responses. See
  docs/director.md.

## 0.1.0 (2026-10-29): the floor
One song section, end to end, as a package: `python -m realone render --clip <take> --song <master> --section hook
--look collage|clean|receipt|all`. Keyless test runs (`--mask local --local-plate`) write `out/TEST_LOCAL_*`.
- `realone/audio.py`: beat grid (librosa, constant-tempo fit, 21 ms onset-lag correction), downbeats (low-band
  kick + chord change phase), bar-level sections (slow-feature weighting, PCA, over-segment then label, merge and
  refine), hook = loudest repeated section. Writes `out/<tag>_timeline.json`.
- `realone/sync.py`: phone mic vs master by onset-envelope cross-correlation over percussive and harmonic
  bands; parabolic sub-hop refinement; container start skew applied. Refuses (exit 3) on a weak match, a silent
  mic, or a take that no window can place uniquely (a copy-pasted loop); `--sync-near` for short takes.
- `realone/plan.py`: cuts on downbeats every `--cut-bars` (default 2), each cut a world swap plus a reframe
  (wide 1.0x, medium 1.4x, close 1.9x) centred on the smoothed mask bounding box. Writes `out/<tag>_<section>_shots.json`.
- `realone/composite.py`: edge despill for every look (local F/B colour unmixing in the edge band, plus small
  enclosed pockets of the source room, scaled with the reframe zoom); three looks: `clean` (the spike's),
  `collage` (paper border, hard offset shadow, paper grain, 12 fps boil over a 30 fps artist), `receipt`
  (generated pixels grey, the filmed artist in colour).
- `realone/masks.py`, `realone/worlds.py`, `realone/encode.py`: the spike's SAM path, Apple Vision path, Muse
  Image call, particles and encoder moved into the package; Muse Image now one world per prompt, cached by prompt
  hash; three procedural test plates (neon alley, sunset roof, acid stage).
- After every render: spec check; hard cuts detected in the encoded file must match the plan within 1 frame, with
  no extra cut; the output audio must sit on the master within 1 frame. Contact sheet per look and
  `out/TEST_LOCAL_looks.jpg`, read back from the encoded files. A render over `--max-minutes` (20) is stopped.
- Keyed path wired (`--mask sam`, `--world`), refuses without `MODEL_API_KEY` before any work, quotes spend in
  US$ and CA$, refuses over `--max-usd` (1.00). `# UNCONFIRMED` marks unchanged from 0.0.2.
- `spike/make_test_song.py` (procedural 98 BPM song with known structure and a synthetic vocal line) and
  `spike/make_test_clip.py --song` (figure bobs on the beat; phone-mic audio from a known offset).
- 32 pytest tests. Mutation checks run 2026-10-29, each guard broken on purpose, its test went red, the source
  restored byte-identical (sha256): M1 beat snapping replaced by a fixed 2.5 s grid; M2 sync no-match guard off;
  M3 sync ambiguity guard off; M4 receipt world left in colour; M5 collage boil at 30 fps; M6 despill not scaled
  by the reframe zoom; M7 key gate removed for `--mask sam`; M8 onset-lag correction removed.
- Pinned: librosa 1.0.0, scipy 1.18.1, numba 0.68.1, llvmlite 0.50.0, soundfile 0.14.0, pytest 9.2.0.

## 0.0.2 (2026-10-28)
- Hook Freeze spike: `spike/freeze_test.py` cuts the artist out of one phone clip (SAM 3.1), composites them on a
  Muse Image world with a slow push-in and a code-drawn particle layer, and freezes the world for 3 s while the
  artist keeps moving. Writes `out/freeze_test.mp4` (1080x1920, H.264, yuv420p, faststart, original audio) and
  `out/contact.jpg` (6 timestamped frames, 2 inside the freeze).
- Edge work: erode, feather, light wrap, LAB colour match, one shared film grain.
- SAM masks and the world are cached, so a rerun costs nothing. Refuses to run without `MODEL_API_KEY` unless
  `--mask local` plus a world image or local plate (test only, `TEST_LOCAL_` outputs).
- `spike/make_test_clip.py` (synthetic clip), `spike/test_sam_decode.py` (offline SAM output test).
- `requirements.txt` pinned; `docs/implementation-notes.md` lists which API shapes are confirmed and which are not.

## 0.0.1 (2026-10-28)
- Repo created: README, plan, ignore rules for footage, outputs and keys.
