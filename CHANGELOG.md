# Changelog

## Unreleased
- **The director's loop ran keyed through every step (Nov 1; neither run reached `finish`, see below).** `direct --stage` on the stand-in hook clip: Muse Wonder
  generated the world, rendered through the tracked stage with SAM 3.1 masks, watched its own cut, said "revise",
  rewrote the world prompt, regenerated and re-rendered (US$0.45, 16 calls; the run then stopped on our own
  `--max-calls 16`, before watching the re-render). Its notes found two things the code had hidden from it:
  - **The watch never told Wonder when the camera move ran or what it looks like.** It called a working dolly zoom
    "not visible" (frames 0 to 146 at full size: the world magnifies about 2.5x while he keeps his size). `watch`
    now carries the move, its seconds in the clip and its on-screen signature from the renderer's own shot plan
    (for a dolly, the background magnification (z + dz) / z from the solved subject distance).
  - **The sparkles were ours.** Wonder read the dust overlay as "glitter, magical rather than exhausted" and tried
    to remove it with a world prompt, which cannot reach a layer drawn over the image. `revise` gains a `particles`
    field (dust, rain, embers, none) that reaches the renderer (`--particles`, and `--from-treatment` for a
    section the director revised); the watch is told the overlay is ours. `particles` is local-only in the
    treatment schema, never in the strict schema sent to Wonder.
  - A watch note over 300 characters cost a re-upload and a re-watch (2 of the 16 calls): the limit is now 500.
    Run 2 hit the same on `suggestion.why` (240; twice): now 500 too.
  - **Run 2 (same clip, US$0.28, 19 calls) confirmed both fixes in Wonder's own words.** Told when the dolly ran,
    it wrote "The dolly zoom reads clearly and works for the subtext: although he keeps walking and dancing forward,
    he holds roughly the same size while the sedan and pillars behind him expand and flatten." It then used the new
    field (particles dust -> none, "the sparkling dust ... turns exhaustion and stalled refusal into a magical,
    confident dance moment") and re-rendered. Its second watch still says revise: the clean look's glossy floor
    and bright tee read "polished dance-video sheen", and the cut-out edge "pasted against the haze". It stopped on
    the 8-turn cap before calling finish (the log says so).
  - 2 tests; 8 mutations of the new guards, all red, file restored byte-identical.
- **Watch page, after a blind judge critic (Ansatz 8/8/8/8, ours 5/4/6/7 on the Stage 2 rubric):**
  - Credits count what ran: SAM 3.1 one streamed request and one mask per frame on the page, priced per 1,000
    frames; Muse Image one call per world; Wonder and Voice Transcribe the keyed ledger's calls and cost. A number
    nobody measured is `null` (the page shows n/a), never 0. The live page had shown CALLS 0 and FRAMES 0 under
    models it marked RAN: YES.
  - The footer said "FOOTAGE AND SONG ARE THE ARTIST'S OWN." on a stock clip and a synthetic song. It is now
    written by the build: that line only on a real render, "TEST RENDER: THE FOOTAGE AND SONG HERE ARE STAND-INS."
    otherwise; the build refuses a template that lost the marker.
  - Stand-ins are named: the repo's synthetic test song is recognised by content (a renamed copy too), and
    `--stand-in TEXT` names footage that is not the artist's take. The banner says what stood in and, separately,
    which Meta models really produced output.
  - `--direct-log`: the director's review of its own cut goes on the page verbatim (notes with times, the one
    change and its reason, the same moment of cut 1 and cut 2 side by side, the second watch); refuses a log with
    no watch of the section, labels a dry-run log FIXTURE (and credits no model from it), refuses a render that is
    not on disk. The loop's own calls are credited to the models that made them (Wonder: 2 treatment calls plus 12
    in the loop on the live page).
  - Live from `showcase2` (Nov 1, 23:30): the stand-in hook rendered from the director's final treatment (SAM 3.1
    masks and the Muse Image world both cached, US$0.00; frames 588/588, cuts 3/3, audio +0.1 ms). Checked live at
    1440 and 390: no horizontal scroll, both videos readyState 4, no console errors, 8 notes and both stills load
    (`docs/screenshots/2026-11-01_watch-live-*`).
  - Costs under a dime keep a third decimal; the phone banner is shorter.
  - 7 tests; 17 mutations of the new guards, all red, file restored byte-identical.
  - **Round 2 (a fresh blind critic, same rubric, Nov 1 23:30): Ansatz 7/9/9/8, ours 4/3/5/6.** Its gap for us is
    still the stand-in footage. Two of its readings were our copy: "Real Meta model output here: ..." read as "the
    output goes here" (so not on this page), and Voice Transcribe was credited although it heard no words in the
    synthetic song (every section's `heard` is empty; the lyric shown is the sheet). Now a model counts as output
    on the page only with evidence (Voice: a transcript that heard words); the banner names what each one made
    ("SAM 3.1 (the cut-out), Muse Wonder (the direction and its review), Muse Image (the world)"); Voice's card
    says it ran and heard nothing; the receipt panel says FILMED is the inside of SAM 3.1's cut-out. 4 mutations,
    all red.
- **Muse Glimmer, wired, not yet run.** `plan --brain glimmer` sends the treatment through the same schema, guard and
  validation as Wonder, to Muse Glimmer 30B (Apache-2.0, Meta's Q4_K_M GGUF) on a llama.cpp server we host. It reads
  the lyrics and the beat grid (no audio input). `scripts/glimmer_pod.py run -- <command>` rents one 24 GB GPU on
  RunPod for exactly the length of the command (deleted in a `finally`), with a one-run API key on the server
  because its proxy URL is public; it refuses a balance under US$1 or a pod already running. Status and down
  checked against the live API; `run` has not created a pod (the balance is under US$1). llama.cpp supports the
  architecture, the vision projector and the chat template (read in its source, Nov 1). 2 tests.
- **README:** the Muse Spark Open row said it writes treatments; it never ran with a key. It now says "wired, not
  yet run".
- **Lyrics in the world, `--lyrics-in-world` (Nov 2).** James asked for the on-screen lyrics viral music clips carry
  (he named Evan Blum's videos for Lay Bankz), then asked for them "3d tracked to the environment ... angled".
  Real One writes each section's lyric lines INTO the scene behind the artist, a word on each beat. On the tracked
  stage each line stands on its own sign in the world: a vertical plane on the floor 1.4 m behind where his feet are
  at the middle of the line (the contact point's ray meets the tracked floor), facing that frame's camera, turned 15
  degrees (alternating side line by line, each new line swinging in from 30). Lines change on the same downbeats as
  the shots, so each line is placed for its own shot: its centre where the ray through the upper middle of that
  shot's framing (25% down, after the punch-in) meets the plane, and 72% of that framing wide. From then on it is
  fixed in the world and the camera moves past it like anything else in the scene. Three placements were tried in a
  lab that rebuilds any frame through the real compositor from a kept render (`REALONE_KEEP_WORK=1`): a fixed 2.4 m
  sign 2 m up (words off the top and the left edge, because camera and performer travel), the same sign behind his
  feet at the line's start (words drifting out of the punched-in medium and close shots), then this one.
  - **Blind critic round 1** (a viral-clip editor's eye, frames every 0.5 s): "no" as it stood, 4/10 against the best
    viral lyric clips and 2/10 for reading as 3D. Its one gap: the words sat dead centre at head height, so his head
    and arms covered a key word about half the time, and an 18 degree turn was too gentle to read as perspective.
    Fixed: each line now stands BESIDE him, alternating left and right (19% of the shot's width off his centre line),
    turned 28 degrees with its far end toward the frame's edge (the near, bigger letters face him, so the edge never
    cuts them), two words a row (up to four rows), sized by measuring its real projected width through the render's
    own warp (the floor pin scales later shots, and a size from the bare camera came out at half).
  - **Phrases, not lines.** A lyric line is shown as phrases of at most 4 words ("lights on at four / and I'm still
    holding"), each with its own sign for its own bar (by beats when a section has more phrases than bars): a sign
    that lives half as long drifts half as far as the camera moves, and a phrase reads at a glance.
  - **The side away from him.** Each phrase stands on the side of the frame opposite his body's centre in that
    shot's framing (his tracked box), so his head is not behind the words. Fitting the sign into the free strip
    beside his box was tried and lost: with his arms out the box fills the frame and the sign shrank to nothing.
  - **Blind critic round 2** (the same brief): "with fixes", 5.5/10 against the best viral lyric clips (from 4) and
    5/10 for reading as 3D (from 2); the current phrase fully readable about 60% of the lyric time (phrases 1-2 about
    85%, the punched-in second half 30 to 50%, where his head and arms cross the words). Its next gap: anchor phrases
    to real surfaces (a pillar face, the floor) clear of his silhouette, and keep the performer-in-front effect for
    single words as a deliberate beat. Shipped as an option, not the default; the live watch page stays on the render
    without lyrics until the real take decides. It is warped every frame by the same plane-induced homography as the floor and the depth cards, with the
  floor pin and a dolly's virtual camera, so it has true perspective and parallax (a test proves it slides with a
  sideways camera move while the backdrop at infinity does not), and a depth test hides it wherever the world is
  nearer (a pillar in front of it). He walks in front of it because it is blended into the world layer before he is
  composited; his pixels are never touched and the Real Receipt counts every letter as generated. Without the stage
  the words sit in screen space, still behind him.
  - Timing is the beat grid, not the vocal: each line gets an equal share of its section's bars from a downbeat,
    its words land on successive beats (half-beats when the line is busier than its beats). Voice Transcribe gives
    one time per turn, not per word, so karaoke alignment is not claimed.
  - The Hook Freeze proof hashes the world layer: the hash is now taken before the words are blended in, so the
    world still proves it stood still while the lyrics carry on with the vocal.
  - Helvetica Neue Condensed Black, white over a soft shadow, the newest word sliding up and fading in over 4
    frames; a line keeps its final layout so words never jump.
  - Refuses: no lyric lines (it needs `--from-treatment`), no beat grid, no line inside the rendered range.
  - 14 tests (lyric timing, painter, depth test, swing, sign geometry, parallax, floor point); 21 mutations, all red,
    files
    restored byte-identical. The first render died on a one-channel colour layer that only the stage's warp exposed;
    a test now runs that path.
- **The demo's song is a licensed track (Nov 2).** James has no release of his own, so the demo is him performing a
  Creative Commons BY track from ccMixter (three human-made candidates on the shoot card; Pixabay's vocal rap was
  4 of 6 labelled "AI generated", which a pitch of "the artist is never generated" cannot use). The watch page no
  longer assumes a real render's song is the artist's own: it refuses until told `--own-song` or
  `--song-credit 'TEXT'`, refuses both at once, and a licensed track puts its credit in the footer and the credits
  statement ("the footage is the performer's own; the song is licensed: ..."). The pipeline diagram's footer said
  "Footage and song are the artist's own"; it now says the song is the artist's master or a licensed track. 1 test;
  4 mutations, all red.
- **Monday's chain, walked before the take exists.** Every flag in the edit kit checked against `--help`. Two real
  gaps found and closed: `direct` without `--treatment` would have re-planned (a different treatment from the
  full render's), so the kit now passes it; and the watch page refused a real render without a hand-written
  credits file. Credits are now generated from evidence for every render, marked REAL when nothing stood in
  (a model that did not run reads "not used on this render"; the statement says the footage and song are the
  artist's own only then), with `--credits FILE` still taking precedence. The total is what the models' output on
  the page cost, summed from the same evidence. `--direct-log` now works on a full-song render (it shows the
  section the director watched). 1 test; 6 mutations, all red.
- **README media from the real pipeline.** The hero GIF and receipt image were a drawn figure with no Meta model.
  They now come from `showcase3`: the stand-in stock clip with SAM 3.1 masks, the Muse Image world from Wonder's
  prompt, the tracked stage and a Hook Freeze with rain (all checks pass: world identical on all 147 frozen frames,
  no jump on the release; US$0.00, cached). `scripts/readme_media.py`:
  - refuses a keyed render of the synthetic test song unless `--stand-in`, which stamps "STAND-IN CLIP  real SAM
    3.1 + Muse Image" (caught by content, so a renamed copy counts);
  - when a render opens already frozen (a take that starts on the hook's downbeat), the GIF shows the release
    instead of the landing, so the rain is seen hanging and then falling;
  - the stamp sits above the frame instead of over it (it covered the receipt counter).
  - 2 tests; 3 mutations, all red.
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
    - Round 4: the fix won both worlds (7 vs 5, 5 vs 3.5). The next gap: the feet float, because the real
      plate's floor under a planted sole is about half as bright as beside it. Fixed with a contact pocket under
      the lowest sole, scaled by how planted it is. Planted is read from the contact point's motion on the
      TRACKED floor, not on screen. The pocket is tuned to the real plate's measured under/beside luminance: the
      darkest-10% frames go from 0.63 to 0.40 (the real plate reads 0.40).
    - Round 5: the pocket LOST both pairs (6.5 vs 5, 4.5 vs 3.5). It matched the probe's number but drew a hard
      rectangle that flicked on with each step, so it was removed.
    - Round 6: the artist's real shadow, lifted from the take (clean-plate ratio), plus a rule against darkening
      above the sole. Lost (4 vs 6.5): the rule's column cut-off drew a box. The rule was removed and the capture
      window feathered.
    - Round 7: real shadow alone vs the round-4 synthetic contact: 5 vs 6. It is right under a lifted foot but
      loses the dense pocket under a planted one, so it ships as `--contact real` and the synthetic contact stays
      the default. A world generated in light that matches the take (blue hour) beat the dusk world, 4 vs 3. The
      loop stopped there: rounds 4 to 7 sit at 6 to 7 out of 10 against the real plate.
- **Camera moves on the beat, `--move KIND[:BARS]@SECTION`** (`realone/stage/moves.py`). Higgsfield-style
  presets, done on Meta's worlds with the artist's filmed pixels:
  - `dolly[:BARS]@SECTION`: a dolly zoom from the section's first downbeat. The world's camera goes back 3 m
    while its lens zooms by (zs + dz) / zs, so everything at the artist's distance (zs) holds still and the world
    behind him stretches. zs is read every frame off the solved floor under his contact point and smoothed like
    a focus pull; without that, the lowest matte pixel hopping between feet made 15 px pops. It holds to the
    next cut, then resets. Needs `--stage tracked`, and refuses without it.
  - `crash@SECTION`: a snap zoom toward his face (box top + 12%, smoothed) on every downbeat, x1.28 over 4
    frames with a 3-sample zoom blur on the snap, held to the next downbeat.
  - `swap@SECTION`: a new world every bar, cycling the section's world list, with the framing uninterrupted.
    Under the stage every world is pinned to the same tracked floor. It refuses with one world, and refuses
    inside a Hook Freeze.
  - Crash snaps and swap downbeats are designed whole-frame changes, so the cut detector does not count them.
  - End to end on the test clip (`--mask local`): dolly, and swap + crash together, both pass every render
    check (frames, cuts on plan, audio +0.5 ms).
  - 7 tests (`tests/test_moves.py`), 7 guards mutation-red. A centre clamp in the crash zoom was found
    unnecessary (zooming in about a point inside the frame cannot show a border) and removed.
- `World3D.render(virtual=(t_v, f_scale))`: a camera move on top of the phone's own (what the dolly uses).
- **Pocket cuts no longer punch holes in the artist** (found on the first showcase render). The despill cut
  "pockets", interior patches that match the background, so the world shows through gaps the mask filled. A bald
  head's highlights against a white sky matched the background by colour, were cut, and the blue world showed
  through his forehead.
  - Now a patch lying along the matte's outline (over 25% of its perimeter) is never a pocket.
  - Pockets are cut only for the Apple Vision test masks, which fill gaps (`--pockets auto`). SAM 3.1 follows
    the gaps itself, so there an interior "pocket" is the artist.
  - `--pockets on|off` overrides.
  - The perimeter test had its own bug, also fixed: a blob cropped to its bounding box eroded with the crop
    border as foreground and had no perimeter.
  - 1 test, 3 guards mutation-red.
- The clean-plate pass captures the real shadow only for `--contact real`. The capture cost as much as the
  refine itself, and the first showcase render spent about 12 minutes in it.
- **Worlds lit like the take** (`realone/stage/light.py`). Gauntlet round 7 found that a world lit like the take
  beats one lit against it. With `--stage tracked`, the take's light is measured off the background around the
  artist and named in the world prompt as one sentence:
  - `low` when the background is dark;
  - `hard` when the sky band reads clear blue;
  - `soft` for an overcast grey sky or no sky at all.
  `--no-world-light` turns it off for A/B. The first version mis-read the real overcast park as "direct sun,
  warm" (dark trees against white sky looked like contrast; pink brick and green leaves looked like warm light).
  It was rebuilt on the sky, and the light's colour was dropped (scene colour is not light colour). The real
  plate now reads soft. 2 tests, 3 guards mutation-red.
- **`--from-treatment`**: the director's treatment drives the render in one flag: each section's world, its move,
  the freeze and the receipt-look sections (only those inside the rendered range). 2 tests.
- **Tripod takes work in the stage** (no camera operator needed). A locked-off take has no motion to solve, so
  the stage now treats the camera as level, `--camera-m` (default 1.3 m, a chest-high tripod) above the floor,
  and places the horizon from the artist standing on the floor (feet - (feet - head) x camera / figure). The
  world holds still and the virtual moves (dolly, crash, swap) run on top. A take that reads as locked (under
  3 px of floor drift) goes straight to this path, so a jitter-solved normal is never trusted. End to end on a
  truly locked test clip (the dancer composited onto a frozen background), 3 bars with a dolly zoom pass every
  render check. 1 test.
- **SAM 3.1 masks decode ~100x faster** (`realone/samcodec.py`). Meta's parser validates every mask by decoding
  and re-encoding it in pure Python, and the pipeline then decoded it again: about 456k Python calls per mask. A
  numba port of the same range coder (SAM License permits derivative works) is installed into the parser for the
  parse and used for the decode.
  - On real SAM 3.1 output (a 150-frame take, 629 masks), parsing went from 134 s to 4 s, decoding from 104 to
    2.2 ms per mask, and the whole decode from 435 s to 4.5 s. The masks are byte-identical (same md5).
  - Tests pin it to the reference: the same payload gives the same raster and the same raster the same payload
    on 4 synthetic masks, 40 real masks match, and 154 corrupt or truncated payloads give the same result or the
    same error. 5 guards mutation-red.
  - The reference's clamp of the decoded value is left out, because it cannot change a bit (the value is only
    compared with zero <= total - 1).
  - Without numba, the parser's own codec runs unchanged.
- **The director places the moves.** The treatment schema gains a required `move` per section (none, dolly,
  crash, swap). Muse Wonder's instructions say when each is earned: a dolly once, at the moment the ground shifts;
  a crash for emphasis or confrontation; a swap for restlessness. The semantic rules refuse a second dolly and a
  swap on the frozen hook. The director's section renders pass `--move kind@#index`; `plan`/`direct --stage`
  runs keyed renders through the tracked stage, and a dolly in a dry run (procedural plates, no depth) is skipped
  with a note. `realone render --moves-from TREATMENT` takes the moves for the rendered sections. A swap on a
  section with one world cycles every world in the render. 3 tests, 6 guards mutation-red; suite 240/240.
  `--moves-from` is not yet run end to end (needs a take that covers a whole section).
  - **Tests:** `tests/test_stage.py` has 14 tests. They include a camera solve against a synthetic floor filmed by a
    KNOWN dolly-back camera, which catches the sign twin that reads a dolly back as a dolly in. 14 guards were broken
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
- **Watch page: a test render credits exactly the Meta models that ran.** The banner said "No Meta model ran" on
  the Nov 1 showcase, whose world came from Muse Image and whose treatment from a keyed Muse Wonder run (only the
  masks were local). `models_that_ran()` now reads the evidence: SAM 3.1 only when the masks were not Apple Vision,
  Muse Image only when the world's name matches a `muse_*.json` prompt the run wrote, Muse Wonder and Voice
  Transcribe only from a keyed treatment's ledger (a dry-run or FIXTURE treatment never counts). The banner names
  them; the credits mark them `ran: true` with the ledger's cost. 2 tests; every new guard mutated red, file
  restored byte-identical. The longer banner line wrapped badly at 390 px (page 545 px wide): it now wraps; checked
  live at 1440 and 390 (`docs/screenshots/2026-11-01_watch-live-{desktop,mobile}.png`, no horizontal scroll, both
  videos readyState 4, no console errors).
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
