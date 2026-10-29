# Implementation notes

## 0.2.0: the engine (Hook Freeze, native punch-ins, full song, Real Receipt, collage legibility)

Built 2026-10-29 by the engine agent while James was at work; every decision below was made without him and is
logged here so it can be reversed. The director (`realone/director/`, `docs/director.md`) is a separate agent's
work and calls `python -m realone render` by subprocess; every 0.1.0 flag still works unchanged and every new
flag is optional.

### 1. Hook Freeze in the renderer (`realone/freeze.py`)

```
.venv/bin/python -m realone render --clip footage/perf_test.mp4 --song footage/test_song.wav \
    --section hook --freeze hook --look collage --mask local --local-plate --tag eng_freeze
```

`--freeze [SECTION]` (repeatable) and `--freeze-bars N` (default 2). The freeze starts on the section's first
downbeat. Bare `--freeze` freezes the rendered section (this is the form the director's `--help` probe picks up:
it only passes a flag that takes no value). A bare role (`--freeze hook`) means every section with that role
inside the rendered bars; `hook:2`, `#3`, `bars:A-B` name one, with the same grammar as `--section`. A freeze
longer than its section, or running past the rendered bars, is refused (exit 2), not clipped.

What stops and what does not:
- The world clock `wf` (frames) advances 1 per frame, except inside a freeze, where every frame shows the clock
  of the freeze's first frame. The first frame after the freeze is one step on, so it resumes from where it
  stopped. Plate push-in (now measured on `wf`, so it holds and resumes), the sideways drift, the particles and
  the parallax are all pure functions of `wf` (`worlds.world_layer`). Without a freeze `wf[i] = i` and
  `world_layer` reproduces the 0.1.0 inline maths byte for byte (checked when it was factored out).
- The virtual camera (reframe crop) holds its first-frame framing for the whole freeze, then eases back onto
  the live track over 0.5 s (smoothstep, stopping at the next cut). The artist kept moving, so the live track
  has moved on; an ease is a camera move, a snap would be a jump.
- The artist's frames, the grain and the collage boil run on the real frame index: a frozen frame with live
  grain reads as a stopped world, with dead grain it reads as a still image (the spike's finding).

Decisions:
- **A freeze always starts on a cut, and the cuts inside it and on its release are dropped.** With the default
  2-bar cuts and a 2-bar freeze on the hook, the release would otherwise land on a cut, which swaps the world at
  the very moment it should come back to life, and "resumes from where it stopped" would mean nothing. So the
  hook's first shot runs 4 bars (2 frozen, 2 live) and the next cut is bar 5 of the hook. A section boundary
  always stays a cut; if a freeze fills its whole section the resume check is reported "n/a: the release is a
  section cut".
- **The framing pattern restarts at every section**, so each section, and therefore every freeze, opens on the
  wide shot: the freeze shows the world it stops.

Checks after every render with a freeze (port of the spike's, on the rendered world layer; any failure exits 2
and the run json keeps the report under `freeze`):
- frozen: the md5 of the world layer (plate + particles) is identical on every frame of the freeze;
- moving: every frame outside a freeze whose world clock advanced has a different hash from the frame before
  (same shot); this covers "moves before" and "after";
- resumes: the 3 frames after the release each differ from the one before;
- no jump: the clock step at the release is 1, and the plate's change at the release (mean abs difference of a
  135x240 plate thumbnail, particles excluded) is at most 3x the median change over the next 2 s of the same shot;
- the artist moves (the mask differs between the first and last frozen frame), and a control: the world at the
  unfrozen clock one frame later differs from the frozen world (so it was the freeze that held it, not a world
  that never moves).
The world cache that skips redrawing a frozen world is keyed on exactly `world_layer`'s inputs (world, world
time, push fraction, crop), so a bug that fed the real clock into the particles would miss the cache, draw a
moving world, and fail the check.

Measured (2026-10-29, `--section hook --freeze hook`, collage, local plates, load average 17 to 21 on 12 cores
from two other agents and a Blender render): freeze on frames 0-146 (4.90 s, bars 13-14), world identical on
all 147 frozen frames, resumes, no jump, the artist moves, the control world moves; cuts on plan 2/2 (bars 17 and
19), audio vs master +0.1 ms; 588 frames in 246 s (2.4 fps, single process).
Sheet `out/TEST_LOCAL_eng_freeze_hook_collage_freeze1.jpg`, read: the rain streaks sit in the same places in
frozen frames 3, 73 and 143 while the pose changes; the "changed" tile (frame 3 vs 143, grain ignored) is black
everywhere except the figure; the released frames (world clock 3 and 31) show the rain moving again.

Tests (`tests/test_freeze.py`, 7): clock holds and steps by one; the world layer is byte-identical in a freeze,
moving outside, and its release is not a jump; a release that snaps to real time is caught as a jump; the checks
fail when the world moves inside a freeze, stands still outside one, or the artist does not move; the plan
starts the freeze on a cut and drops the cuts inside and on the release; `hook` resolves to every hook and
refuses what does not fit; the camera holds then eases with no step over a fifth of the gap.
Mutation checks (2026-10-29, run on a scratch copy of the repo so the director's live renders never saw a
mutated file; each restored byte-identical by sha256): M9 the freeze hold removed from `world_clock` (2 tests
red); M10 the clock snapping back to real time after the release (2 tests red).

## 0.1.0: the floor (`realone/`, `python -m realone render`)

One section of the song, end to end: a phone take plus the master track in, a finished 9:16 clip out, cuts on
downbeats, master audio in sync. The spike's working parts (normalize, SAM path, Apple Vision path, Muse Image
call, particles, alpha refinement, LAB colour match, light wrap, grain, encode, faststart check) were moved into
the package, not rewritten. `spike/freeze_test.py` is left as the frozen 0.0.2 record.

### Run it

```
.venv/bin/pip install -r requirements.txt

# keyed (the demo path; needs MODEL_API_KEY in the environment, never in the repo)
.venv/bin/python -m realone render --clip footage/take1.mov --song footage/master.wav --section hook \
    --look collage --world "rain-soaked Toronto alley at night, neon" --world "rooftop at dusk" --world "empty stage"

# keyless test (never the demo; every output is prefixed TEST_LOCAL_)
.venv/bin/python spike/make_test_song.py footage/test_song.wav
.venv/bin/python spike/make_test_clip.py footage/perf_test.mp4 --song footage/test_song.wav --offset 26.317 --secs 26
.venv/bin/python -m realone render --clip footage/perf_test.mp4 --song footage/test_song.wav \
    --section hook --look collage --mask local --local-plate
.venv/bin/python -m realone render --clip footage/perf_test.mp4 --song footage/test_song.wav \
    --section hook --look all --mask local --local-plate      # all 3 looks + out/TEST_LOCAL_looks.jpg
.venv/bin/python -m pytest -q tests

# stages on their own
.venv/bin/python -m realone analyze --song footage/test_song.wav --local
.venv/bin/python -m realone sync --clip footage/perf_test.mp4 --song footage/test_song.wav
```

### Pipeline

| Stage | Module | What it does | Writes |
| --- | --- | --- | --- |
| audio | `audio.py` | beat grid, downbeats, sections, hook flag, from the master only | `out/<tag>_timeline.json` |
| sync | `sync.py` | phone mic vs master, onset cross-correlation; refuses when unsure | `out/<tag>_<section>_sync.json` |
| plan | `plan.py` | cuts on downbeats every N bars; world swap + reframe per cut | `out/<tag>_<section>_shots.json` |
| mask | `masks.py` | SAM 3.1 (keyed) or Apple Vision (TEST ONLY), cached | `out/masks/` |
| world | `worlds.py` | Muse Image per prompt (keyed, cached by prompt hash) or 3 procedural plates (TEST ONLY) | `out/worlds/` |
| composite | `composite.py` | reframe, despill, colour match, look, grain | |
| encode | `encode.py` | H.264 yuv420p BT.709 faststart, master audio for the section | `out/<tag>_<section>_<look>.mp4` |
| check | `render.py` | spec, cuts found in the encoded file vs plan, output audio vs master | `out/<tag>_<section>_run.json` |
| sheets | `sheets.py` | contact sheet per look, looks comparison, read back from the encoded files | `*_contact.jpg`, `TEST_LOCAL_looks.jpg` |

`<tag>` is `floor` by default (`--tag`), prefixed `TEST_LOCAL_` whenever `--mask local`.

**Audio.** librosa beat tracker on a mel onset envelope (hop 256 at 22.05 kHz). A constant-tempo grid is fitted to
the beats (beat indices counted interval by interval, so a skipped beat cannot shift the index); when the
residual is under 15 ms the fitted grid replaces the raw beats (`grid: "constant"`), otherwise the raw beats stay
(`"tracked"`). The centred onset envelope peaks 21 ms after a sharp attack (measured on the test song's exact
onsets; `ONSET_LAG_S`, pinned by a test). Downbeat = the bar phase with the most low-band onset energy plus chord
change; 4 beats per bar unless 3 is 1.5x stronger, or `--beats-per-bar`. Sections: per-bar features (chroma,
MFCC, loudness, onset density, bar medians so a crash on the one does not define the bar), robust-scaled,
weighted by how slowly each feature moves (the chord cycle flips every bar and says nothing about sections),
projected on the principal components holding 80% of the variance. Every local novelty peak is a candidate
boundary; segments are clustered into labels, same-label neighbours merged, then each boundary may move 1 bar
toward whichever label the transition bars resemble. Hook = the repeated label with the most energy. On the test
song (98 BPM) and a second structure (87 BPM, intro 2 / verse 8 / hook 8 / bridge 4 / hook 8, also a test) the
hook boundaries land on the exact bar. A third structure checked by hand (124 BPM, ending on a hook plus a
4-bar hook tag) found its first hook exactly but ended the last one 2 bars early and called those 2 bars an
outro: the end of a song is the weakest spot.

**Sync.** song_time = clip_time + offset. Six onset envelopes at hop 128 (5.8 ms): percussive low/mid/high and
harmonic 150-600 / 600-1500 / 1500-4000 Hz (HPSS). Loop-based beats repeat the drums exactly, so the note onsets
(vocal, lead, chords) get their own bands. Normalised cross-correlation over every lag (the take may hang off
either end of the song by up to 40%), parabolic refinement. Refuses (exit 3):
- best r under 0.35 ("does not match": wrong song, garbage, a phone that did not hear the playback);
- silent mic, or under 4 s of audio;
- ambiguous: when another lag is within 0.08 r, the best lag must beat EVERY competitor within 0.12 r by at
  least 0.12 in some 3 s window and lose none by more than 0.04. A take wholly inside a copy-pasted loop has no
  such window and refuses. `--sync-near <song seconds>` restricts the search to +-6 s for a short take.
The container's audio/video start skew is applied after measuring (padding the gap with digital silence would
plant a fake onset). Measured on the synthetic take: 20 of 20 noise seeds correct (artist singing live at 0, +3,
+6 dB over the playback, room noise at 15 dB SNR), worst error 4 ms, zero wrong locks; end to end through the
real mp4 (AAC) -2.7 ms. Before the tie-break existed, two short takes locked onto the wrong repeat and were
refused by the margin rule, not rendered.

**Plan.** Cuts every `--cut-bars` (1, 2, 4 or 8; default 2) on detected downbeats, each rounded to the nearest
output frame (error at most half a frame, 16.7 ms). One take, so a cut is a world swap plus a reframe. Framing
cycles wide (1.0x) / medium (1.4x) / close (1.9x) / medium; medium and close centre on the mask's bounding box
(medium on 40% down it, close on 17%), smoothed with a 0.6 s Gaussian inside each shot so the camera breathes
rather than tracks, clamped inside the frame. The world zooms a quarter as much as the artist, toward the same
point, so the artist feels closer than the world (cheap parallax), plus a 4% push-in per shot.

**Despill (all looks).** The local test showed the source room bleeding into a concave pocket (between chin,
hand and collar): Apple Vision filled the hole, so the pocket was inside the mask, 50 to 80 px from the true edge.
Model: an edge pixel P mixes the artist's local colour F and the room's local colour B, P = a F + (1 - a) B. F and
B are smooth local estimates (normalised convolution at quarter resolution; F is re-estimated without the pocket
pixels). Where the mix is real (F and B differ, P lies near the F-B line) the room share is swapped for the
target: P' = P + (1 - a)(T - B). T is the blurred world (clean), the grey world (receipt), or paper white
(collage). This runs on the 8 px edge band everywhere, and deeper (up to 120 px) only in small enclosed
bg-coloured blobs (24 to 9000 px), whose alpha is also cut so the world shows through; the cut then grows 2 px
into the blob's anti-aliased rim, gated by each rim pixel's own room share. A large bg-coloured region (a blue
shirt on a blue wall) is not a pocket and is left alone (tested). Pixels deeper than 120 px inside the mask are
never touched (tested: max change 0.0). All distances and areas scale with the reframe zoom (the first full
render missed the pocket in every medium and close shot until they did). Cost 100 to 190 ms a frame, estimated
once per frame for all looks.

**Looks** (`--look clean|collage|receipt|all`):
- `clean`: the spike's look (erode 2, feather 1.4, light wrap 0.6, LAB match 35%, grain).
- `collage`: a paper cut-out. The mask dilated by 12 px with a jagged edge (smooth noise, clamped to 6-19 px),
  near-white paper with fibre and grain texture, a hard offset shadow (20, 28 px, 74% dark, 1.2 px soft). The
  border noise, the paper texture and the shadow jitter re-randomise at 12 fps (step = frame * 12 // 30, integer
  maths), while the artist and the world stay at 30 fps. A crisper scissor edge on the artist (erode 1, feather
  0.8); pockets cut by the despill show paper.
- `receipt`: every generated pixel (plate, particles) goes to BT.709 luma; the filmed artist keeps full colour
  (no grade at all, no light wrap, grey particles kept off the artist: the phone's pixels, decontaminated at
  the edge only).

**Checks after every render** (exit non-zero if any fails): H.264 / 1080x1920 / yuv420p / faststart / audio /
duration; hard cuts detected in the ENCODED video by frame differencing must sit within 1 frame of every planned
cut, and no other hard cut may appear; the output's audio is cross-correlated against the master and must sit at the section start within 1 frame.
A render that passes `--max-minutes` (default 20) is stopped. Contact sheets and the looks sheet are read back
from the encoded files.

### Test material (keyless, never the demo)

- `spike/make_test_song.py`: procedural 98 BPM song with a 0.40 s lead-in, intro 4 / verse 8 / hook 8 / verse 8
  / hook 8 bars; kick, snare, hats, bass, pad; the hook adds claps, open hats, a lead, a wide saw chord, +3.5 dB.
  A formant-synth "vocal" (not a person) sings a different line in each verse and the same hook line twice, with
  ad-libs in hook 2. Truth in `<out>.truth.json`. The vocal was added after the first sync calibration: with
  drums and chords only, the two halves of the song were rhythm-identical and every take was (correctly)
  ambiguous. Real songs differ between repeats mainly in the vocals; the fixture now does too.
- `spike/make_test_clip.py --song`: the drawn figure bobs on the song's beat (lowest point on each beat, so
  lip-sync is visible by eye), and the audio is a phone-mic simulation of the song from a known offset
  (150 Hz to 7 kHz, a short room, the artist's separate vocal take at 0 dB, noise at 20 dB SNR).

### Tests (`pytest -q tests`, 32 tests)

audio (BPM within 1, downbeats within 1 frame of the true bar lines, hook boundaries within 1 bar on two
structures, the onset-lag constant does real work), sync (offset within 1 frame for clean, loud-artist, pre-roll
and container-skew takes; white noise and another song refuse as no-match, silence as silent; an identical loop refuses as
ambiguous; `--sync-near` resolves a short take), plan (every rendered cut within 1 frame of a detected AND a true
downbeat for 1/2/4-bar cuts; every cut changes world and framing; crops stay in frame and follow the artist),
looks (receipt saturation, collage ring near-white, boil distinct exactly 12 times per second and changing only
on 12 fps steps, despill pocket and clothing, despill through the compositor at a 1.9x close-up), SAM decode offline, CLI (keyed paths refuse without a key in under
10 s with nothing called; a key is never echoed).

Mutation checks run on 2026-10-29 (each guard broken on purpose, its test went red, source restored
byte-identical): beat snapping, sync no-match guard, sync ambiguity guard, receipt greyscale, 12 fps boil, despill
zoom scaling, the `--mask sam` key gate, the onset-lag correction. M6 exists because the first full render showed
the pocket surviving every medium and close shot; the test was added with the fix and confirmed red without it.

### Still UNCONFIRMED until the key lands

Everything in the 0.0.2 list below still holds; the floor added no new API shapes. Specifically: SAM 3.1 masks on
real footage (edge quality, whether holes between arm and body are kept open, how many frames come back empty),
the SDK pass-through of `metadata` and `input_video`, the newline handling in streamed deltas, and Muse Image's
`size="1080x1920"`. The keyed path is wired (`--mask sam`, `--world`), quotes the spend in US$ and CA$ before any
call, refuses over `--max-usd` (default US$1.00), and refuses without `MODEL_API_KEY` before any work. One hook of
8 bars at 98 BPM is 588 frames: SAM about US$0.12, three worlds US$0.03.

### Known limits of the floor

- Everything visual was judged on a drawn figure with a Vision mask. The despill and the collage edge have not
  met real hair, motion blur or a busy room.
- Thresholds (sync r and margins, despill pocket sizes, section novelty) were calibrated on procedural audio and
  one synthetic take.
- Close-ups upscale the 1080x1920 normalized take 1.9x; a 4K source would need the normalize step to keep more
  resolution for the crops.
- Section labels other than the hook are cosmetic (verse 2 with an extra layer can come out as its own label).
- Speed: about 2.9 fps for one look and 1.2 to 1.7 fps for all three (single process, 1080x1920; the low end
  with other sessions loading the Mac). One 8-bar section is 3.5 minutes in one look and 6 to 11 in all three;
  a full 3-minute song in one look would be about 30 minutes, past the 20-minute stop, so the full-song
  milestone needs frame-parallel workers.
- The collage's white paper border loses its edge on a light world (the acid stage's white rings); only the
  offset shadow separates the cut-out there.

## 0.0.2: Hook Freeze spike (`spike/freeze_test.py`)

Question the spike answers: does a SAM 3.1 cut-out of a phone clip, on a generated world, survive a phone screen,
or does it read as cheap green screen? Verdict is James's, on a phone, from `out/freeze_test.mp4` and
`out/contact.jpg`. If the edge fails, the plan's floor is plain Real One and the honest pivot is Cartographer
(round6-fable.md).

### Run it

```
export MODEL_API_KEY=...        # lives outside the repo
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python spike/freeze_test.py --clip footage/take1.mov --world "rain-soaked Toronto alley at night, neon signs" --final
```

No-key plumbing test (never the demo; every output is prefixed `TEST_LOCAL_`):

```
.venv/bin/python spike/make_test_clip.py footage/synthetic_test.mp4     # drawn figure, not footage of anyone
.venv/bin/python spike/freeze_test.py --clip footage/synthetic_test.mp4 --mask local --local-plate
.venv/bin/python spike/test_sam_decode.py                                # offline test of the SAM output path
```

### Pipeline

1. Normalize: ffmpeg, 30 fps, scale + centre-crop to 1080x1920, first 10 s, BT.709 tags. Audio extracted to AAC.
   The normalized clip is what SAM sees, so mask frame indices equal output frame indices.
2. Mask: `--mask sam` (default) uploads the normalized clip, calls `sam-3.1` with "the person singing" (fallback
   "the person"). Raw output text cached to `out/masks/<clip hash>__<prompt>.txt`, decoded masks to `.masks.npy`.
   A rerun makes zero API calls.
3. World: Muse Image from `--world`, cached to `out/world.png` with the prompt beside it (a changed prompt
   regenerates; the same prompt does not). Or `--world-image`.
4. Composite per frame: erode 2 px, feather (sigma 1.4), light wrap from the blurred world onto the inner edge,
   LAB mean/std colour transfer to the world at 35 %, one shared grain over the whole frame.
5. World motion: slow push-in (10 % over 10 s) plus a code-drawn particle layer (dust default; rain, embers).
   Everything is a pure function of the world clock. For the freeze the clock stops (and resumes from where it
   stopped, so there is no jump). Grain keeps animating during the freeze on purpose: a frozen frame with live grain
   reads as a stopped world, a frozen frame with dead grain reads as a still image. ~18 % of particles are drawn
   over the artist as soft near-field bokeh.
6. Encode: ffmpeg, H.264 slow crf 17, yuv420p, BT.709, faststart, original audio.

The script proves the freeze mechanically too: it hashes the world layer (plate + particles) of every frame and
prints whether all freeze frames are identical, whether every pre-freeze frame differs, and whether it resumes.

### Meta Model API: what was confirmed from the docs (dev.meta.ai/docs, read 10-28, signed-in Chrome, read only)

CONFIRMED (documented request and response shapes):
- Base URL `https://api.meta.ai/v1`, OpenAI SDK compatible, Bearer key (`MODEL_API_KEY`).
- Files API: `POST /v1/files`, `purpose="user_data"`, synchronous, returns `id`; MP4 faststart is remuxed
  losslessly if needed (ours already is).
- SAM 3.1: `POST /v1/responses`, `model="sam-3.1"`, one user message with an `input_text` noun phrase and an
  `input_video` part (`video_url` or `file_id`); `stream: true`; output is special-token text in the output_text
  lane, one line per frame, `<Nf>` marker, records `id<|box;...|><|mask;x=0;y=0;data=H,W,<enc>payload|>`;
  `metadata.mask_encoding = "one_bit"` for small payloads; `metadata.include_confidence`; H.264/H.265/MPEG-4 in
  MP4/MOV; max ~15,000 frames; US$0.20 per 1k frames, min 50 frames per request. No polling: it is a stream.
- Masks are box-local (raster covers the box), base85, decoded by the official Python package `meta-sam-parser`
  (`decode_mask_to_raster`). The script uses its parser for the text and its decoder for the pixels.
- Muse Image: `POST /v1/images/generations`, `model="muse-image-1.0"`, `size="WxH"` sets the aspect ratio only,
  `response_format` default `b64_json`, `output_format` webp/png/jpeg, Meta extensions via `extra_body`
  (`reasoning_strength`, `tool_enablement`). US$0.01 per image.

UNCONFIRMED (written from the closest documented example, marked `# UNCONFIRMED` in the code):
- Whether the streamed `output_text` deltas carry the newline between frame lines. The parser needs one; the script
  restores it at each `|><Nf>` boundary if the text has none. Tested offline both ways.
- `size="1080x1920"` for 9:16. The docs say any "WxH" sets the aspect; the exact accepted set is not listed.
- That the Python OpenAI SDK (3.41.0) forwards `metadata` and an `input_video` part with a `file_id` unchanged on
  `responses.stream`. The docs' own Python example uses exactly this call, so this is low risk.
- The real mask output on real footage. Only a round trip through the library's own one_bit encoder has been
  exercised (`spike/test_sam_decode.py`). No API call has been made; there is no key yet.
- Lossless (`~`) masks are decoded by the library to a binary raster (coverage >= 129), so SAM masks are binary
  either way; the feather is what gives the edge its softness.

### Local test mode (TEST ONLY)

`--mask local` uses Apple Vision through pyobjc (already a light install, no model download).
`VNGeneratePersonSegmentationRequest` was tried first and returned an empty mask on the drawn test figure (it only
recognises real people), so the script uses `VNGenerateForegroundInstanceMaskRequest` (subject lifting). mediapipe
and rembg were not needed. This cannot say anything about SAM's edge quality. It only proves the plumbing, the
freeze, the encode and the contact sheet.

### Known limits

- HDR phone clips: this ffmpeg build has no `zscale`, so HLG/PQ is not tone-mapped; the script warns. Film with
  HDR Video off for the test.
- Masks decode in pure Python (about 5 us per pixel): the first SAM run spends roughly 2 to 4 minutes decoding on
  10 cores; the decode is cached.
- No contact shadow or re-lighting of the artist beyond colour match and light wrap. That is a next step only if
  the edge test passes.
