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

### 2. Native-resolution punch-ins and the phone-clip routes (`realone/native.py`)

```
.venv/bin/python spike/make_test_clip.py footage/perf_4k.mp4 --song footage/test_song.wav --offset 29.0 --secs 9 \
    --size 2160x3840
.venv/bin/python -m realone render --clip footage/perf_4k.mp4 --song footage/test_song.wav --section bars:12-15 \
    --cut-bars 1 --look receipt --mask local --local-plate --sync-near 29 --tag eng_4k_native     # add --no-native for A/B
```

The take is normalized once into two files from ONE ffmpeg filter graph (`split`), so they cannot drift a frame
apart: the 1080x1920 proxy (what the segmenter sees; mask index == output frame index, as before) and the 9:16
centre crop at the source's own resolution (2160x3840 from portrait 4K, 1214x2160 from landscape 4K). Wide shots
use the proxy. Medium and close shots cut their pixels from the native frame (`native_crop`: the crop in native
coordinates, a low-pass only when going down in size, cubic resample). Only the output is 1080x1920. Workers
stream both files frame by frame; nothing holds the clip in memory. Intermediates use a 1 s GOP so a worker that
starts mid-file decodes at most 29 extra frames. `--no-native` forces the 0.1.0 upscale (A/B only).

Routes (`normalize`, printed and kept in the run json under `normalize`):
- `sdr`: an SDR take no wider than about the output: the 0.1.0 path, byte for byte, so cached masks keep their
  keys.
- `sdr+native`: an SDR take with at least 5% more pixels across than the output.
- `hdr-vt`, `hdr-vt+native`: HLG (arib-std-b67) or PQ (smpte2084). This ffmpeg (9.0.1) has no zscale, so the
  tone map goes through VideoToolbox: `-hwaccel videotoolbox -hwaccel_output_format videotoolbox_vld` and
  `scale_vt=...:color_matrix=bt709:color_primaries=bt709:color_transfer=bt709,hwdownload,format=p010le,
  format=yuv420p`, then the rotation by hand, then `fps=30` (iPhone clips are 59.94). Facts from the coordinator's
  measurement on James's own iPhone clip (HEVC 10-bit HLG, 3840x2160, rotation -90): the hwaccel path does not
  autorotate; rotation -90 needs transpose=1 (transpose=2 is upside down); the VT tone map has visibly better
  contrast and skin than the plain software path.
- Found while testing: `media.probe` never asked ffprobe for side data, so the rotation read as 0 and an iPhone
  clip would have come out sideways. `native.probe_video` asks for the display matrix. Also found: the iPhone
  clip's "ambient viewing environment" side data rides into the output and makes `ffprobe -of csv` print `30,`,
  which crashed `count_frames`; it now reads the first field. Checked on 1 s of the real clip (local only, never
  committed, intermediates deleted): route `hdr-vt+native`, 2160x3840 native, 30 frames, BT.709, upright.

Masks stay at proxy resolution. In a close shot (mask blown up 1.6x or more) the mask is edge-refined against
the native pixels: a colour guided filter (He et al., the fast form: coefficients solved at half resolution,
window radius 2 there, eps 1e-4, applied to the full-resolution guide) in a band around the upsampled edge, then
alpha is eroded/feathered in output pixels per native pixel. Measured against the 4K truth on 6 crops of the test
figure: close 0.55% -> 0.38% of the artist's area wrong at the edge; medium 0.61% with or without it, so medium
keeps the plain upsample (a sweep of 36 settings, pre-blurred guides included, found no medium gain).

Measured (2026-10-29, load average 4 to 6):
- Encoded output, read back, Laplacian variance of the artist (receipt look, so the artist's pixels are not graded;
  region = the coloured pixels in both files): close 26.5 -> 860.6 (32x) over 73 frames; medium 28.7 -> 59.7
  (2.1x); wide 42.7 -> 42.7 (unchanged, as it should be).
- Honest split of that number: the synthetic 4K figure carries detail that only exists at native resolution (a
  2 px fabric weave and per-pixel sensor noise), which the proxy averages to flat; that is most of the 32x. With
  weave and noise off (tests), the drawn edges alone measure 39.0 -> 53.0 near the edges (1.36x). Real phone 4K
  has real texture (skin, fabric, hair), so the true gain sits between those, closer to whatever the texture is.
- Sheet `out/TEST_LOCAL_eng_4k_punchin_ab.jpg` (same frames from both encodes, 1:1 crops), read: on the face the
  old path's eyes, mouth and jaw are soft and the native ones crisp, with skin noise visible; on the jacket the old
  path is flat red and the native one keeps the weave and a crisper edge.
- Speed: 220 frames, one look, 75 s native vs 69 s old (native crop 20 ms + refinement 45 ms on close frames).
  `/usr/bin/time -l` peak (python plus its ffmpeg children) 2.06 GB native, 3.05 GB old (the old route decodes the
  4K source inside ffmpeg's scaler); the python process alone peaked at 1.12 GB on the native run (run json
  `peak_rss_mb_python`; it includes librosa's analysis of the whole song).

`spike/make_test_clip.py --size WxH`: the same figure drawn at native density through a small drawing wrapper
(at 1080 it passes the integers straight through: the 1080 clips stay byte-identical, checked by hash), the room
extended to the sides for landscape, the blur in native pixels (a 4K sensor resolves a sharper edge), plus the
weave and sensor noise (`WEAVE_AMP`, `NOISE_SIGMA`).

Tests (`tests/test_native.py`, 8): close shot detail (texture case > 20x) and edges only (> 1.2x near the drawn
edges); the refined mask is closer to the 4K truth; a dot lands on the same output pixel through the native and
proxy paths (portrait 2x, and landscape where x and y scales differ by 0.1%); a seek returns exactly the frame
asked for (B-frames, 1 s GOP; frames 1, 29, 30, 31, 47, 74); a synthetic 10-bit HLG HEVC take at 59.94 fps with
rotation -90 comes out upright (a red block in the coded bottom-left ends top-left), BT.709, yuv420p, 30 frames at
30 fps (skipped without VideoToolbox); a 1440x2560 SDR take takes the native route and proxy frame k matches native
frame k, not its neighbour; a 1080x1920 take keeps the 0.1.0 route.
Mutation M11 (2026-10-29, scratch copy): the compositor ignores the native frame: 2 tests red; restored.

### 3. Full-song render (`--section all`, `realone/units.py`, `realone/worldlist.py`)

```
.venv/bin/python spike/make_test_clip.py footage/full_test.mp4 --song footage/test_song.wav --offset 0 --secs 90
.venv/bin/python -m realone render --clip footage/full_test.mp4 --song footage/test_song.wav --section all \
    --look collage --freeze hook --mask local --local-plate --tag eng_song                  # --workers N to set
# keyed: one world list per section
.venv/bin/python -m realone render --clip take.mov --song master.wav --section all --worlds-json worlds.json
#   worlds.json: {"hook": ["rain-soaked alley, neon"], "hook:2": [{"image": "roof.png"}], "verse": ["..."],
#                 "#0": ["..."], "default": ["..."]}       or repeat --section-world hook="rooftop at dusk"
```

Every render, one section or the whole song, now goes through the same pipeline: plan, then render units in
worker processes, then join. `--section all` plans every section the analysis found, contiguous; each section
boundary is a cut, each section opens wide and draws from its own world list (lookup per section: `#index`,
`role:occurrence`, `role`, `default`, then the 0.1.0 `--world`/`--world-image` flags; one global, deduplicated
list; a section with nothing is refused before any work). With `--local-plate` each role starts the rotation on a
different plate. `--max-seconds` (40) still guards single sections; a whole song is guarded by `--max-minutes`.

Units and workers (`--workers`, default CPU cores minus 2): one unit per shot, longest first (better packing); a
shot longer than 1.6x the per-worker share and over 240 frames is split, never within 2 frames of a freeze release
(its no-jump check needs the frame before from the same worker), and a split piece warms its colour-match EMA on
the 10 frames before it. Each worker builds its own Compositor once (spawn pool), streams its frames from the
proxy (and native) file with a frame-exact seek, reads masks from a memory-mapped .npy, and encodes one raw H.264
segment per look. Everything a frame needs is a function of its index (world clock, crop, grain, boil step)
except the colour-match EMA, which restarts per unit; a test renders 12 frames whole and in 3 units and gets the
same world hash on every frame and the same pixels within encoder noise.

Joining, decided by measurement (each tried on a 514-frame numbered sequence in 5 segments, one of them 1 frame):
- mp4 segments through the concat demuxer: frames drifted after the 1-frame segment (no duration to stamp);
- raw segments WITH B-frames, stamped by the muxer: the first 4 frames came back reordered;
- raw segments, no B-frames (`-bf 0`), byte-concatenated: 514 of 514 in order. Chosen. Then two ffmpeg steps:
  raw -> video-only mp4 stamped at 30 fps with BT.709 written into the SPS (`h264_metadata`; 0.1.0's files said
  "unknown" for transfer and primaries), then that mp4 + the master slice -> AAC, `-shortest`, faststart. One
  step lost the audio entirely on 1080p (the raw stream reaches the muxer without timestamps).
Checks per output (exit 2 on any failure): spec, frame count == plan, every planned cut found in the encoded file
and no other, output audio vs master within 1 frame, the freeze checks. A song with more than 12 shots gets a
song sheet (middle frame of every shot, frozen shots outlined) instead of the per-shot columns.

Measured (2026-10-29, the synthetic full test song: 36 bars, 88.2 s, 2645 frames; collage; Hook Freeze on both
hooks; local masks and plates; `out/TEST_LOCAL_eng_song_all_collage.mp4`):
- **6 min 10 s wall (370 s) with the masks cached, about 7 min 20 s with them computed** (the masks took 70 s for
  2645 frames on 4 processes in the run before). Stages: analysis 3.1 s, sync 1.4 s, normalize 0.4 s, render 338 s
  (7.8 fps overall, 2.0 to 2.4 fps per worker), concat 3.1 s, checks 5.1 s, sheets 16.2 s. 4 workers.
- **Under what load**: never idle. 1-minute load average 11 to 18 through the run, swap 7.2 to 8.7 GB in use, from
  two other agents, a Blender render and headless Chrome, on a 12-core (8 performance + 4 efficiency), 16 GB Mac.
- Checks all passed: 2645/2645 frames, cuts 15/15 on plan and none off it, audio vs master +0.1 ms, both freezes
  (147 frozen frames each identical; moves before (a real check now: the verse and the bridge precede the hooks),
  resumes, no jump, artist moves, control moves). Python 1.55 GB peak, largest worker 0.95 GB.
- **The first attempt used 10 workers (cores minus 2) and was stopped by hand after 8 minutes with no unit
  finished**: the workers sat at 8 to 20% CPU in the "stuck" state, kernel_task at 275% compressing memory, swap
  12.5 of 13.3 GB used. Ten workers on a shared 16 GB Mac is slower than four. Decision: the default is now
  CPU cores minus 2 capped by free memory (vm_stat free + inactive + speculative + purgeable, minus 1.5 GB, over
  0.95 GB per worker, the measured peak); `--workers N` overrides; the run json says why (`workers_why`). It chose
  4 here (5.5 GB reclaimable), and each worker then ran at 70 to 80% CPU.
- **A 3-minute song, extrapolated from these measured rates (not measured)**: 5400 frames at 7.8 fps is 690 s,
  masks at 38 fps 142 s, the rest about 60 s: about 15 minutes under THIS load, i.e. right at the goal, not under
  it with margin. On an idle machine with the memory for 10 workers at the per-worker rate measured here (2.2 fps)
  it would be about 4 minutes of render plus 2.5 of masks, about 7 to 8 minutes, but the machine was not idle once
  today, so that number is an estimate and is not claimed. The SAM path replaces the Vision masks with a decode of
  the API's output (cached) and is unmeasured (no key).
- Song sheet `out/TEST_LOCAL_eng_song_all_collage_contact.jpg`, read: 16 shots in order, every section opens wide,
  the two freezes outlined, the worlds rotate per section (hook 2 opens on the roof because the bridge ended in the
  alley). Freeze sheet 2 (`..._freeze2.jpg`), read: the bridge's alley before, the roof frozen at world clock 1911
  on frames 2060, 2130 and 2200 (the embers in place), released at 1914 and 1942, the changed tile only the figure.
  The acid stage still swallows the white collage border (shots 1, 4, 9, 12, 15): item 5.

Found on the way:
- **A whole-song take was refused as ambiguous.** 0.1.0's tie-break needed the true lag to lose NO 3 s window by
  more than 0.04 to a competitor. On a 90 s take (59 windows), sections that loop every 4 bars fit a 4-bar shift
  almost as well, so 2 to 9 windows lose by noise (worst -0.092) while the true lag wins 20+ windows by 0.28 to
  0.64. A take with 24 windows or more (about 36 s) is now judged on all of them: a clear win (0.12) still
  required, at most a fifth of the windows lost by more than 0.04, none lost by 0.12 or more, and the mean
  advantage at least 2.5 standard errors (windows overlap by half, so n/2 count). Shorter takes keep the 0.1.0
  rule exactly; every 0.1.0 sync test still passes (the identical loop and the 8 s take still refuse). Replayed
  on every sync fixture before choosing: a t-test ALONE would have accepted the loop fixture (t up to 8.7), so the
  clear-win rule stays for every take. Tests: three whole-song takes sync within a frame; each condition of the
  long rule refuses on its own; mutation M13 (t-test dropped) turns that test red.
- Local masks (TEST ONLY) now use `VNGeneratePersonSegmentationRequest` when it finds a person on 2 of 3 probe
  frames (decided once per clip, so it cannot flip mid-take), else foreground-instance (the drawn figure, where
  person segmentation returns empty). The coordinator saw foreground-instance pick the bench and dumbbells in and
  out on James's real clip. Measured on 2 s of it (local only, nothing kept): person segmentation 12.9 to 15.5% of
  the frame, foreground-instance 16.8 to 20.6%. Masks are written straight into a memory-mapped .npy by up to 4
  processes (a full song is about 260 KB a frame); a test pins 2 processes == 1 process, byte for byte.
  `init` on the person request is unavailable through pyobjc; `initWithCompletionHandler_(None)` works.

Tests (`tests/test_song.py` 6, `tests/test_masks_local.py` 1, 2 in `test_sync.py`, 5 in `test_cli.py`): the whole
song plan (section cuts, wide openings, worlds by role, every cut within a frame of a true downbeat); world lists by
key and every refusal; units tile every frame exactly once and never split a release, for 1 to 32 workers; raw
segments with a 1-frame one join frame-exactly with audio and BT.709 tags; a 12-frame real render whole vs in 3
units (same world hash per frame, pixels within encoder noise); the worker default never outgrows free memory;
parallel Vision masks == one process; whole-song takes sync; the long-take rule refuses each of its conditions; the
new flags refuse at the gate in under 10 s. Mutations (scratch copy, restored byte-identical): M12 a unit drops its
last frame (red); M13 the long-take t-test dropped (red).

### 4. Real Receipt as a song mode, and the hard number (`realone/receipt.py`)

```
.venv/bin/python -m realone render --clip footage/full_test.mp4 --song footage/test_song.wav --section bars:26-30 \
    --look collage --receipt-reveal hook:2 --receipt-counter --mask local --local-plate --tag eng_receipt
.venv/bin/python -m realone render ... --section all --look collage --receipt-reveal hook:last --receipt-counter
```

- **The number, on every render**: `out/<tag>_<section>_receipt.json` (`<tag>` as in every other output) holds,
  per output, every frame's filmed share, the whole video's, and each section's, plus the look of every frame (as
  runs) and the reveal spans. Filmed = the artist mask in output space (after the reframe) minus any pocket of the
  source room the despill cut open to show the world, over the 1080x1920 frame; everything else (plate,
  particles, collage paper, the counter) is generated. Decision: the mask, not the eroded and feathered matte,
  because the ask defines filmed as "the artist mask"; the matte differs from it by about 0.3 points wide and 0.6
  in a 1.4x punch-in (the erosion scales with the zoom), which would break the 0.5-point promise. It is the same
  for every look and a pure function of the frame (so it does not depend on the units).
- **`--receipt-reveal SECTION`** (repeatable; the grammar of `--section`, plus `role:last`): that section renders
  in the receipt look in every output. The reveal's first and last bars are forced cuts, so the flip from colour
  to grey always lands on a cut (the cut check would otherwise see the flip as an off-grid cut and fail the
  render). `--look receipt` makes the whole render a receipt section.
- **`--receipt-counter`**: on receipt frames, `FILMED 24.7%  GENERATED 75.3%` in DIN Condensed Bold (macOS
  Supplemental; then DIN Alternate, Helvetica, DejaVu Condensed, Pillow's own), set like a till receipt: label,
  dotted leader, right-aligned figure to one decimal (the two always sum to 100.0), and a thin bar filled to the
  filmed share. Greyscale on purpose: in the receipt look colour belongs to the filmed pixels only. Per frame, so
  it ticks with the artist's movement. Drawn before the grain, so it sits in the picture.
  - Place: chosen once per shot among top left, top right and lower left (inside the 9:16 safe area, clear of a
    phone's caption band), whichever the artist's box covers least over the shot; ties keep top left.
  - Ink: the first render put white type on the acid stage's white rings and the GENERATED row vanished. Mean
    luminance could not see it (the rings average to mid grey). Now the counter takes the ink that fewer pixels
    under it fight (white vs pixels brighter than 0.6, near-black vs darker than 0.4), with hysteresis so a shot
    cannot flicker, and on a mixed ground lays a soft feathered pad of the opposite tone behind the block only as
    strong as that conflict (none on a plain ground: just type).
  - Read on `out/TEST_LOCAL_eng_receipt_counter_shots.jpg` (hook 2 in the receipt look, one crop per shot, read
    back from the encode): wide 14.1%, medium 24.7%, close 34.3%, medium 25.0%, each equal to that frame's json value
    rounded (14.07, 24.69, 34.30, 24.97); the close shot on the rings is legible on its pad; in the last medium shot
    a bright neon tube crosses "75.0%", still readable through the halo but not perfectly clean (a known limit).
    `out/TEST_LOCAL_eng_receipt_counter_1to1.jpg` (bars 26-30, collage with hook 2 revealed): frame 146 collage,
    no counter; frame 147 receipt with the counter at 24.7%, the json's value for that frame.
- Measured: bars 26-30 with the reveal, 294 frames, cut 1/1 on plan (the reveal's), audio -0.3 ms, 54 s on 2
  workers (load about 4); hook 2 in receipt with the counter, 588 frames, cuts 3/3, audio +0.3 ms, filmed 25.2% over
  the hook, 62 s on 4 workers.

Tests (`tests/test_receipt.py`, 9): the counter's number is within 0.5 points of the mask area, wide and in a 1.4x
punch-in, and equals the measured share rounded to 0.1; the counter changes only its own block and only on the
receipt look; every look reports a filmed share; filmed + generated == 100.0; section selection incl. `hook:last`;
the place goes where the artist is not, and a free place beats a slightly covered one; it draws at its place.
Mutation M14 (2026-10-29, scratch copy): the counter under-reports by 10%: 2 tests red; restored byte-identical.
Pillow 13.0.0 is now a direct dependency and pinned.

### 5. Collage legibility (`composite.border_ink`, `render.collage_inks`)

0.1.0's known limit: on a light world (the acid stage's white rings) the near-white paper border vanishes and only
the offset shadow separates the cut-out. Now the border becomes a dark outline (BGR 0.065, 0.065, 0.075, the same
paper grain on it) when the world under it is light, automatically, with no flag.
- What is measured: the band the border actually covers (outside the artist, out to 1.6x the border width, the
  widest jag), BT.709 luma of the world there (plate plus back particles). A share, not a mean: the share of the
  band brighter than 0.72 (where near-white paper at 0.955 no longer reads). Over 0.35 means dark. The rings
  average to mid grey and still swallow half the border, so a mean would miss exactly the case that prompted this.
- Decided once per shot (on its first, middle and last frame, through the shot's own camera), so the border never
  flickers mid-shot; a cut is where it may change. The shot list records `collage_ink` and `border_light_share`.
  The collage despill target follows the ink (the edge decontaminates toward what will sit behind it).
- Measured on the hook (bars 13-20, collage, Hook Freeze): neon alley 0.000 white, sunset roof 0.067 white, acid
  stage close-up 0.469 dark; frames 588/588, cuts 2/2, audio +0.1 ms, the freeze checks all OK.
- Read on `out/TEST_LOCAL_eng_ink_ab.jpg` (frame 514 of the item 1 render vs now, 1:1): before, the white border
  melts into the white ring beside the head; now the dark outline holds the silhouette against it. Honest costs:
  the border fills any gap narrower than itself (it did in white too), so in dark the hair strands merge into one
  dark shape; and where the same band crosses a black ring the dark outline disappears (the skin edge still
  separates there). On a mixed ground no one border colour wins everywhere; the rule picks the one that loses
  less. Contact sheet `out/TEST_LOCAL_eng_ink_hook_collage_contact.jpg`.

Tests (`tests/test_collage_ink.py`, 3): a white world switches to a dark outline (ring luma under 0.15, 0.7 under
the world); a dark world keeps the paper (over 0.85); a world white only where the border sits switches, white only
far from the artist does not. Mutation M15 (scratch copy): the border never switches: 2 tests red; restored.

### 6. SAM prompt union (`--sam-prompt`, the coordinator's optional note)

```
.venv/bin/python -m realone render --clip take.mov --song master.wav --section hook \
    --sam-prompt "the person" --sam-prompt "the microphone" --world "..."
```

The coordinator saw props pop in and out of a cut-out on James's real clip. `--sam-prompt` (repeatable) makes one
SAM 3.1 call per phrase, each cached and each in the spend quote (US$0.20 per 1k frames per uncached phrase),
decodes each on its own (so each phrase keeps its own objects), and ORs the masks per frame into a memory-mapped
union. A phrase that matches nothing is left out and named; nothing matched is refused. Without the flag, the 0.1.0
chain ("the person singing", then "the person") is unchanged. `--sam-prompt` with `--mask local` is refused (the
local test masks take no prompt). UNCONFIRMED like every SAM path here: tested offline only, on text built with
the parser library's own encoder (`tests/test_sam_union.py`: person + microphone land exactly where their boxes say
on every frame, an empty phrase is left out, the quote counts only uncached phrases, no key means no call).
Mutation M16: a "union" that keeps only the first phrase: red. A per-phrase hold of missing frames is the 0.1.0
rule; a prop that leaves the frame for good holds its last mask (a known limit until real output is seen).

### Known limits of 0.2.0

- Everything visual was still judged on a drawn figure with Apple Vision masks, plus 2 s of James's real clip for
  orientation and mask area only. SAM 3.1 and Muse Image remain unproven until the key lands.
- The 3-minute-in-15 goal is met only by extrapolation, and only at the line under today's load (about 15 min);
  an idle-machine run was never possible today.
- Native punch-ins: the close-shot mask refinement helps (0.55% to 0.38% wrong edge pixels) but medium shots use
  the plain upsample; the detail gain on real skin and fabric is unmeasured (the synthetic weave is the stand-in).
- Collage dark outline: fills hair gaps solid and vanishes over black where the ground is mixed.
- Receipt counter: a bright stripe crossing the figures stays readable through the halo but not perfectly clean.
- Long-take sync rule: calibrated on the procedural song (4-bar loops); a real song with long identical repeats
  may still refuse; `--sync-near` remains the way out.

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
`out/contact.jpg`. If the edge fails, the floor is plain Real One in the cut-out collage look.

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
