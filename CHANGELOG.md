# Changelog

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
