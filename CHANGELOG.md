# Changelog

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
