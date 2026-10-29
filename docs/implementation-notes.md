# Implementation notes

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
