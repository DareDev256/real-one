"""Artist masks: Meta SAM 3.1 (keyed, the real path) or Apple Vision (TEST ONLY). Factored from
spike/freeze_test.py; the SAM wire handling is unchanged from the spike and still unproven against a live key."""
import asyncio
import json
import os
import re
import time
from multiprocessing import Pool

import cv2
import numpy as np

from realone.common import H, OUT, W, Refuse
from realone.meta_api import SAM_MIN_FRAMES, SAM_MODEL, SAM_USD_PER_1K_FRAMES
from realone.media import read_frames

SAM_PROMPTS = ["the person singing", "the person"]


def sam_cost(frames):
    return max(frames, SAM_MIN_FRAMES) * SAM_USD_PER_1K_FRAMES / 1000.0


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def call_sam(client, norm, prompt):
    """Upload the normalized clip (Files API, purpose=user_data), then stream sam-3.1 over /v1/responses.
    Returns (concatenated output_text, file_id).
    Shapes: docs /docs/sam/segmenting (upload once + input_video.file_id, metadata.mask_encoding).
    # UNCONFIRMED: that the Python OpenAI SDK passes `metadata` and `input_video` through unchanged;
    # the docs' own Python example uses client.responses.stream(...) with this exact input."""
    with open(norm, "rb") as f:
        up = client.files.create(purpose="user_data", file=f)
    parts = []
    with client.responses.stream(
        model=SAM_MODEL,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_video", "file_id": up.id}]}],
        metadata={"mask_encoding": "one_bit"},
    ) as stream:
        for ev in stream:
            if ev.type == "response.output_text.delta":
                parts.append(ev.delta)
    return "".join(parts), up.id


def sam_text(client, norm, tag, n_frames):
    """Cached: out/masks/<tag>__<prompt>.txt (raw output_text) + .json. A rerun costs nothing.
    Tries 'the person singing', then 'the person'. Returns (text, prompt, billed_usd_this_run)."""
    mdir = OUT / "masks"
    mdir.mkdir(parents=True, exist_ok=True)
    billed = 0.0
    for prompt in SAM_PROMPTS:
        txt, meta = mdir / f"{tag}__{slug(prompt)}.txt", mdir / f"{tag}__{slug(prompt)}.json"
        if not txt.exists():
            if client is None:
                raise Refuse("no cached SAM masks for this clip and no MODEL_API_KEY")
            print(f"SAM 3.1: segmenting '{prompt}' on {n_frames} frames ...")
            t0 = time.time()
            text, file_id = call_sam(client, norm, prompt)
            txt.write_text(text)
            meta.write_text(json.dumps({"prompt": prompt, "file_id": file_id, "frames": n_frames,
                                        "seconds": round(time.time() - t0, 1), "empty": not text.strip()}, indent=2))
            billed += sam_cost(n_frames)
        else:
            print(f"SAM 3.1: cache hit for '{prompt}' ({txt.name}), no API call")
        text = txt.read_text()
        if text.strip():
            return text, prompt, billed
        print(f"SAM 3.1: no match for '{prompt}', trying the fallback prompt")
    raise Refuse("SAM 3.1 matched nothing for any prompt; check the clip (is the artist in frame?)")


async def _parse_async(text):
    import meta_sam_parser as m
    lane = {"item_id": "msg", "output_index": 0, "content_index": 0}

    async def src():
        if text:
            yield {"type": "response.output_text.delta", **lane, "delta": text}
        yield {"type": "response.output_text.done", **lane, "text": text}
        yield {"type": "response.completed"}

    return await m.parse_responses_stream(src(), m.video_segmentation_format()).final_result()


def parse_sam_text(text):
    """-> list of (frame, object_id, left, top, right, bottom, encoding, payload, mask_w, mask_h)
    using Meta's own parser (meta-sam-parser) so the base85/`|>` edge cases are theirs, not ours."""
    if "\n" not in text:
        # UNCONFIRMED: whether the stream's deltas carry the newline between frame lines (the parser needs one).
        # If they do not, restore it at each `|><Nf>` boundary (a record terminator followed by a frame marker).
        text = re.sub(r"(?<=\|>)(<\d+f>)", r"\n\1", text)
    res = asyncio.run(_parse_async(text))
    if res.diagnostics:
        print(f"SAM parser: {len(res.diagnostics)} diagnostics (first: {res.diagnostics[0]})")
    out = []
    for r in res.records:
        if getattr(r, "kind", None) != "mask":
            continue
        b = r.bounds
        out.append((r.frame.frame_index if r.frame else 0, r.object_id, b.left, b.top, b.right, b.bottom,
                    r.mask.encoding, r.mask.payload, r.mask.width, r.mask.height))
    return out


def _decode_job(job):
    enc, payload, w, h = job
    import meta_sam_parser as m
    if enc == "one_bit":
        try:  # same decoder, minus the canonical re-encode check (which doubles the time)
            from meta_sam_parser import _mask_codec as mc
            raster = mc._decode_raster(payload, w, h)
        except Exception:
            raster = m.decode_mask_to_raster(m.SegmentationMask(enc, payload, w, h))
    else:
        raster = m.decode_mask_to_raster(m.SegmentationMask(enc, payload, w, h))
    return np.packbits(np.frombuffer(raster, np.uint8).reshape(h, w) > 0)


def place_mask(canvas, raster_bool, left, top, right, bottom):
    """Masks are box-local (docs: reading-segmentation): raster covers [left,right) x [top,bottom).
    Resize if the raster and the box disagree by a pixel."""
    x0, y0 = int(round(left)), int(round(top))
    bw, bh = int(round(right - left)), int(round(bottom - top))
    h, w = raster_bool.shape
    if (w, h) != (bw, bh) and bw > 0 and bh > 0:
        raster_bool = cv2.resize(raster_bool.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST) > 0
        h, w = raster_bool.shape
    x1, y1 = min(W, x0 + w), min(H, y0 + h)
    xs, ys = max(0, x0), max(0, y0)
    if x1 <= xs or y1 <= ys:
        return
    canvas[ys:y1, xs:x1] |= raster_bool[ys - y0:y1 - y0, xs - x0:x1 - x0]


def hold_missing(packed, have):
    """Frames with no mask hold the nearest earlier mask (or the next one at the start)."""
    n = len(packed)
    missing = int((~have).sum())
    if missing:
        last = None
        for i in range(n):
            if have[i]:
                last = i
            elif last is not None:
                packed[i] = packed[last]
        nxt = None
        for i in range(n - 1, -1, -1):
            if have[i]:
                nxt = i
            elif packed[i].sum() == 0 and nxt is not None:
                packed[i] = packed[nxt]
    return missing


def sam_masks(text, n_frames):
    """SAM output text -> packed bool masks (n_frames, W*H/8), plus stats.
    # UNCONFIRMED: only exercised on text built with the parser library's own one_bit encoder
    # (tests/test_sam_decode.py); no live sam-3.1 response has been decoded yet (no key)."""
    recs = parse_sam_text(text)
    if not recs:
        raise Refuse("SAM output parsed to zero masks")
    jobs = [(r[6], r[7], r[8], r[9]) for r in recs]
    t0 = time.time()
    with Pool(max(1, min(8, (os.cpu_count() or 4) - 4))) as pool:
        rasters = pool.map(_decode_job, jobs, chunksize=4)
    print(f"SAM masks decoded: {len(recs)} masks in {time.time() - t0:.0f}s")
    area, decoded = {}, []
    for r, packed in zip(recs, rasters):
        h, w = r[9], r[8]
        ras = np.unpackbits(packed)[:w * h].reshape(h, w).astype(bool)
        decoded.append(ras)
        area[r[1]] = area.get(r[1], 0) + int(ras.sum())
    biggest = max(area.values())
    keep = {k for k, v in area.items() if v >= 0.30 * biggest}
    if len(keep) > 1:
        print(f"SAM: {len(area)} objects matched; keeping ids {sorted(keep)} (>=30% of the biggest)")
    frames = {}
    for r, ras in zip(recs, decoded):
        if r[1] not in keep:
            continue
        f = r[0]
        if f not in frames:
            frames[f] = np.zeros((H, W), bool)
        place_mask(frames[f], ras, r[2], r[3], r[4], r[5])
    packed = np.zeros((n_frames, W * H // 8), np.uint8)
    have = np.zeros(n_frames, bool)
    for f, m in frames.items():
        if 0 <= f < n_frames:
            packed[f] = np.packbits(m)
            have[f] = True
    missing = hold_missing(packed, have)
    if missing:
        print(f"WARNING: SAM emitted no mask on {missing}/{n_frames} frames; held the neighbouring mask")
    return packed, {"missing_frames": missing}


def sam_masks_cached(client, norm, tag, n_frames):
    text, prompt, billed = sam_text(client, norm, tag, n_frames)
    mcache = OUT / "masks" / f"{tag}__{slug(prompt)}.masks.npy"  # decoded masks: reruns skip the decode
    if mcache.exists():
        return np.load(mcache), f"SAM 3.1 '{prompt}'", billed
    packed, _ = sam_masks(text, n_frames)
    np.save(mcache, packed)
    return packed, f"SAM 3.1 '{prompt}'", billed


def vision_masks(norm, n_frames, cache):
    """TEST ONLY. Apple Vision foreground-instance mask (pyobjc, no model download).
    VNGeneratePersonSegmentationRequest returned an empty mask on the drawn test figure (it only recognises
    real people), so subject lifting is used. Says nothing about SAM's edge quality."""
    if cache.exists():
        print(f"local mask: cache hit {cache.name}")
        return np.load(cache)
    import objc
    import Quartz
    import Vision
    from Foundation import NSData
    packed = np.zeros((n_frames, W * H // 8), np.uint8)
    have = np.zeros(n_frames, bool)
    t0 = time.time()
    for i, frame in enumerate(read_frames(norm)):
        if i >= n_frames:
            break
        with objc.autorelease_pool():
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
            req = Vision.VNGenerateForegroundInstanceMaskRequest.alloc().initWithCompletionHandler_(None)
            okp, err = handler.performRequests_error_([req], None)
            res = req.results() if okp else None
            if not res:
                continue
            obs = res[0]
            pb, err = obs.generateScaledMaskForImageForInstances_fromRequestHandler_error_(
                obs.allInstances(), handler, None)
            if pb is None:
                continue
            Quartz.CVPixelBufferLockBaseAddress(pb, 1)
            w = Quartz.CVPixelBufferGetWidth(pb)
            h = Quartz.CVPixelBufferGetHeight(pb)
            bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
            base = Quartz.CVPixelBufferGetBaseAddress(pb)
            arr = np.frombuffer(base.as_buffer(bpr * h), np.float32).reshape(h, bpr // 4)[:, :w].copy()
            Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
        if (w, h) != (W, H):
            arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
        packed[i] = np.packbits(arr > 0.5)
        have[i] = packed[i].any()
    empty = int((~have).sum())
    print(f"local mask: Apple Vision foreground-instance, {n_frames} frames in {time.time() - t0:.0f}s, {empty} empty")
    if empty > n_frames * 0.5:
        raise Refuse("Apple Vision found no subject on most frames")
    hold_missing(packed, have)
    np.save(cache, packed)
    return packed
