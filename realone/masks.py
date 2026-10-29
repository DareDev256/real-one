"""Artist masks: Meta SAM 3.1 (keyed, the real path) or Apple Vision (TEST ONLY). Factored from
spike/freeze_test.py; the SAM wire handling is unchanged from the spike and still unproven against a live key."""
import asyncio
import json
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path

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


def sam_text_one(client, norm, tag, n_frames, prompt, mdir=None):
    """One prompt, cached: <mdir>/<tag>__<prompt>.txt (raw output_text) + .json. A rerun costs nothing.
    -> (text, billed_usd_this_run)."""
    mdir = Path(mdir or OUT / "masks")
    mdir.mkdir(parents=True, exist_ok=True)
    txt, meta = mdir / f"{tag}__{slug(prompt)}.txt", mdir / f"{tag}__{slug(prompt)}.json"
    if txt.exists():
        print(f"SAM 3.1: cache hit for '{prompt}' ({txt.name}), no API call")
        return txt.read_text(), 0.0
    if client is None:
        raise Refuse(f"no cached SAM output for '{prompt}' on this clip and no MODEL_API_KEY")
    print(f"SAM 3.1: segmenting '{prompt}' on {n_frames} frames ...")
    t0 = time.time()
    text, file_id = call_sam(client, norm, prompt)
    txt.write_text(text)
    meta.write_text(json.dumps({"prompt": prompt, "file_id": file_id, "frames": n_frames,
                                "seconds": round(time.time() - t0, 1), "empty": not text.strip()}, indent=2))
    return text, sam_cost(n_frames)


def sam_text(client, norm, tag, n_frames):
    """The default chain: 'the person singing', then 'the person'. Returns (text, prompt, billed_usd_this_run)."""
    billed = 0.0
    for prompt in SAM_PROMPTS:
        text, b = sam_text_one(client, norm, tag, n_frames, prompt)
        billed += b
        if text.strip():
            return text, prompt, billed
        print(f"SAM 3.1: no match for '{prompt}', trying the fallback prompt")
    raise Refuse("SAM 3.1 matched nothing for any prompt; check the clip (is the artist in frame?)")


def sam_uncached(tag, prompts, mdir=None):
    """The prompts that would cost an API call (no cached output yet)."""
    mdir = Path(mdir or OUT / "masks")
    return [p for p in prompts if not (mdir / f"{tag}__{slug(p)}.txt").exists()]


def _decoded(text, n_frames, path):
    if not path.exists():
        tmp = path.with_suffix(".tmp.npy")
        packed, _ = sam_masks(text, n_frames, out=tmp)
        packed.flush()
        del packed
        tmp.rename(path)
    return path


def sam_masks_union(client, norm, tag, n_frames, prompts, mdir=None):
    """--sam-prompt, repeated (0.2.0): one SAM 3.1 call per prompt ("the person", "the microphone"), each cached
    and each billed, decoded on its own (so each keeps its own objects), then OR-ed per frame, so a held prop stays
    in the cut-out. A prompt that matches nothing is left out and named. -> (path, engine, billed).
    # UNCONFIRMED: like every SAM path here, exercised only on text built with the parser library's own encoder."""
    mdir = Path(mdir or OUT / "masks")
    billed, paths, used = 0.0, [], []
    for p in prompts:
        text, b = sam_text_one(client, norm, tag, n_frames, p, mdir)
        billed += b
        if not text.strip():
            print(f"SAM 3.1: '{p}' matched nothing; left out of the union")
            continue
        paths.append(_decoded(text, n_frames, mdir / f"{tag}__{slug(p)}.masks.npy"))
        used.append(p)
    if not paths:
        raise Refuse(f"SAM 3.1 matched nothing for any --sam-prompt {prompts}")
    if len(paths) == 1:
        return paths[0], f"SAM 3.1 '{used[0]}'", billed
    out = mdir / f"{tag}__union__{'+'.join(slug(p) for p in used)}.masks.npy"
    if not out.exists():
        tmp = out.with_suffix(".tmp.npy")
        u = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n_frames, W * H // 8))
        arrays = [np.load(p, mmap_mode="r") for p in paths]
        for i in range(n_frames):
            u[i] = np.bitwise_or.reduce([a[i] for a in arrays])
        u.flush()
        del u
        tmp.rename(out)
    return out, f"SAM 3.1 union of {used}", billed


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


def sam_masks(text, n_frames, out=None):
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
    packed = (np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(n_frames, W * H // 8))
              if out is not None else np.zeros((n_frames, W * H // 8), np.uint8))  # a memmap for a full song
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
    """-> (path of the packed .npy, engine, billed). Decoded masks are cached: reruns skip the decode."""
    text, prompt, billed = sam_text(client, norm, tag, n_frames)
    return _decoded(text, n_frames, OUT / "masks" / f"{tag}__{slug(prompt)}.masks.npy"), f"SAM 3.1 '{prompt}'", billed


def _vision_one(frame, Vision, Quartz, NSData, objc, person):
    """One BGR frame -> (H, W) bool mask or None. person=True: VNGeneratePersonSegmentationRequest (real people,
    stable from frame to frame); False: VNGenerateForegroundInstanceMaskRequest (subject lifting: works on the
    drawn test figure, but on real footage it changes which objects it includes from frame to frame)."""
    with objc.autorelease_pool():
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        if person:
            req = Vision.VNGeneratePersonSegmentationRequest.alloc().initWithCompletionHandler_(None)
            req.setQualityLevel_(0)  # accurate
            okp, err = handler.performRequests_error_([req], None)
            res = req.results() if okp else None
            if not res:
                return None
            pb = res[0].pixelBuffer()
        else:
            req = Vision.VNGenerateForegroundInstanceMaskRequest.alloc().initWithCompletionHandler_(None)
            okp, err = handler.performRequests_error_([req], None)
            res = req.results() if okp else None
            if not res:
                return None
            obs = res[0]
            pb, err = obs.generateScaledMaskForImageForInstances_fromRequestHandler_error_(
                obs.allInstances(), handler, None)
        if pb is None:
            return None
        Quartz.CVPixelBufferLockBaseAddress(pb, 1)
        w = Quartz.CVPixelBufferGetWidth(pb)
        h = Quartz.CVPixelBufferGetHeight(pb)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
        base = Quartz.CVPixelBufferGetBaseAddress(pb)
        if Quartz.CVPixelBufferGetPixelFormatType(pb) == Quartz.kCVPixelFormatType_OneComponent8:
            arr = np.frombuffer(base.as_buffer(bpr * h), np.uint8).reshape(h, bpr)[:, :w].astype(np.float32) / 255
        else:
            arr = np.frombuffer(base.as_buffer(bpr * h), np.float32).reshape(h, bpr // 4)[:, :w].copy()
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
    if (w, h) != (W, H):
        arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
    return arr > 0.5


def _vision_range(job):
    """Worker: frames [f0, f1) of the proxy -> rows of the shared .npy memmap. Returns which rows got a mask."""
    norm, f0, f1, tmp, person = job
    import objc
    import Quartz
    import Vision
    from Foundation import NSData
    packed = np.load(tmp, mmap_mode="r+")
    have = np.zeros(f1 - f0, bool)
    for j, frame in enumerate(read_frames(norm, start=f0)):
        if f0 + j >= f1:
            break
        m = _vision_one(frame, Vision, Quartz, NSData, objc, person)
        if m is not None and m.any():
            packed[f0 + j] = np.packbits(m)
            have[j] = True
    packed.flush()
    return f0, have


def pick_vision_request(norm, n_frames):
    """Person segmentation when it finds a person (real footage), else foreground-instance (the drawn test figure,
    on which person segmentation returns empty). Decided once per clip on 3 frames, so it cannot flip mid-take."""
    import objc
    import Quartz
    import Vision
    from Foundation import NSData
    probes = sorted({int(n_frames * q) for q in (0.1, 0.5, 0.9)})
    hits = 0
    for f in probes:
        frame = next(read_frames(norm, start=f))
        m = _vision_one(frame, Vision, Quartz, NSData, objc, True)
        hits += int(m is not None and m.mean() > 0.01)
    return hits >= 2


def vision_masks(norm, n_frames, cache, workers=1):
    """TEST ONLY. Apple Vision masks (pyobjc, no model download) -> path of a packed (n, W*H/8) .npy, written as a
    memmap (a full song is ~260 KB a frame; nothing holds it all in RAM) by up to `workers` processes.
    VNGeneratePersonSegmentationRequest for real people; VNGenerateForegroundInstanceMaskRequest for the drawn
    test figure (person segmentation returns empty on it). Says nothing about SAM's edge quality."""
    cache = Path(cache)
    person_cache = cache.with_name(cache.name.replace("local_", "local_person_", 1))
    for c in (cache, person_cache):
        if c.exists():
            print(f"local mask: cache hit {c.name}")
            return c, ("VNGeneratePersonSegmentationRequest" if c == person_cache else
                       "VNGenerateForegroundInstanceMaskRequest")
    person = pick_vision_request(norm, n_frames)
    out = person_cache if person else cache
    name = "VNGeneratePersonSegmentationRequest" if person else "VNGenerateForegroundInstanceMaskRequest"
    tmp = out.with_suffix(".tmp.npy")
    np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n_frames, W * H // 8)).flush()
    k = max(1, min(workers, 4, n_frames // 30 or 1))
    edges = [round(n_frames * j / k) for j in range(k + 1)]
    jobs = [(str(norm), edges[j], edges[j + 1], str(tmp), person) for j in range(k)]
    t0 = time.time()
    if k == 1:
        results = [_vision_range(jobs[0])]
    else:
        with Pool(k) as pool:
            results = pool.map(_vision_range, jobs)
    have = np.zeros(n_frames, bool)
    for f0, h in results:
        have[f0:f0 + len(h)] = h
    empty = int((~have).sum())
    print(f"local mask: Apple Vision {name}, {n_frames} frames in {time.time() - t0:.0f}s on {k} process(es), "
          f"{empty} empty")
    if empty > n_frames * 0.5:
        tmp.unlink(missing_ok=True)
        raise Refuse("Apple Vision found no subject on most frames")
    packed = np.load(tmp, mmap_mode="r+")
    hold_missing(packed, have)
    packed.flush()
    del packed
    tmp.rename(out)
    return out, name
