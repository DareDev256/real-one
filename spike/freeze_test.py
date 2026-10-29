#!/usr/bin/env python3
"""Real One, spike 1: the Hook Freeze green-screen test.

Takes ONE phone clip of the artist, cuts the artist out (Meta SAM 3.1), puts
them on a generated world (Muse Image) that slowly pushes in and carries a
light particle layer, and for 3 s in the middle freezes the world (camera and
particles stop dead) while the artist keeps moving. The question it answers:
does the composite survive a phone screen, or does it read as cheap green
screen?

Demo run (needs MODEL_API_KEY in the environment):

    python spike/freeze_test.py --clip footage/take1.mov --world "rain-soaked Toronto alley at night, neon signs"

No-key plumbing test (never used for the demo; outputs get a TEST_LOCAL_ prefix):

    python spike/freeze_test.py --clip footage/synthetic_test.mp4 --mask local --local-plate

Outputs (in out/):  freeze_test.mp4   1080x1920 H.264 yuv420p faststart + the clip's own audio
                    contact.jpg       6 frames with timestamps, 2 inside the freeze
                    edge_crop.jpg     1:1 crop of the artist's edge inside the freeze (extra)
"""
import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1080, 1920, 30
MAX_SECS = 10.0
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "out"

BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
IMAGE_MODEL = "muse-image-1.0"
SAM_USD_PER_1K_FRAMES = 0.20
SAM_MIN_FRAMES = 50
IMAGE_USD = 0.01
SAM_PROMPTS = ["the person singing", "the person"]
WORLD_SIZE = "1080x1920"  # sets the aspect ratio only (docs: image-generation, `size`)
WORLD_WRAP = (
    ". Vertical 9:16 empty background plate for a music video: no people, no text, no logos, "
    "cinematic lighting, camera at chest height, open floor space in the lower centre, no motion blur."
)
VERMILION = (43, 65, 232)  # BGR


def die(msg, code=2):
    print(f"freeze_test: {msg}", file=sys.stderr)
    sys.exit(code)


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


# --------------------------------------------------------------------------- normalize

def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,pix_fmt,color_transfer,duration,r_frame_rate",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def normalize(clip, seconds, work):
    """ffmpeg to 30 fps 1080x1920 (scale + centre crop), first `seconds` s.
    Returns (norm_video, audio_or_None, n_frames_expected)."""
    work.mkdir(parents=True, exist_ok=True)
    info = probe(clip)
    streams = info.get("streams", [])
    vid = next((s for s in streams if s.get("codec_type") == "video"), None)
    if vid is None:
        die(f"no video stream in {clip}")
    if vid.get("color_transfer") in ("arib-std-b67", "smpte2084"):
        print("WARNING: this clip is HDR (HLG/PQ). ffmpeg here has no zscale, so it is NOT tone-mapped; "
              "colours will look flat. Film with HDR Video off, or convert to SDR first.")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    dur = min(float(info["format"]["duration"]), seconds)
    norm = work / "norm.mp4"
    audio = work / "audio.m4a"
    if not norm.exists():
        sh(["ffmpeg", "-y", "-v", "error", "-t", f"{seconds}", "-i", str(clip), "-an",
            "-vf", f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,"
                   "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart", str(norm)])
    if has_audio and not audio.exists():
        sh(["ffmpeg", "-y", "-v", "error", "-t", f"{seconds}", "-i", str(clip), "-vn",
            "-c:a", "aac", "-b:a", "192k", str(audio)])
    return norm, (audio if has_audio else None), int(round(dur * FPS))


def read_frames(path):
    """Yield BGR uint8 (H, W, 3) frames from a 1080x1920 video via an ffmpeg raw pipe."""
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE)
    size = W * H * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


# --------------------------------------------------------------------------- SAM 3.1 (Meta Model API)

def make_client(key):
    from openai import OpenAI
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=1800.0)


def call_sam(client, norm, prompt):
    """Upload the normalized clip (Files API, purpose=user_data), then stream sam-3.1 over /v1/responses.
    Returns (concatenated output_text, file_id).
    Shapes: docs /docs/sam/segmenting (upload once + input_video.file_id, metadata.mask_encoding).
    # UNCONFIRMED: only that the Python OpenAI SDK passes `metadata` and `input_video` through unchanged;
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


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def sam_text(client, norm, tag, n_frames):
    """Cached: out/masks/<tag>__<prompt>.txt (raw output_text) + .json. A rerun costs nothing.
    Tries 'the person singing', then 'the person'. Returns (text, prompt, billed_usd_this_run, cached)."""
    mdir = OUT / "masks"
    mdir.mkdir(parents=True, exist_ok=True)
    billed = 0.0
    for prompt in SAM_PROMPTS:
        txt, meta = mdir / f"{tag}__{slug(prompt)}.txt", mdir / f"{tag}__{slug(prompt)}.json"
        cached = txt.exists()
        if not cached:
            if client is None:
                die("no cached SAM masks for this clip and no MODEL_API_KEY")
            print(f"SAM 3.1: segmenting '{prompt}' on {n_frames} frames ...")
            t0 = time.time()
            text, file_id = call_sam(client, norm, prompt)
            txt.write_text(text)
            meta.write_text(json.dumps({"prompt": prompt, "file_id": file_id, "frames": n_frames,
                                        "seconds": round(time.time() - t0, 1),
                                        "empty": not text.strip()}, indent=2))
            billed += sam_cost(n_frames)
        else:
            print(f"SAM 3.1: cache hit for '{prompt}' ({txt.name}), no API call")
        text = txt.read_text()
        if text.strip():
            return text, prompt, billed, cached
        print(f"SAM 3.1: no match for '{prompt}', trying the fallback prompt")
    die("SAM 3.1 matched nothing for any prompt; check the clip (is the artist in frame?)")


def sam_cost(frames):
    return max(frames, SAM_MIN_FRAMES) * SAM_USD_PER_1K_FRAMES / 1000.0


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
    using Meta's own parser (meta-sam-parser) so the base85/`|>` edge cases are theirs, not mine."""
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


def sam_masks(text, n_frames):
    """SAM output text -> packed bool masks (n_frames, W*H/8), plus stats."""
    recs = parse_sam_text(text)
    if not recs:
        die("SAM output parsed to zero masks")
    jobs = [(r[6], r[7], r[8], r[9]) for r in recs]
    t0 = time.time()
    with Pool(max(1, min(10, (os.cpu_count() or 4) - 2))) as pool:
        rasters = pool.map(_decode_job, jobs, chunksize=4)
    print(f"SAM masks decoded: {len(recs)} masks in {time.time() - t0:.0f}s")
    area = {}
    decoded = []
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
    missing = int((~have).sum())
    if missing:
        print(f"WARNING: SAM emitted no mask on {missing}/{n_frames} frames; holding the neighbouring mask")
        last = None
        for i in range(n_frames):
            if have[i]:
                last = i
            elif last is not None:
                packed[i] = packed[last]
        nxt = None
        for i in range(n_frames - 1, -1, -1):
            if have[i]:
                nxt = i
            elif packed[i].sum() == 0 and nxt is not None:
                packed[i] = packed[nxt]
    return packed, {"missing_frames": missing}


# --------------------------------------------------------------------------- local test segmenter (Apple Vision)

def vision_masks(norm, n_frames, cache):
    """TEST ONLY. Apple Vision foreground-instance mask (pyobjc, no model download).
    VNGeneratePersonSegmentationRequest was tried first and returned an empty mask on the drawn test figure
    (it only recognises real people), so it is not used here."""
    if cache.exists():
        print(f"local mask: cache hit {cache.name}")
        return np.load(cache)
    import objc
    import Quartz
    import Vision
    from Foundation import NSData
    packed = np.zeros((n_frames, W * H // 8), np.uint8)
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
    empty = int((packed.sum(axis=1) == 0).sum())
    print(f"local mask: Apple Vision foreground-instance, {n_frames} frames in {time.time() - t0:.0f}s, {empty} empty")
    if empty > n_frames * 0.5:
        die("Apple Vision found no subject on most frames")
    np.save(cache, packed)
    return packed


# --------------------------------------------------------------------------- world

def gen_world(client, prompt, path, wrap=True):
    """Muse Image 9:16 via POST /v1/images/generations (docs: image-generation).
    # UNCONFIRMED: `size="1080x1920"` (docs say size sets the aspect ratio only and accepts any "WxH");
    # `extra_body.tool_enablement` per the docs' extra_body example."""
    meta = path.with_suffix(".json")
    full = prompt + (WORLD_WRAP if wrap else "")
    if path.exists() and meta.exists() and json.loads(meta.read_text()).get("prompt") == full:
        print(f"world: cache hit {path.name} (same prompt), no API call")
        return 0.0
    print("Muse Image: generating the world plate ...")
    resp = client.images.generate(
        model=IMAGE_MODEL, prompt=full, n=1, size=WORLD_SIZE, output_format="png",
        extra_body={"reasoning_strength": "high",
                    "tool_enablement": {"enable_web_search": False, "enable_shell": False}})
    b64 = resp.data[0].b64_json
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        die("Muse Image returned something that is not a decodable image")
    cv2.imwrite(str(path), img)
    meta.write_text(json.dumps({"prompt": full, "model": IMAGE_MODEL, "size_requested": WORLD_SIZE,
                                "size_returned": [img.shape[1], img.shape[0]]}, indent=2))
    print(f"world: saved {path.name} ({img.shape[1]}x{img.shape[0]})")
    return IMAGE_USD


def local_plate(frame, mask_bool, path):
    """TEST ONLY: a blurred, colour-graded, person-inpainted frame of the clip (warm amber dusk)."""
    s = cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask_bool.astype(np.uint8) * 255, (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)
    m = cv2.dilate(m, np.ones((15, 15), np.uint8))
    s = cv2.inpaint(s, m, 5, cv2.INPAINT_TELEA)
    s = cv2.GaussianBlur(s, (0, 0), 7).astype(np.float32) / 255
    s = np.clip(s * np.array([0.55, 0.85, 1.25], np.float32), 0, 1) ** 0.85  # BGR gain: amber
    yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
    vig = 1 - 0.55 * (((xx / (W / 4) - 0.5) ** 2 + (yy / (H / 4) - 0.5) ** 2) * 2.2)
    s = np.clip(s * vig[..., None] * 1.3, 0, 1)
    plate = cv2.resize((s * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(path), plate)


def load_plate(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        die(f"cannot read world image {path}")
    return img


def plate_transform(plate, zoom, dx, dy):
    ph, pw = plate.shape[:2]
    s = max(W / pw, H / ph) * zoom  # cover-fit, then the push-in
    M = np.float32([[s, 0, W / 2 - s * pw / 2 + dx], [0, s, H / 2 - s * ph / 2 + dy]])
    return cv2.warpAffine(plate, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


class Particles:
    """Code-drawn particles, positions a pure function of the WORLD clock, so a frozen clock is a frozen layer."""

    def __init__(self, kind, seed=7):
        self.kind = kind
        r = np.random.default_rng(seed)
        n = {"dust": 150, "rain": 240, "embers": 110, "none": 0}[kind]
        self.n = n
        self.x0 = r.uniform(0, W, n)
        self.y0 = r.uniform(0, H, n)
        self.depth = r.uniform(0, 1, n)  # 0 far .. 1 near
        self.phase = r.uniform(0, 6.28, n)
        self.front = self.depth > 0.82  # ~18% drawn over the artist

    def layers(self, wt):
        back = np.zeros((H, W, 3), np.float32)
        front = np.zeros((H, W, 3), np.float32)
        if self.n == 0:
            return back, front
        d, ph = self.depth, self.phase
        for i in range(self.n):
            tgt = front if self.front[i] else back
            z = d[i]
            if self.kind == "dust":
                x = (self.x0[i] + (6 + 20 * z) * wt + 18 * np.sin(wt * 0.5 + ph[i])) % W
                y = (self.y0[i] - (10 + 30 * z) * wt) % H
                rad = 1.2 + 4.5 * z
                a = (0.22 + 0.5 * z) * (0.7 + 0.3 * np.sin(wt * 1.7 + ph[i]))
                col = (a * 0.80, a * 0.92, a * 1.0)
                cv2.circle(tgt, (int(x), int(y)), int(round(rad)), col, -1, cv2.LINE_AA)
            elif self.kind == "rain":
                spd = 1300 + 900 * z
                x = (self.x0[i] - 0.18 * spd * wt) % W
                y = (self.y0[i] + spd * wt) % (H + 200) - 100
                ln = 40 + 70 * z
                a = 0.18 + 0.3 * z
                cv2.line(tgt, (int(x), int(y)), (int(x - 0.18 * ln), int(y - ln)),
                         (a * 0.92, a * 0.86, a * 0.80), 1 + int(z > 0.6), cv2.LINE_AA)
            else:  # embers
                x = (self.x0[i] + 25 * np.sin(wt * 0.8 + ph[i]) + 8 * wt) % W
                y = (self.y0[i] - (60 + 170 * z) * wt) % H
                a = (0.35 + 0.65 * z) * (0.6 + 0.4 * np.sin(wt * 6 + ph[i]))
                cv2.circle(tgt, (int(x), int(y)), 1 + int(2.6 * z), (a * 0.16, a * 0.55, a * 1.0), -1, cv2.LINE_AA)
        sig = 1.3
        back = cv2.GaussianBlur(back, (0, 0), sig)
        front = cv2.GaussianBlur(front, (0, 0), sig * 2.4) * 1.6  # near = soft bokeh
        if self.kind == "embers":
            back += cv2.GaussianBlur(back, (0, 0), 9) * 1.5
        return back, front


# --------------------------------------------------------------------------- composite

def refine_alpha(mask_bool, erode_px, feather):
    m = mask_bool.astype(np.uint8) * 255
    if erode_px > 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)))
    a = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), feather)
    return a * a * (3 - 2 * a)


def lab_stats(bgr_f, weight=None):
    lab = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    if weight is not None:
        lab = lab[weight.reshape(-1)]
        if len(lab) < 200:
            return None
    return lab.mean(axis=0), lab.std(axis=0) + 1e-3


class Compositor:
    def __init__(self, plate, args):
        self.plate = plate
        self.a = args
        self.parts = Particles(args.particles)
        small = cv2.resize(plate, (270, 480), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        self.t_mu, self.t_sd = lab_stats(small)
        self.ema = None
        self._wcache = None

    def world(self, wt):
        if self._wcache is not None and self._wcache[0] == wt:  # frozen clock: reuse, it is the same image
            return self._wcache[1], self._wcache[2]
        z = 1.0 + self.a.push * (wt / MAX_SECS)
        dx, dy = 16 * np.sin(wt * 0.55), -10 * wt / MAX_SECS * 3
        base = plate_transform(self.plate, z, dx, dy).astype(np.float32)
        back, front = self.parts.layers(wt)
        res = (np.clip(base + back * 255, 0, 255), front * 255)
        self._wcache = (wt, res[0], res[1])
        return res

    def match_color(self, fg, alpha):
        k = self.a.match
        if k <= 0:
            return fg
        small = cv2.resize(fg, (270, 480), interpolation=cv2.INTER_AREA)
        sa = cv2.resize(alpha, (270, 480), interpolation=cv2.INTER_AREA) > 0.5
        st = lab_stats(small, sa)
        if st is not None:
            mu, sd = st
            self.ema = (mu, sd) if self.ema is None else (0.85 * self.ema[0] + 0.15 * mu, 0.85 * self.ema[1] + 0.15 * sd)
        if self.ema is None:
            return fg
        mu, sd = self.ema
        gain = 1 + k * (np.clip(self.t_sd / sd, 0.6, 1.4) - 1)
        shift = k * (self.t_mu - mu)
        lab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB)
        lab = (lab - mu) * gain + mu + shift
        return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR), 0, 1)

    def frame(self, src_bgr, mask_bool, wt, idx):
        a = self.a
        alpha = refine_alpha(mask_bool, a.erode, a.feather)
        world, front = self.world(wt)
        world_key = hashlib.md5(world.tobytes() + front.tobytes()).hexdigest()
        fg = src_bgr.astype(np.float32) / 255
        fg = self.match_color(fg, alpha)
        # light wrap: blurred world bleeds onto the artist's inner edge
        wb = cv2.resize(cv2.GaussianBlur(cv2.resize(world, (270, 480), interpolation=cv2.INTER_AREA),
                                         (0, 0), 4.5), (W, H), interpolation=cv2.INTER_LINEAR) / 255
        near = cv2.GaussianBlur(1 - alpha, (0, 0), 7)
        w = (np.clip(near * 2, 0, 1) * alpha * a.wrap)[..., None]
        fg = fg * (1 - w) + wb * w + wb * w * 0.25
        out = fg * alpha[..., None] * 255 + world * (1 - alpha[..., None])
        out = out + front * (1 - 0.35 * alpha[..., None])
        # one shared film grain over the whole frame (animates even while the world is frozen)
        g = np.random.default_rng(1000 + idx).standard_normal((H, W)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 0.7) * 1.9
        lum = out.mean(axis=2) / 255
        g *= a.grain * (0.55 + 2.0 * lum * (1 - lum))
        out = np.clip(out + g[..., None], 0, 255).astype(np.uint8)
        return out, world_key, alpha


# --------------------------------------------------------------------------- encode + reports

def start_encoder(out, audio, final=False):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-i", str(audio)]
    cmd += ["-map", "0:v"] + (["-map", "1:a"] if audio else [])
    cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264",
            "-preset", "slow" if final else "veryfast", "-crf", "17" if final else "20", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-movflags", "+faststart", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def faststart_ok(path):
    """moov atom before mdat?"""
    with open(path, "rb") as f:
        pos, seen = 0, []
        size = os.path.getsize(path)
        while pos < size:
            f.seek(pos)
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            sz, typ = struct.unpack(">I4s", hdr)
            if sz == 1:
                sz = struct.unpack(">Q", f.read(8))[0]
            seen.append(typ.decode("latin1"))
            if sz == 0:
                break
            pos += sz
    return "moov" in seen and "mdat" in seen and seen.index("moov") < seen.index("mdat")


def verify_output(path):
    info = probe(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio = any(s["codec_type"] == "audio" for s in info["streams"])
    rep = {"codec": v["codec_name"], "size": f"{v['width']}x{v['height']}", "pix_fmt": v["pix_fmt"],
           "duration_s": round(float(info["format"]["duration"]), 2), "audio": audio,
           "faststart": faststart_ok(path)}
    good = (rep["codec"] == "h264" and rep["size"] == f"{W}x{H}" and rep["pix_fmt"] == "yuv420p"
            and rep["faststart"] and rep["duration_s"] <= MAX_SECS + 0.1)
    return rep, good


def tile(img, label, freeze):
    t = cv2.resize(img, (360, 640), interpolation=cv2.INTER_AREA)
    bar = np.zeros((30, 360, 3), np.uint8)
    cv2.putText(bar, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, VERMILION if freeze else (230, 230, 230), 1, cv2.LINE_AA)
    if freeze:
        cv2.rectangle(t, (0, 0), (359, 639), VERMILION, 4)
    return np.vstack([t, bar])


def contact_sheet(video, times, fs, fe, path):
    idxs = [min(int(round(t * FPS)), 10**6) for t in times]
    want = {i: None for i in idxs}
    for i, fr in enumerate(read_frames(video)):
        if i in want:
            want[i] = fr.copy()
        if i >= max(idxs):
            break
    tiles, frames = [], {}
    for t, i in zip(times, idxs):
        fr = want[i]
        if fr is None:
            fr = np.zeros((H, W, 3), np.uint8)
        frames[t] = fr
        frz = fs <= t < fe
        tiles.append(tile(fr, f"t={t:5.2f}s" + ("  FREEZE" if frz else ""), frz))
    gut = 8
    rows = []
    for r in range(2):
        row = [np.full((tiles[0].shape[0], gut, 3), 16, np.uint8)]
        for c in range(3):
            row += [tiles[r * 3 + c], np.full((tiles[0].shape[0], gut, 3), 16, np.uint8)]
        rows.append(np.hstack(row))
    sep = np.full((gut, rows[0].shape[1], 3), 16, np.uint8)
    sheet = np.vstack([sep, rows[0], sep, rows[1], sep])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return frames


def edge_crop(frame, bbox, path):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) // 2, y0 + int((y1 - y0) * 0.18)  # head + shoulders: where edges are hardest
    s = 270
    xs, ys = int(np.clip(cx - s, 0, W - 2 * s)), int(np.clip(cy - s, 0, H - 2 * s))
    crop = frame[ys:ys + 2 * s, xs:xs + 2 * s]
    cv2.imwrite(str(path), cv2.resize(crop, (1080, 1080), interpolation=cv2.INTER_NEAREST), [cv2.IMWRITE_JPEG_QUALITY, 94])


# --------------------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Real One Hook Freeze green-screen test")
    p.add_argument("--clip", required=True, help="the artist's own phone clip")
    p.add_argument("--world", help="prompt for the Muse Image world (9:16), cached to out/world.png")
    p.add_argument("--world-image", help="use this image as the world instead of generating one")
    p.add_argument("--mask", choices=["sam", "local"], default="sam",
                   help="sam = Meta SAM 3.1 (default); local = Apple Vision, TEST ONLY, outputs get TEST_LOCAL_")
    p.add_argument("--local-plate", action="store_true", help="local mode: build the world from a graded frame of the clip")
    p.add_argument("--seconds", "--max-seconds", dest="seconds", type=float, default=MAX_SECS, help="clip length used (max 10)")
    p.add_argument("--final", action="store_true", help="slow preset, crf 17 (default is veryfast, crf 20)")
    p.add_argument("--freeze-len", type=float, default=3.0)
    p.add_argument("--freeze-at", type=float, help="freeze start in s (default: centred)")
    p.add_argument("--particles", choices=["dust", "rain", "embers", "none"], default="dust")
    p.add_argument("--push", type=float, default=0.10, help="push-in over 10 s of world clock")
    p.add_argument("--erode", type=int, default=2, help="mask erosion px")
    p.add_argument("--feather", type=float, default=1.4, help="mask feather sigma px")
    p.add_argument("--wrap", type=float, default=0.6, help="light wrap strength")
    p.add_argument("--match", type=float, default=0.35, help="LAB colour match strength")
    p.add_argument("--grain", type=float, default=5.0, help="film grain strength")
    p.add_argument("--no-world-wrap", action="store_true", help="send --world to Muse Image without the plate instructions")
    return p.parse_args(argv)


def main():
    a = parse_args()
    clip = Path(a.clip)
    if not clip.exists():
        die(f"clip not found: {clip}")
    local = a.mask == "local"
    key = os.environ.get("MODEL_API_KEY")
    has_plate = bool(a.world_image) or a.local_plate
    if not (a.world or has_plate):
        die("give --world '<prompt>' or --world-image <path> (or --local-plate with --mask local)")
    if a.local_plate and not local:
        die("--local-plate is test-only and needs --mask local")
    if not key and not (local and has_plate):
        die("MODEL_API_KEY is not set. Without a key this only runs as: --mask local plus --world-image or --local-plate "
            "(test only, TEST_LOCAL_ outputs).")
    if a.world_image and not Path(a.world_image).exists():
        die(f"world image not found: {a.world_image}")
    secs = min(a.seconds, MAX_SECS)
    pre = "TEST_LOCAL_" if local else ""
    OUT.mkdir(exist_ok=True)
    print(f"mode: {'LOCAL TEST (never the demo)' if local else 'DEMO (SAM 3.1 + Muse Image)'}; "
          f"MODEL_API_KEY {'present' if key else 'absent'}")

    work = OUT / "work" / f"{file_md5(clip)[:12]}_{secs:g}s"
    norm, audio, n_frames = normalize(clip, secs, work)
    tag = file_md5(norm)[:12] + f"_{n_frames}f"
    dur = n_frames / FPS
    if dur < 4:
        die(f"clip is {dur:.1f}s; the freeze needs at least 4 s of footage")
    flen = min(a.freeze_len, dur - 1)
    fs = a.freeze_at if a.freeze_at is not None else (dur - flen) / 2
    fe = fs + flen
    print(f"clip {dur:.1f}s ({n_frames} frames), freeze {fs:.2f}s to {fe:.2f}s")

    client = make_client(key) if key else None
    cost = {"sam_billed": 0.0, "img_billed": 0.0}

    # --- mask
    if local:
        (OUT / "masks").mkdir(exist_ok=True)
        masks = vision_masks(norm, n_frames, OUT / "masks" / f"local_{tag}.npy")
        engine = "Apple Vision (VNGenerateForegroundInstanceMaskRequest)"
    else:
        text, used_prompt, cost["sam_billed"], _ = sam_text(client, norm, tag, n_frames)
        mcache = OUT / "masks" / f"{tag}__{slug(used_prompt)}.masks.npy"  # decoded masks: reruns skip the 2-min decode
        if mcache.exists():
            masks = np.load(mcache)
        else:
            masks, _stats = sam_masks(text, n_frames)
            np.save(mcache, masks)
        engine = f"SAM 3.1 '{used_prompt}'"

    # --- world
    if a.world_image:
        wpath = Path(a.world_image)
    elif a.local_plate:
        wpath = OUT / f"{pre}world.png"
        mid = n_frames // 2
        for i, fr in enumerate(read_frames(norm)):
            if i == mid:
                local_plate(fr, np.unpackbits(masks[mid])[:W * H].reshape(H, W) > 0, wpath)
                break
        print(f"world: local plate {wpath.name} (test only)")
    else:
        wpath = OUT / f"{pre}world.png"
        cost["img_billed"] = gen_world(client, a.world, wpath, wrap=not a.no_world_wrap)
    plate = load_plate(wpath)

    # --- composite + encode
    comp = Compositor(plate, a)
    out_mp4 = OUT / f"{pre}freeze_test.mp4"
    enc = start_encoder(out_mp4, audio, a.final)
    keys, bboxes, areas = [], [], []
    t0 = time.time()
    for i, fr in enumerate(read_frames(norm)):
        if i >= n_frames:
            break
        t = i / FPS
        wt = t if t < fs else (fs if t < fe else t - flen)  # the world clock stops for the freeze
        m = np.unpackbits(masks[i])[:W * H].reshape(H, W) > 0
        out, wkey, alpha = comp.frame(fr, m, wt, i)
        enc.stdin.write(out.tobytes())
        keys.append(wkey)
        ys, xs = np.nonzero(m[::4, ::4])
        bboxes.append((int(xs.min()) * 4, int(ys.min()) * 4, int(xs.max()) * 4, int(ys.max()) * 4) if len(xs) else (0, 0, W, H))
        areas.append(int(m.sum()))
        if i % 60 == 0:
            print(f"  frame {i}/{n_frames}  ({(i + 1) / max(time.time() - t0, 1e-6):.1f} fps)")
    enc.stdin.close()
    if enc.wait() != 0:
        die("ffmpeg encode failed", 1)

    # --- checks that prove the freeze mechanically
    fi = [i for i in range(len(keys)) if fs <= i / FPS < fe]
    frozen_same = len({keys[i] for i in fi}) == 1
    moving_before = len({keys[i] for i in range(0, fi[0])}) == fi[0]
    # the clock resumes from where it stopped, so the first post-freeze frame still equals the frozen one
    resumes = any(k != keys[fi[-1]] for k in keys[fi[-1] + 1:fi[-1] + 4]) if fi[-1] + 1 < len(keys) else True
    artist_moves = abs(areas[fi[-1]] - areas[fi[0]]) > 0 or bboxes[fi[-1]] != bboxes[fi[0]]

    rep, good = verify_output(out_mp4)
    times = [0.6, max(0.8, fs - 0.6), fs + 0.6, fe - 0.6, fe + 0.6, max(fe + 0.8, dur - 0.5)]
    times = [round(min(t, dur - 0.1), 2) for t in times]
    frames = contact_sheet(out_mp4, times, fs, fe, OUT / f"{pre}contact.jpg")
    mid_t = times[2]
    edge_crop(frames[mid_t], bboxes[min(int(mid_t * FPS), len(bboxes) - 1)], OUT / f"{pre}edge_crop.jpg")

    sam_c, img_c = sam_cost(n_frames), IMAGE_USD
    print("\n=== result ===")
    print(f"video    {out_mp4}  {rep}")
    print(f"contact  {OUT / (pre + 'contact.jpg')}   times {times}  (freeze {fs:.2f}-{fe:.2f}s)")
    print(f"edge     {OUT / (pre + 'edge_crop.jpg')}")
    print(f"masks    {engine}")
    print(f"freeze   world identical on all {len(fi)} freeze frames: {frozen_same}; world moves every frame before: "
          f"{moving_before}; resumes after: {resumes}; artist moves during freeze: {artist_moves}")
    if local:
        print("cost     US$0.00 (local test, no API calls)")
    else:
        used_img = bool(a.world) and not a.world_image
        print(f"cost     SAM 3.1: {max(n_frames, SAM_MIN_FRAMES)} frames x US${SAM_USD_PER_1K_FRAMES}/1k = US${sam_c:.4f}"
              f"{'' if cost['sam_billed'] else '  (cached, US$0 this run)'}")
        if used_img:
            print(f"         Muse Image: 1 x US${img_c:.2f}{'' if cost['img_billed'] else '  (cached, US$0 this run)'}")
        print(f"         billed this run: US${cost['sam_billed'] + cost['img_billed']:.4f}   "
              f"uncached total: US${sam_c + (img_c if used_img else 0):.4f}")
    if not good:
        die(f"output failed the spec check: {rep}", 1)
    (OUT / f"{pre}run.json").write_text(json.dumps({
        "clip": str(clip), "engine": engine, "freeze": [fs, fe], "contact_times": times, "output": rep,
        "freeze_checks": {"frozen_same": frozen_same, "moving_before": moving_before, "resumes": resumes,
                          "artist_moves": artist_moves}}, indent=2))


if __name__ == "__main__":
    main()
