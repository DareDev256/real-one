"""Native-resolution punch-ins and the phone-clip normalize routes (0.2.0).

A phone shoots 4K (2160x3840 portrait, or 3840x2160 landscape). 0.1.0 normalized every take to 1080x1920 and the
close shot upscaled that 1.9x, so it was soft. Now the take is normalized ONCE into two frame-aligned files from
one ffmpeg filter graph (`split`, so they cannot drift a frame apart):
  proxy   1080x1920, what the segmenter sees (mask index == output frame index, as before);
  native  the 9:16 centre crop at the source's own resolution, kept for the medium and close crops.
The output is still 1080x1920; only the pixels inside a punch-in come from the native file. Masks stay at proxy
resolution and are upsampled into each crop with an edge-aware refinement (a colour guided filter whose guide is
the native crop), so the matte edge follows the real edge instead of the proxy's staircase.

HDR phone clips (HLG arib-std-b67, PQ smpte2084) are tone-mapped to SDR BT.709 through VideoToolbox (`scale_vt`;
this ffmpeg has no zscale). The hwaccel path does not autorotate, so the stream's display rotation is applied by
hand (rotation -90 -> transpose=1; measured on a real iPhone clip by the coordinator: transpose=2 is upside
down). SDR clips that are not bigger than the output stay on the 0.1.0 path, byte for byte, so cached masks keep
their keys.
"""
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from realone.common import FPS, H, W, Refuse, sh
from realone.media import count_frames, normalize_span

HDR_TRANSFERS = ("arib-std-b67", "smpte2084")
REFINE_MIN_UPSAMPLE = 1.6  # edge-refine the mask only where it is blown up this much (close shots); measured on
#                            the 4K test figure: close 1.9x 0.55% -> 0.38% wrong edge px, medium 1.4x no gain (0.61%)
NATIVE_MIN_GAIN = 1.05  # keep a native file only when it has at least 5% more pixels across than the output
GOP = FPS               # 1 s GOP in the intermediates: a worker seeking to a shot decodes at most 29 extra frames


def rotation_of(vid):
    """Display rotation in degrees from the stream's side data (display matrix) or the legacy `rotate` tag."""
    for sd in vid.get("side_data_list") or []:
        if "rotation" in sd:
            return int(round(float(sd["rotation"]))) % 360
    tag = (vid.get("tags") or {}).get("rotate")
    return int(tag) % 360 if tag else 0


def transpose_for(rot):
    """ffmpeg filters that turn coded frames upright for a display rotation (what autorotate would insert)."""
    return {0: [], 270: ["transpose=1"], 90: ["transpose=2"], 180: ["hflip", "vflip"]}.get(rot % 360, [])


def native_crop_size(dw, dh):
    """The 9:16 centre crop of a dw x dh (display) frame, even sides."""
    if dw * 16 > dh * 9:
        ch = dh - dh % 2
        cw = int(ch * 9 / 16) // 2 * 2
    else:
        cw = dw - dw % 2
        ch = int(cw * 16 / 9) // 2 * 2
    return cw, ch


def probe_video(clip):
    """The first video stream with its side data (the display matrix carries a phone clip's rotation;
    media.probe does not ask for it)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,width,height,pix_fmt,color_transfer,color_primaries,r_frame_rate:"
                          "stream_side_data=rotation:stream_tags=rotate", "-of", "json", str(clip)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise Refuse(f"cannot read {clip}: {out.stderr.strip()[-200:]}")
    streams = json.loads(out.stdout or "{}").get("streams") or []
    return streams[0] if streams else None


def probe_route(clip):
    """-> dict describing the clip and which normalize route it takes."""
    vid = probe_video(clip)
    if vid is None:
        raise Refuse(f"no video stream in {clip}")
    w, h = int(vid["width"]), int(vid["height"])
    rot = rotation_of(vid)
    dw, dh = (h, w) if rot in (90, 270) else (w, h)
    cw, ch = native_crop_size(dw, dh)
    trc = vid.get("color_transfer") or "unknown"
    return {"coded": (w, h), "rotation": rot, "display": (dw, dh), "crop": (cw, ch), "transfer": trc,
            "hdr": trc in HDR_TRANSFERS, "fps": vid.get("r_frame_rate"), "codec": vid.get("codec_name"),
            "pix_fmt": vid.get("pix_fmt"), "gain": cw / W}


def _x264(gop, preset, crf):
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-g", str(gop), "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart"]


def normalize(clip, start, dur, work, keep_native=True):
    """[start, start+dur) of the take -> {"proxy", "n", "native", "native_size", "ns", "route", ...}.

    Routes: `sdr` (0.1.0, proxy only), `sdr+native`, `hdr-vt`, `hdr-vt+native`. The native file is the 9:16 crop
    at source resolution; `ns` = (native w / 1080, native h / 1920)."""
    r = probe_route(clip)
    native = keep_native and r["gain"] >= NATIVE_MIN_GAIN
    n_frames = int(round(dur * FPS))
    if not r["hdr"] and not native:
        norm, n = normalize_span(clip, start, dur, work)
        return dict(r, proxy=norm, n=n, native=None, native_size=None, ns=(1.0, 1.0), route="sdr")
    work.mkdir(parents=True, exist_ok=True)
    cw, ch = r["crop"]
    route = ("hdr-vt" if r["hdr"] else "sdr") + ("+native" if native else "")
    stem = f"{start:.3f}_{n_frames}f_{route.replace('+', '_')}"
    proxy = work / f"norm_{stem}.mp4"
    nat = work / f"native_{stem}_{cw}x{ch}.mp4" if native else None
    if not proxy.exists() or (nat is not None and not nat.exists()):
        inp = ["-ss", f"{start:.6f}", "-i", str(clip)]
        head = []
        if r["hdr"]:
            w, h = r["coded"]
            inp = ["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"] + inp
            head = [f"scale_vt=w={w}:h={h}:color_matrix=bt709:color_primaries=bt709:color_transfer=bt709",
                    "hwdownload", "format=p010le", "format=yuv420p"] + transpose_for(r["rotation"])
        body = head + [f"fps={FPS}", f"crop={cw}:{ch}:(iw-{cw})/2:(ih-{ch})/2", "setsar=1"]
        to709 = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
        tmp_p = proxy.with_suffix(".tmp.mp4")
        if native:
            graph = (",".join(body) + f",split=2[a][b];[a]scale={W}:{H}:flags=area,{to709}[p];[b]{to709}[n]")
            tmp_n = nat.with_suffix(".tmp.mp4")
            cmd = (["ffmpeg", "-y", "-v", "error"] + inp + ["-an", "-filter_complex", graph,
                   "-map", "[p]", "-frames:v", str(n_frames)] + _x264(GOP, "veryfast", 12) + [str(tmp_p),
                   "-map", "[n]", "-frames:v", str(n_frames)] + _x264(GOP, "ultrafast", 12) + [str(tmp_n)])
        else:
            vf = ",".join(body + [f"scale={W}:{H}:flags=area", to709])
            cmd = (["ffmpeg", "-y", "-v", "error"] + inp + ["-an", "-vf", vf, "-frames:v", str(n_frames)]
                   + _x264(GOP, "veryfast", 12) + [str(tmp_p)])
        sh(cmd)
        tmp_p.rename(proxy)
        if native:
            tmp_n.rename(nat)
    got = count_frames(proxy)
    if got < n_frames:
        raise Refuse(f"the clip ran out: needed {n_frames} frames from {start:.2f}s, got {got}")
    if nat is not None and count_frames(nat) != got:
        raise Refuse(f"native and proxy files disagree on frame count ({count_frames(nat)} vs {got})")
    ns = ((cw / W, ch / H) if native else (1.0, 1.0))
    return dict(r, proxy=proxy, n=n_frames, native=nat, native_size=(cw, ch) if native else None, ns=ns,
                route=route)


# --------------------------------------------------------------------------- per-frame: crop and mask

def native_crop(native_bgr, crop, ns):
    """The output-size view of a punch-in, cut from the native frame.
    crop = (x0, y0, cw, ch) in proxy pixels; ns = (sx, sy) native px per proxy px. -> float32 (H, W, 3) 0..1."""
    nh, nw = native_bgr.shape[:2]
    x0, y0, cw, ch = crop
    sx, sy = ns
    X0, Y0, CW, CH = x0 * sx, y0 * sy, cw * sx, ch * sy
    dx, dy = CW / W, CH / H  # native px per output px
    m = 3
    ix0, iy0 = max(0, int(math.floor(X0)) - m), max(0, int(math.floor(Y0)) - m)
    ix1, iy1 = min(nw, int(math.ceil(X0 + CW)) + m), min(nh, int(math.ceil(Y0 + CH)) + m)
    roi = native_bgr[iy0:iy1, ix0:ix1].astype(np.float32) * (1 / 255)
    d = max(dx, dy)
    if d > 1.05:  # going down in size: low-pass first so the resample does not alias
        roi = cv2.GaussianBlur(roi, (0, 0), 0.5 * math.sqrt(d * d - 1))
    M = np.float32([[1 / dx, 0, -(X0 - ix0) / dx], [0, 1 / dy, -(Y0 - iy0) / dy]])
    out = cv2.warpAffine(roi, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out, 0, 1)


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_refine(guide, p, r=2, eps=1e-4, sub=2):
    """Fast colour guided filter (He et al. 2010/2015): p (H, W) soft mask, guide (H, W, 3) 0..1.
    The linear coefficients are solved on a 1/sub grid (window radius r there, so r*sub px at full size) and
    applied to the full-resolution guide, so the matte edge lands on the guide's real edge."""
    h, w = p.shape
    hs, ws = h // sub, w // sub
    I = cv2.resize(guide, (ws, hs), interpolation=cv2.INTER_AREA)
    ps = cv2.resize(p, (ws, hs), interpolation=cv2.INTER_AREA)
    mI = _box(I, r)
    mp = _box(ps, r)
    mIp = _box(I * ps[..., None], r) - mI * mp[..., None]
    # covariance of the guide per window (symmetric 3x3), regularised
    c = {}
    for i in range(3):
        for j in range(i, 3):
            c[i, j] = _box(I[..., i] * I[..., j], r) - mI[..., i] * mI[..., j] + (eps if i == j else 0)
    a11, a12, a13, a22, a23, a33 = c[0, 0], c[0, 1], c[0, 2], c[1, 1], c[1, 2], c[2, 2]
    i11 = a22 * a33 - a23 * a23
    i12 = a13 * a23 - a12 * a33
    i13 = a12 * a23 - a13 * a22
    i22 = a11 * a33 - a13 * a13
    i23 = a13 * a12 - a11 * a23
    i33 = a11 * a22 - a12 * a12
    det = a11 * i11 + a12 * i12 + a13 * i13
    v0, v1, v2 = mIp[..., 0], mIp[..., 1], mIp[..., 2]
    A = np.stack([(i11 * v0 + i12 * v1 + i13 * v2), (i12 * v0 + i22 * v1 + i23 * v2),
                  (i13 * v0 + i23 * v1 + i33 * v2)], -1) / det[..., None]
    b = mp - (A * mI).sum(-1)
    A = cv2.resize(_box(A, r), (w, h), interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(_box(b, r), (w, h), interpolation=cv2.INTER_LINEAR)
    return (A * guide).sum(-1) + b


def refine_mask_in_crop(fg, mask_bool, crop):
    """Proxy mask -> output-space hard mask for a punch-in, edge-refined against the native view `fg`.
    The guided result is used only in a band around the upsampled edge; deep inside and far outside, the
    upsampled mask stands (a guided filter can bleed across a busy interior)."""
    x0, y0, cw, ch = crop
    z = W / cw
    M = np.float32([[z, 0, -z * x0], [0, z, -z * y0]])
    up = cv2.warpAffine(mask_bool.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    q = guided_refine(fg, up)
    reach = max(2, int(round(1.5 * z)))  # how far the true edge can sit from the upsampled one
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    hard0 = (up > 0.5).astype(np.uint8)
    band = (cv2.dilate(hard0, k) > 0) & ~(cv2.erode(hard0, k) > 0)
    out = np.where(band, q, up)
    return out > 0.5
