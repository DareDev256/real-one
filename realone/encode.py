"""ffmpeg encode (H.264 yuv420p BT.709 faststart, master audio) and the output checks. From the spike."""
import os
import struct
import subprocess

import numpy as np

from realone.common import FPS, H, W
from realone.media import probe, read_frames


def start_encoder(out, audio, final=False):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-i", str(audio)]
    cmd += ["-map", "0:v"] + (["-map", "1:a"] if audio else [])
    cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264",
            "-preset", "slow" if final else "veryfast", "-crf", "17" if final else "20", "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "256k", "-shortest"]
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


def verify_output(path, want_s):
    info = probe(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio = any(s["codec_type"] == "audio" for s in info["streams"])
    rep = {"codec": v["codec_name"], "size": f"{v['width']}x{v['height']}", "pix_fmt": v["pix_fmt"],
           "duration_s": round(float(info["format"]["duration"]), 3), "audio": audio,
           "faststart": faststart_ok(path)}
    good = (rep["codec"] == "h264" and rep["size"] == f"{W}x{H}" and rep["pix_fmt"] == "yuv420p"
            and rep["faststart"] and audio and abs(rep["duration_s"] - want_s) < 0.1)
    return rep, good


def detect_cuts(path, min_ratio=6.0):
    """Find hard cuts in the rendered video itself: frames whose difference from the previous frame stands far
    above the clip's typical frame-to-frame change. Returns frame indices (the first frame of each new shot)."""
    prev, diffs = None, []
    for fr in read_frames(path, size=(135, 240)):
        f = fr.astype(np.float32)
        diffs.append(0.0 if prev is None else float(np.abs(f - prev).mean()))
        prev = f
    d = np.array(diffs)
    if len(d) < 3:
        return [], d
    base = np.median(d[1:]) + 1e-6
    cuts = [i for i in range(1, len(d)) if d[i] > min_ratio * base and d[i] > 6.0]
    return cuts, d
