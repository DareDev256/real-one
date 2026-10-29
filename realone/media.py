"""ffmpeg plumbing: probe, decode audio, normalize a span of the phone clip, read frames. From spike/freeze_test.py."""
import json
import subprocess
from pathlib import Path

import numpy as np

from realone.common import FPS, H, W, Refuse, sh


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,pix_fmt,color_transfer,duration,r_frame_rate,start_time,sample_rate",
         "-show_entries", "format=duration,start_time", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def stream(info, kind):
    return next((s for s in info.get("streams", []) if s.get("codec_type") == kind), None)


def av_skew(info):
    """Seconds the first audio sample sits after the first video frame (phone files can carry an offset).
    A decoded audio array starts at the audio stream's own start, so clip-video time = audio time + skew."""
    a, v = stream(info, "audio"), stream(info, "video")
    if not a or not v:
        return 0.0
    return float(a.get("start_time") or 0) - float(v.get("start_time") or 0)


def load_audio(path, sr, start=None, dur=None):
    """Decode any audio/video file to mono float32 at `sr` through ffmpeg (no audioread, no codec surprises).
    Sample 0 is the audio track's own first sample; see av_skew for the video clock."""
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.6f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.6f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise Refuse(f"cannot decode audio from {path}: {p.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(p.stdout, np.float32).copy()


def load_audio_native(path):
    """Master audio at its own sample rate, as stereo (for the final mux, sample-accurate slicing)."""
    info = probe(path)
    a = stream(info, "audio")
    if a is None:
        raise Refuse(f"no audio stream in {path}")
    sr = int(a.get("sample_rate") or 44100)
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", str(sr),
                        "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).reshape(-1, 2).copy(), sr


def normalize_span(clip, start, dur, work):
    """ffmpeg: [start, start+dur) of the clip -> 30 fps, 1080x1920 (scale + centre crop), BT.709. No audio (the
    render uses the master). The normalized clip is what the segmenter sees, so mask index == output frame index.
    Returns (path, n_frames)."""
    work.mkdir(parents=True, exist_ok=True)
    info = probe(clip)
    vid = stream(info, "video")
    if vid is None:
        raise Refuse(f"no video stream in {clip}")
    if vid.get("color_transfer") in ("arib-std-b67", "smpte2084"):
        print("WARNING: this clip is HDR (HLG/PQ). ffmpeg here has no zscale, so it is NOT tone-mapped; "
              "colours will look flat. Film with HDR Video off, or convert to SDR first.")
    n_frames = int(round(dur * FPS))
    norm = work / f"norm_{start:.3f}_{n_frames}f.mp4"
    if not norm.exists():
        tmp = norm.with_suffix(".tmp.mp4")
        sh(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.6f}", "-i", str(clip), "-an", "-frames:v", str(n_frames),
            "-vf", f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,"
                   "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart", str(tmp)])
        tmp.rename(norm)
    got = count_frames(norm)
    if got < n_frames:
        raise Refuse(f"the clip ran out: needed {n_frames} frames from {start:.2f}s, got {got}")
    return norm, n_frames


def count_frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out or 0)


def read_frames(path, size=(W, H)):
    """Yield BGR uint8 frames via an ffmpeg raw pipe (optionally scaled to `size`)."""
    w, h = size
    cmd = ["ffmpeg", "-v", "fatal", "-i", str(path)]
    if (w, h) != (W, H):
        cmd += ["-vf", f"scale={w}:{h}:flags=area"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def write_wav(path, stereo, sr):
    import soundfile as sf
    sf.write(str(path), stereo, sr, subtype="PCM_16")
    return Path(path)
