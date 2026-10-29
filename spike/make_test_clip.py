#!/usr/bin/env python3
"""Synthesize a portrait test clip: a drawn, shaded figure swaying and walking in a dim room. Plumbing test
only: it is not footage of anyone, and it never goes in the demo.

    python spike/make_test_clip.py footage/synthetic_test.mp4            # 10 s, beat + tone audio (spike 0.0.2)

Performance mode (floor 0.1.0): the figure bobs on the test song's beat, and the clip's audio is what a phone
on set would record from a KNOWN offset into the song (band-limited playback, a room, the artist singing a
separate take live, room noise). Ground truth goes to <out>.truth.json.

    python spike/make_test_clip.py footage/perf_test.mp4 --song footage/test_song.wav --offset 26.317 --secs 26
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECS = 1080, 1920, 30, 10


def capsule(img, p0, p1, r0, r1, col, shade=0.35):
    """Tapered limb with a cheap cylinder shading."""
    n = 24
    for i in range(n + 1):
        t = i / n
        c = (int(p0[0] + (p1[0] - p0[0]) * t), int(p0[1] + (p1[1] - p0[1]) * t))
        r = int(r0 + (r1 - r0) * t)
        cv2.circle(img, c, r, col, -1, cv2.LINE_AA)
    # highlight line down the limb
    hl = tuple(min(255, int(v * (1 + shade))) for v in col)
    a = (int(p0[0] - r0 * 0.35), int(p0[1]))
    b = (int(p1[0] - r1 * 0.35), int(p1[1]))
    cv2.line(img, a, b, hl, max(2, int(r0 * 0.35)), cv2.LINE_AA)


def figure(t, beat=None):
    """Draw the figure on a transparent BGRA layer at time t. beat=(song_time, beat_s, first_beat_s): bob on it."""
    layer = np.zeros((H, W, 4), np.uint8)
    img = np.zeros((H, W, 3), np.uint8)
    m = np.zeros((H, W), np.uint8)

    sway = np.sin(t * 2 * np.pi * 0.5)
    step = np.sin(t * 2 * np.pi * 0.9)
    cx = int(W * 0.5 + sway * 120)
    base = int(H * 0.88 + abs(np.sin(t * 2 * np.pi * 0.9)) * -14)
    if beat is not None:  # phase-locked to the song: lowest point of the bob exactly on each beat
        st, bs, b0 = beat
        ph = ((st - b0) / bs) % 1.0
        cx = int(W * 0.5 + np.sin((st - b0) / (8 * bs) * 2 * np.pi) * 120)  # sway over 2 bars
        base = int(H * 0.88 + 30 * np.cos(ph * 2 * np.pi) - 16)

    skin = (120, 150, 205)       # BGR warm brown-ish
    jacket = (40, 40, 150)       # deep red
    pants = (55, 45, 40)
    shoe = (230, 230, 230)

    hip = (cx, base - 560)
    sh = (cx + int(sway * 18), base - 940)
    neck = (sh[0], sh[1] - 40)
    head = (neck[0] + int(sway * 14), neck[1] - 105)

    # legs
    for side, ph in ((-1, 0.0), (1, np.pi)):
        s = np.sin(t * 2 * np.pi * 0.9 + ph)
        knee = (hip[0] + side * 42 + int(s * 40), hip[1] + 270)
        foot = (hip[0] + side * 52 + int(s * 85), base - int(max(0, s) * 55))
        capsule(img, (hip[0] + side * 38, hip[1]), knee, 58, 44, pants)
        capsule(img, knee, foot, 44, 34, pants)
        cv2.ellipse(img, (foot[0] + side * 12, foot[1] + 10), (62, 26), 0, 0, 360, shoe, -1, cv2.LINE_AA)
    # torso
    pts = np.array([[sh[0] - 150, sh[1] + 10], [sh[0] + 150, sh[1] + 10],
                    [hip[0] + 120, hip[1] + 40], [hip[0] - 120, hip[1] + 40]], np.int32)
    cv2.fillPoly(img, [pts], jacket, cv2.LINE_AA)
    cv2.line(img, (sh[0], sh[1] + 20), (hip[0], hip[1] + 20), (70, 70, 190), 6, cv2.LINE_AA)
    # arms (one swings, one holds a mic toward the face)
    for side in (-1, 1):
        a = np.sin(t * 2 * np.pi * 0.9 + (0 if side < 0 else np.pi))
        s0 = (sh[0] + side * 150, sh[1] + 40)
        if side > 0:
            el = (s0[0] + 60, s0[1] + 190)
            hand = (head[0] + 60, head[1] + 90 + int(np.sin(t * 7) * 6))
        else:
            el = (s0[0] - 40, s0[1] + 200)
            hand = (s0[0] - 60 + int(a * 50), s0[1] + 400 + int(a * 40))
        capsule(img, s0, el, 46, 38, jacket)
        capsule(img, el, hand, 38, 28, skin)
        cv2.circle(img, hand, 30, skin, -1, cv2.LINE_AA)
    # neck + head
    capsule(img, sh, neck, 40, 34, skin)
    cv2.ellipse(img, head, (78, 98), 0, 0, 360, skin, -1, cv2.LINE_AA)
    cv2.ellipse(img, (head[0], head[1] - 52), (84, 60), 0, 180, 360, (25, 25, 30), -1, cv2.LINE_AA)
    for k in range(-5, 6):  # hair strands that break the silhouette
        cv2.line(img, (head[0] + k * 14, head[1] - 90),
                 (head[0] + k * 17 + int(np.sin(t * 5 + k) * 6), head[1] - 125), (25, 25, 30), 4, cv2.LINE_AA)
    cv2.circle(img, (head[0] - 28, head[1] - 8), 6, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(img, (head[0] + 28, head[1] - 8), 6, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.ellipse(img, (head[0], head[1] + 45), (22, 9 + int(abs(np.sin(t * 6)) * 8)), 0, 0, 360, (40, 40, 110), -1, cv2.LINE_AA)

    # mask = every drawn pixel
    m = (img.sum(axis=2) > 0).astype(np.uint8) * 255
    m = cv2.GaussianBlur(m, (0, 0), 1.1)
    # soft shading: darken one side, add rim noise so it is not flat vector art
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    shade = 0.82 + 0.28 * np.clip((xx - (cx - 220)) / 440.0, 0, 1)
    imgf = img.astype(np.float32) * shade[..., None]
    imgf = cv2.GaussianBlur(imgf, (0, 0), 1.2)
    rng = np.random.default_rng(int(t * 1000))
    imgf += rng.normal(0, 4, imgf.shape).astype(np.float32)
    layer[..., :3] = np.clip(imgf, 0, 255).astype(np.uint8)
    layer[..., 3] = m
    return layer


def background(t):
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    bg = np.zeros((H, W, 3), np.float32)
    bg[..., 0] = 70 - 30 * yy
    bg[..., 1] = 55 - 25 * yy
    bg[..., 2] = 45 - 20 * yy
    off = int(np.sin(t * 0.7) * 30)
    cv2.rectangle(bg, (140 + off, 300), (520 + off, 900), (95, 80, 60), -1)   # window
    cv2.rectangle(bg, (700 + off, 500), (980 + off, 1200), (60, 50, 70), -1)  # shelf
    cv2.rectangle(bg, (0, int(H * 0.88)), (W, H), (35, 30, 28), -1)           # floor
    bg = cv2.GaussianBlur(bg, (0, 0), 5)
    return bg


def performance_audio(song, offset, secs, out_wav, seed=1):
    """Phone-mic audio for a take that started `offset` s into the song. Returns the song's truth dict."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import soundfile as sf
    from make_test_song import SR, make_live_vocal, phone_mic
    truth = json.loads(Path(song).with_suffix(".truth.json").read_text())
    y, sr = sf.read(str(song), dtype="float32")
    y = y.mean(axis=1) if y.ndim == 2 else y
    if sr != SR:
        raise SystemExit(f"expected a {SR} Hz test song")
    live = make_live_vocal(bpm=truth["bpm"], lead_in=truth["lead_in_s"], seed=truth["seed"])
    mic = phone_mic(y, offset, secs, seed=seed, live_vocal=live, vocal_db=0.0, snr_db=20.0)
    sf.write(str(out_wav), mic, SR, subtype="PCM_16")
    return truth


def main(out, song=None, offset=0.0, secs=SECS):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    audio = out.with_suffix(".tmp.wav")
    beat = None
    if song:
        truth = performance_audio(song, offset, secs, audio)
        beat = (60.0 / truth["bpm"], truth["first_downbeat_s"])
        out.with_suffix(".truth.json").write_text(json.dumps({
            "song": str(song), "offset_s": offset, "secs": secs, "fps": FPS,
            "convention": "song_time = clip_time + offset_s",
            "audio": "phone-mic simulation: band-limited playback, room, live vocal take at 0 dB, noise at 20 dB SNR",
            "note": "drawn figure, not footage of anyone; never the demo"}, indent=2))
    else:  # beat + tone so the audio path is exercised
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        f"sine=f=220:d={secs}", "-f", "lavfi", "-i",
                        f"aevalsrc='0.6*sin(2*PI*55*t)*exp(-8*mod(t,0.5))':d={secs}:s=44100",
                        "-filter_complex", "[0][1]amix=inputs=2:normalize=0,volume=0.7", str(audio)], check=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-i", str(audio), "-c:v", "libx264", "-crf", "16",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)
    for i in range(int(round(FPS * secs))):
        t = i / FPS
        bg = background(t)
        f = figure(t, None if beat is None else (t + offset, beat[0], beat[1]))
        a = f[..., 3:4].astype(np.float32) / 255
        frame = f[..., :3].astype(np.float32) * a + bg * (1 - a)
        enc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    audio.unlink(missing_ok=True)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="footage/synthetic_test.mp4")
    ap.add_argument("--song", help="test song (performance mode); its .truth.json must sit beside it")
    ap.add_argument("--offset", type=float, default=0.0, help="song time at the clip's first frame")
    ap.add_argument("--secs", type=float, default=SECS)
    a = ap.parse_args()
    main(a.out, a.song, a.offset, a.secs)
