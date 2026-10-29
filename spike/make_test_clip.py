#!/usr/bin/env python3
"""Synthesize a 10 s portrait test clip (a drawn, shaded figure swaying and
walking in a dim room, with a beat + tone audio track). Plumbing test only:
it is not footage of anyone, and it never goes in the demo.

    python spike/make_test_clip.py footage/synthetic_test.mp4
"""
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


def figure(t):
    """Draw the figure on a transparent BGRA layer at time t."""
    layer = np.zeros((H, W, 4), np.uint8)
    img = np.zeros((H, W, 3), np.uint8)
    m = np.zeros((H, W), np.uint8)

    sway = np.sin(t * 2 * np.pi * 0.5)
    step = np.sin(t * 2 * np.pi * 0.9)
    cx = int(W * 0.5 + sway * 120)
    base = int(H * 0.88 + abs(np.sin(t * 2 * np.pi * 0.9)) * -14)

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


def main(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # beat + tone so the audio path is exercised
    audio = out.with_suffix(".tmp.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"sine=f=220:d={SECS}", "-f", "lavfi", "-i",
                    f"aevalsrc='0.6*sin(2*PI*55*t)*exp(-8*mod(t,0.5))':d={SECS}:s=44100",
                    "-filter_complex", "[0][1]amix=inputs=2:normalize=0,volume=0.7", str(audio)], check=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-i", str(audio), "-c:v", "libx264", "-crf", "16",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)
    for i in range(FPS * SECS):
        t = i / FPS
        bg = background(t)
        f = figure(t)
        a = f[..., 3:4].astype(np.float32) / 255
        frame = f[..., :3].astype(np.float32) * a + bg * (1 - a)
        enc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    audio.unlink(missing_ok=True)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "footage/synthetic_test.mp4")
