#!/usr/bin/env python3
"""Synthesize a portrait test clip: a drawn, shaded figure swaying and walking in a dim room. Plumbing test
only: it is not footage of anyone, and it never goes in the demo.

    python spike/make_test_clip.py footage/synthetic_test.mp4            # 10 s, beat + tone audio (spike 0.0.2)

Performance mode (floor 0.1.0): the figure bobs on the test song's beat, and the clip's audio is what a phone
on set would record from a KNOWN offset into the song (band-limited playback, a room, the artist singing a
separate take live, room noise). Ground truth goes to <out>.truth.json.

    python spike/make_test_clip.py footage/perf_test.mp4 --song footage/test_song.wav --offset 26.317 --secs 26

Phone 4K mode (0.2.0): --size WxH draws the same figure at native resolution, 2160x3840 portrait or 3840x2160
landscape (the figure stands in the 9:16 centre, the room extends to the sides), with detail that only exists at
native resolution: a 2 px fabric weave on the clothes and per-pixel sensor noise. A 1080-wide proxy averages the
weave away; a punch-in on the native frame keeps it.

    python spike/make_test_clip.py footage/perf_4k.mp4 --song footage/test_song.wav --offset 29.0 --secs 9 --size 2160x3840
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECS = 1080, 1920, 30, 10
WEAVE_AMP = 11.0    # native-only fabric weave (+-), drawn when the canvas is bigger than 1080 wide
NOISE_SIGMA = 4.0   # per-pixel sensor noise at the canvas' own resolution


class Draw:
    """cv2 drawing in 1080x1920 figure coordinates onto a canvas `s` times larger, offset `ox` px to the right.
    At s=1, ox=0 every call is exactly the plain cv2 call (the 1080 clips stay byte-identical)."""

    def __init__(self, s=1.0, ox=0.0):
        self.s, self.ox = s, ox

    def p(self, pt):
        return (int(round(pt[0] * self.s + self.ox)), int(round(pt[1] * self.s)))

    def n(self, v):
        return int(round(v * self.s))

    def circle(self, img, c, r, col, th, lt):
        cv2.circle(img, self.p(c), self.n(r), col, th if th < 0 else max(1, self.n(th)), lt)

    def line(self, img, a, b, col, th, lt):
        cv2.line(img, self.p(a), self.p(b), col, max(1, self.n(th)), lt)

    def ellipse(self, img, c, axes, ang, a0, a1, col, th, lt):
        cv2.ellipse(img, self.p(c), (self.n(axes[0]), self.n(axes[1])), ang, a0, a1, col, th, lt)

    def fillPoly(self, img, pts, col, lt):
        if self.s == 1 and self.ox == 0:
            cv2.fillPoly(img, pts, col, lt)
        else:
            cv2.fillPoly(img, [np.round(q * self.s + [self.ox, 0]).astype(np.int32) for q in pts], col, lt)


D1 = Draw()


def capsule(img, p0, p1, r0, r1, col, shade=0.35, d=D1):
    """Tapered limb with a cheap cylinder shading."""
    n = 24
    for i in range(n + 1):
        t = i / n
        c = (int(p0[0] + (p1[0] - p0[0]) * t), int(p0[1] + (p1[1] - p0[1]) * t))
        r = int(r0 + (r1 - r0) * t)
        d.circle(img, c, r, col, -1, cv2.LINE_AA)
    # highlight line down the limb
    hl = tuple(min(255, int(v * (1 + shade))) for v in col)
    a = (int(p0[0] - r0 * 0.35), int(p0[1]))
    b = (int(p1[0] - r1 * 0.35), int(p1[1]))
    d.line(img, a, b, hl, max(2, int(r0 * 0.35)), cv2.LINE_AA)


def canvas_draw(size):
    """(w, h) canvas -> Draw that puts the 1080x1920 figure in its 9:16 centre."""
    cw, chh = size
    s = min(cw / W, chh / H)
    return Draw(s, (cw - W * s) / 2)


def figure(t, beat=None, size=(W, H)):
    """Draw the figure on a transparent BGRA layer at time t. beat=(song_time, beat_s, first_beat_s): bob on it.
    size: the canvas (native resolution); the figure is drawn in its 9:16 centre at native pixel density."""
    CW, CH = size
    d = canvas_draw(size)
    layer = np.zeros((CH, CW, 4), np.uint8)
    img = np.zeros((CH, CW, 3), np.uint8)

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
        capsule(img, (hip[0] + side * 38, hip[1]), knee, 58, 44, pants, d=d)
        capsule(img, knee, foot, 44, 34, pants, d=d)
        d.ellipse(img, (foot[0] + side * 12, foot[1] + 10), (62, 26), 0, 0, 360, shoe, -1, cv2.LINE_AA)
    # torso
    pts = np.array([[sh[0] - 150, sh[1] + 10], [sh[0] + 150, sh[1] + 10],
                    [hip[0] + 120, hip[1] + 40], [hip[0] - 120, hip[1] + 40]], np.int32)
    d.fillPoly(img, [pts], jacket, cv2.LINE_AA)
    d.line(img, (sh[0], sh[1] + 20), (hip[0], hip[1] + 20), (70, 70, 190), 6, cv2.LINE_AA)
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
        capsule(img, s0, el, 46, 38, jacket, d=d)
        capsule(img, el, hand, 38, 28, skin, d=d)
        d.circle(img, hand, 30, skin, -1, cv2.LINE_AA)
    # neck + head
    capsule(img, sh, neck, 40, 34, skin, d=d)
    d.ellipse(img, head, (78, 98), 0, 0, 360, skin, -1, cv2.LINE_AA)
    d.ellipse(img, (head[0], head[1] - 52), (84, 60), 0, 180, 360, (25, 25, 30), -1, cv2.LINE_AA)
    for k in range(-5, 6):  # hair strands that break the silhouette
        d.line(img, (head[0] + k * 14, head[1] - 90),
                 (head[0] + k * 17 + int(np.sin(t * 5 + k) * 6), head[1] - 125), (25, 25, 30), 4, cv2.LINE_AA)
    d.circle(img, (head[0] - 28, head[1] - 8), 6, (20, 20, 20), -1, cv2.LINE_AA)
    d.circle(img, (head[0] + 28, head[1] - 8), 6, (20, 20, 20), -1, cv2.LINE_AA)
    d.ellipse(img, (head[0], head[1] + 45), (22, 9 + int(abs(np.sin(t * 6)) * 8)), 0, 0, 360, (40, 40, 110), -1, cv2.LINE_AA)

    # mask = every drawn pixel
    drawn = img.sum(axis=2) > 0
    m = drawn.astype(np.uint8) * 255
    m = cv2.GaussianBlur(m, (0, 0), 1.1)  # in native px: a 4K sensor resolves a sharper edge than 1080
    # soft shading: darken one side, add rim noise so it is not flat vector art
    if d.s == 1 and d.ox == 0:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    else:
        xx = ((np.arange(CW, dtype=np.float32) - d.ox) / d.s)[None, :]
    shade = 0.82 + 0.28 * np.clip((xx - (cx - 220)) / 440.0, 0, 1)
    imgf = img.astype(np.float32) * shade[..., None]
    imgf = cv2.GaussianBlur(imgf, (0, 0), 1.2)
    if d.s > 1:  # native-only detail: a 2 px weave on the clothes (dark pixels); a 2x proxy averages it to flat
        cloth = (img.max(axis=2) < 160) & drawn
        weave = np.indices((CH, CW)).sum(axis=0) % 2 * 2.0 - 1.0
        imgf += (cloth * weave * WEAVE_AMP)[..., None]
    rng = np.random.default_rng(int(t * 1000))
    imgf += rng.normal(0, NOISE_SIGMA, imgf.shape).astype(np.float32)
    layer[..., :3] = np.clip(imgf, 0, 255).astype(np.uint8)
    layer[..., 3] = m
    return layer


def background(t, size=(W, H)):
    """The dim room. At a native size it is drawn at 1080 density over the canvas' width and resized up (it is
    blurred anyway; the detail test is about the artist)."""
    if tuple(size) != (W, H):
        d = canvas_draw(size)
        wide = int(round(size[0] / d.s))
        base = background_wide(t, wide)
        return cv2.resize(base, tuple(size), interpolation=cv2.INTER_LINEAR)
    return background_wide(t, W)


def background_wide(t, w):
    ox = (w - W) // 2
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    bg = np.zeros((H, w, 3), np.float32)
    bg[..., 0] = 70 - 30 * yy
    bg[..., 1] = 55 - 25 * yy
    bg[..., 2] = 45 - 20 * yy
    off = int(np.sin(t * 0.7) * 30)
    cv2.rectangle(bg, (140 + off + ox, 300), (520 + off + ox, 900), (95, 80, 60), -1)   # window
    cv2.rectangle(bg, (700 + off + ox, 500), (980 + off + ox, 1200), (60, 50, 70), -1)  # shelf
    cv2.rectangle(bg, (0, int(H * 0.88)), (w, H), (35, 30, 28), -1)                     # floor
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


def main(out, song=None, offset=0.0, secs=SECS, size=(W, H)):
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
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
         "-r", str(FPS), "-i", "-", "-i", str(audio), "-c:v", "libx264", "-crf", "16",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)
    for i in range(int(round(FPS * secs))):
        t = i / FPS
        bg = background(t, size)
        f = figure(t, None if beat is None else (t + offset, beat[0], beat[1]), size)
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
    ap.add_argument("--size", default=f"{W}x{H}", help="native canvas WxH, e.g. 2160x3840 or 3840x2160 (phone 4K)")
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.lower().split("x"))
    if len(size) != 2 or min(size) < 360 or size[0] % 2 or size[1] % 2:
        raise SystemExit(f"--size must be WxH with even sides (got {a.size})")
    main(a.out, a.song, a.offset, a.secs, size)
