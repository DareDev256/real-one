"""README media from a render: docs/media/hook_freeze.gif (about 3 s across the freeze downbeat) and docs/media/receipt.jpg
(the finished look and the receipt look of the same frame, side by side). Rerun after the first keyed render so the
README shows real footage; until then every image is stamped TEST RENDER.

usage: .venv/bin/python scripts/readme_media.py --tag TEST_LOCAL_web_sample --section bars11_20 [--look collage]"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "out"
MEDIA = REPO / "docs" / "media"
FPS = 30


class Refuse(Exception):
    pass


def is_test(tag, run):
    """A render is a test unless its tag and its run json both say otherwise: a TEST_LOCAL_ tag, a local mask mode or
    a TEST ONLY mask engine all mean the artist was not cut out by SAM 3.1."""
    return (tag.startswith("TEST_LOCAL") or "local" in str(run.get("mode", "")).lower()
            or "test" in str(run.get("engine", "")).lower())


TEST_SONG = REPO / "footage" / "test_song.wav"
TEST_STAMP = "TEST RENDER  drawn figure, no Meta model ran"
STAND_IN_STAMP = "STAND-IN CLIP  real SAM 3.1 + Muse Image"   # fits the 360 px GIF band


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp_text(tag, run, stand_in=False, test_song=TEST_SONG):
    """-> the band to draw, or None for a real render. A keyed SAM render of the synthetic test song is still not
    the artist's film: it needs --stand-in (a stock clip and a test song, named on the image) or it is refused."""
    if is_test(tag, run):
        return TEST_STAMP
    if stand_in:
        return STAND_IN_STAMP
    song = Path(str(run.get("song", "")))
    song = song if song.is_absolute() else REPO / song
    if song.is_file() and Path(test_song).is_file() and _sha(song) == _sha(test_song):
        raise Refuse("this render uses the synthetic test song; pass --stand-in so the images say what stood in")
    return None


def inputs(tag, section, look):
    base = OUT / f"{tag}_{section}"
    paths = {"run": Path(f"{base}_run.json"), "look": Path(f"{base}_{look}.mp4"), "receipt": Path(f"{base}_receipt.mp4")}
    for name, p in paths.items():
        if "/footage/own/" in str(p.resolve()).lower():
            raise Refuse(f"{p} resolves under footage/own/, which never leaves this Mac")
        if not p.exists():
            raise Refuse(f"missing {name}: {p} (render with --look {look},receipt)")
    return paths


def freeze_window(run):
    """-> (start s, end s) of the first freeze."""
    fz = (run.get("freeze") or {}).get("freezes") or []
    if not fz:
        raise Refuse("the render has no freeze; render with --freeze")
    a, b = (int(x) for x in fz[0]["frames"].split("-"))
    return a / FPS, b / FPS


def freeze_start_s(run):
    return freeze_window(run)[0]


def clip_times(run):
    """-> (GIF start s, receipt still s). The GIF shows the freeze landing (1 s before it, 3.2 s long); when the
    render opens already frozen (a take that starts on the hook's downbeat), it shows the release instead, so the
    rain is seen hanging and then falling. The still is inside the freeze."""
    a, b = freeze_window(run)
    t0 = a - 1.0 if a >= 1.0 else max(0.0, b - 1.6)
    return t0, (a + b) / 2


def band_h(width):
    return max(30, width // 12)


def stamp_png(width, path, text=TEST_STAMP):
    """The stamp band, drawn with Pillow (this ffmpeg has no drawtext). It sits ABOVE the frame (the frame is padded
    down by the band's height), so it never covers the receipt counter in the frame's top corner."""
    from PIL import Image, ImageDraw, ImageFont
    h = band_h(width)
    img = Image.new("RGBA", (width, h), (230, 0, 126, 255))
    font = None
    for f in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            font = ImageFont.truetype(f, int(h * 0.5))
            break
        except OSError:
            continue
    ImageDraw.Draw(img).text((10, h * 0.22), text, fill="white",
                             font=font or ImageFont.load_default())
    img.save(path)
    return path


def ff(*args):
    p = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True, text=True)
    if p.returncode:
        raise Refuse(f"ffmpeg failed: {p.stderr.strip()[:400]}")


def make(tag, section, look, stand_in=False):
    paths = inputs(tag, section, look)
    run = json.loads(paths["run"].read_text())
    label = stamp_text(tag, run, stand_in)
    test = label is not None
    t0, t1 = clip_times(run)
    MEDIA.mkdir(parents=True, exist_ok=True)
    gif, jpg = MEDIA / "hook_freeze.gif", MEDIA / "receipt.jpg"
    gif_in = ["-ss", f"{t0:.3f}", "-t", "3.2", "-i", str(paths["look"])]
    chain = "[0:v]fps=10,scale=360:-2:flags=lanczos[v]"
    if test:
        gif_in += ["-i", str(stamp_png(360, MEDIA / ".stamp360.png", label))]
        chain += f";[v]pad=iw:ih+{band_h(360)}:0:{band_h(360)}[v];[v][1:v]overlay=0:0[v]"
    ff(*gif_in, "-filter_complex",
       chain + ";[v]split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer", "-loop", "0", str(gif))
    jpg_in = ["-ss", f"{t1:.3f}", "-i", str(paths["look"]), "-ss", f"{t1:.3f}", "-i", str(paths["receipt"])]
    chain = "[0:v]scale=360:-2[a];[1:v]scale=360:-2[b];[a][b]hstack=inputs=2[v]"
    if test:
        jpg_in += ["-i", str(stamp_png(720, MEDIA / ".stamp720.png", label))]
        chain += f";[v]pad=iw:ih+{band_h(720)}:0:{band_h(720)}[v];[v][2:v]overlay=0:0[v]"
    ff(*jpg_in, "-filter_complex", chain, "-map", "[v]", "-frames:v", "1", "-q:v", "3", str(jpg))
    for p in MEDIA.glob(".stamp*.png"):
        p.unlink()
    return {"gif": str(gif), "gif_bytes": gif.stat().st_size, "jpg": str(jpg), "test": test, "stamp": label}


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--tag", required=True)
    a.add_argument("--section", required=True)
    a.add_argument("--look", default="collage")
    a.add_argument("--stand-in", action="store_true", help="the clip and song are stand-ins (stock clip, the synthetic "
                                                           "test song): stamp the images so")
    args = a.parse_args(argv)
    try:
        print(json.dumps(make(args.tag, args.section, args.look, args.stand_in), indent=2))
    except Refuse as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
