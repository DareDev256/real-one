"""Join renders of different takes into one music video, section by section, frame-exact on the song clock.

Why (Nov 4): the multi-take edit could not go through the renderer as one clip. The floor track and SAM's video
tracking both assume one continuous shot, and the edited clip REFUSED at its first take change ("frame 206: 0 ground
points survived"). So each take renders its own sections from its own continuous footage (same treatment, same
worlds, same song clock), and this script cuts the finished renders together on the section downbeats, where the
world cuts anyway, then lays the master audio under the whole thing.

    python scripts/splice_takes.py --timeline out/real3_timeline.json --song SONG \
        --part 0=out/mt3_s0_clean.mp4 --part 1=out/mt1_s1_clean.mp4 --part 2=out/mt3_s2_clean.mp4 \
        --part 3=out/real3_all_clean.mp4 --out out/mt_all_clean.mp4

Each part's start on the song clock is read from its own shots file (<video stem minus the look>_shots.json). Refuses
when a part does not cover its section, or the joined frame count is off the song's by more than a frame."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

FPS = 30


def shots_of(video):
    """out/<tag>_<sec>_<look>.mp4 -> its out/<tag>_<sec>_shots.json."""
    v = Path(video)
    stem = v.stem.rsplit("_", 1)[0]
    p = v.with_name(f"{stem}_shots.json")
    if not p.exists():
        raise SystemExit(f"splice: REFUSED: no shots file for {v.name} ({p.name})")
    return json.loads(p.read_text())


def frames_of(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    return int(out.stdout.strip())


def plan(sections, parts, s0):
    """sections: the timeline's sections; parts: {section index: (video, its start s, its frames)}; s0: the joined
    video's first frame on the song clock. -> [(video, first frame, last frame exclusive)] per section, in order."""
    out = []
    secs = sorted(sections, key=lambda s: s["start_s"])
    for j, sec in enumerate(secs):
        if sec["index"] not in parts:
            raise SystemExit(f"splice: REFUSED: no part for section {sec['index']}")
        video, v0, vn = parts[sec["index"]]
        a = round((max(sec["start_s"], s0) - s0) * FPS)                 # on the joined grid
        b = round((secs[j + 1]["start_s"] - s0) * FPS) if j + 1 < len(secs) else None
        fa = a - round((v0 - s0) * FPS)                                  # on the part's own grid
        fb = (b - round((v0 - s0) * FPS)) if b is not None else vn
        if fa < 0 or fb > vn:
            raise SystemExit(f"splice: REFUSED: {Path(video).name} covers frames 0-{vn} of its own clock; section "
                             f"{sec['index']} needs {fa}-{fb}")
        out.append((video, fa, fb))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--song", required=True)
    ap.add_argument("--part", action="append", required=True, help="SECTION=video.mp4 (repeat)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tl = json.loads(Path(a.timeline).read_text())
    parts = {}
    for spec in a.part:
        k, _, v = spec.partition("=")
        sh = shots_of(v)
        parts[int(k)] = (v, float(sh["start_s"]), frames_of(v))
    s0 = min(float(p[1]) for p in parts.values())
    segs = plan(tl["sections"], parts, s0)
    total = sum(fb - fa for _, fa, fb in segs)
    want = round((max(s["end_s"] for s in tl["sections"]) - s0) * FPS)
    if abs(total - want) > 1:
        raise SystemExit(f"splice: REFUSED: the joined video would be {total} frames; the song span is {want}")
    ins, fc, labels = [], [], []
    for j, (video, fa, fb) in enumerate(segs):
        ins += ["-i", video]
        fc.append(f"[{j}:v]trim=start_frame={fa}:end_frame={fb},setpts=PTS-STARTPTS[v{j}]")
        labels.append(f"[v{j}]")
        print(f"splice: section {j}: {Path(video).name} frames {fa}-{fb} ({fb - fa})")
    fc.append("".join(labels) + f"concat=n={len(segs)}:v=1:a=0[v]")
    cmd = ["ffmpeg", "-v", "error", "-y", *ins, "-ss", f"{s0:.4f}", "-i", a.song, "-filter_complex", ";".join(fc),
           "-map", "[v]", "-map", f"{len(segs)}:a", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
           "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "256k", "-t", f"{total / FPS:.4f}",
           "-movflags", "+faststart", a.out]
    subprocess.run(cmd, check=True)
    got = frames_of(a.out)
    print(f"splice: {a.out}: {got} frames (planned {total}), master audio from {s0:.3f} s")
    if got != total:
        raise SystemExit(f"splice: REFUSED: wrote {got} frames, planned {total}")
    rec = {"parts": [{"section": s["index"], "video": v, "frames": [fa, fb]}
                     for s, (v, fa, fb) in zip(sorted(tl["sections"], key=lambda s: s["start_s"]), segs)],
           "s0": s0, "frames": got, "out": a.out}
    Path(a.out).with_suffix(".splice.json").write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
