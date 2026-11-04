"""Several takes, one edit: Muse Wonder watches every take of the song and picks the strongest one for each section;
the picks are cut together on the section downbeats (where the world cuts anyway), and the edited clip goes through
the same render as a single take.

James, Nov 4, at work after filming three run-throughs of song B in one phone file: "we made it a one take thing,
but imagine you're able to give multiple takes and it cuts them together to give you an edited video."

What makes it safe:
- Every take is synced to the master on its own (realone.sync, which refuses a weak or ambiguous match).
- A take can be picked for a section only when it covers the whole section; a pick that does not is refused
  locally, never stretched.
- The phone must not move between takes: the tracked stage reads one camera. The takes are matched by ORB features
  and a RANSAC homography (the artist is the outlier), and a camera move over MAX_SHIFT_PX refuses the edit.
- Wonder sees the takes on one clock: each proxy is padded so video time = song time (a take that started late
  shows black until it starts)."""
import json
import subprocess
from pathlib import Path

from realone.common import Refuse
from realone.director.backend import text_part, user
from realone.director.schema import parse_json, validate

MAX_SHIFT_PX = 12          # at 1080 px wide: the phone moved more than this between takes = refuse
PROXY_H = 640              # what Wonder watches: 360 x 640, the song's clock
COVER_SLACK_S = 0.05       # a take must start no later than this after a section starts (and end no earlier)

INSTRUCTIONS = """You are the director of a music video. The artist filmed the same song several times from the same
locked phone. Each video you are given is one take, and every video runs on the song's own clock: video time is song
time (a black stretch means that take had not started yet).

For each section of the song, pick the take whose performance is strongest for that section. Judge commitment and
energy, the mouth matching the words he sings, body language that fits the section's subtext, and that his whole body
stays in frame. A take that is black, or where he is not in frame, for part of a section cannot be picked for it.
Different sections may use different takes; use the same take for every section if it really is the best each time.
Give one short line of why for each pick, naming what you saw. Return only the JSON."""

PICKS_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["picks", "why"],
    "properties": {
        "picks": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["section", "take", "why"],
            "properties": {"section": {"type": "integer", "minimum": 0},
                           "take": {"type": "integer", "minimum": 1},
                           "why": {"type": "string", "minLength": 4, "maxLength": 300}}}},
        "why": {"type": "string", "maxLength": 500}}}


def spans(takes, song):
    """-> [{"path", "offset_s", "song_start_s", "song_end_s", "sync"}] per take (song time = clip time + offset)."""
    from realone.media import probe
    from realone.sync import sync_clip
    out = []
    for p in takes:
        sy = sync_clip(song, p)
        dur = float(probe(p).get("format", {}).get("duration") or 0)
        out.append({"path": str(p), "offset_s": sy["offset_s"], "song_start_s": round(sy["offset_s"], 3),
                    "song_end_s": round(dur + sy["offset_s"], 3), "sync": {k: sy[k] for k in ("r_peak", "margin")}})
    return out


def covers(span, sec):
    return (span["song_start_s"] <= sec["start_s"] + COVER_SLACK_S
            and span["song_end_s"] >= sec["end_s"] - COVER_SLACK_S)


def check(obj, sections, takes_spans):
    """Wonder's reply -> {section index: take index (0-based)}. Refuses a pick of a take that does not cover its
    section, an unknown section or take, or a section left without a pick."""
    by = {}
    for p in obj.get("picks", []):
        i, k = p.get("section"), p.get("take", 0) - 1
        sec = next((s for s in sections if s["index"] == i), None)
        if sec is None:
            raise Refuse(f"takes: Wonder picked section {i}, which the song does not have")
        if not 0 <= k < len(takes_spans):
            raise Refuse(f"takes: Wonder picked take {k + 1} for section {i}; there are {len(takes_spans)}")
        if not covers(takes_spans[k], sec):
            raise Refuse(f"takes: take {k + 1} does not cover section {i} ({sec['start_s']:.2f}-{sec['end_s']:.2f} s; "
                         f"the take runs {takes_spans[k]['song_start_s']:.2f}-{takes_spans[k]['song_end_s']:.2f} s)")
        by[i] = k
    missing = [s["index"] for s in sections if s["index"] not in by]
    if missing:
        raise Refuse(f"takes: no pick for section(s) {missing}")
    return by


def segments(sections, picks, takes_spans, song_end_s):
    """-> [(take index, clip start s, clip end s, song start s)] in order: each section cut from its take on the song
    clock (clip = song - offset). The last section runs to the song's end where its take allows."""
    out = []
    for j, sec in enumerate(sorted(sections, key=lambda s: s["start_s"])):
        k = picks[sec["index"]]
        sp = takes_spans[k]
        a = sec["start_s"] if j else 0.0
        b = sec["end_s"] if j < len(sections) - 1 else min(song_end_s, sp["song_end_s"])
        a = max(a, sp["song_start_s"])
        out.append((k, round(a - sp["offset_s"], 4), round(b - sp["offset_s"], 4), round(a, 4)))
    return out


def _gray(path, t, width=1080):
    import numpy as np
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1", "-vf", f"scale={width}:-2",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    h = len(raw) // width
    return np.frombuffer(raw[: width * h], np.uint8).reshape(h, width)


def camera_shift(fa, fb):
    """How far the camera moved between two grayscale frames, in px at their width: ORB features matched both ways,
    a RANSAC homography (the artist moves between takes, so his features are the outliers; the wall, the floor and
    the fixtures are the inliers), and the largest displacement it gives the frame's corners. A phase correlation
    over the whole frame measured the ARTIST (299 px between two takes from an unmoved phone, Nov 4), not the camera.
    -> (shift px, inliers)."""
    import cv2
    import numpy as np
    orb = cv2.ORB_create(3000)
    ka, da = orb.detectAndCompute(fa, None)
    kb, db = orb.detectAndCompute(fb, None)
    if da is None or db is None:
        return float("inf"), 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 30:
        return float("inf"), len(m)
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    Hm, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    n_in = int(inl.sum()) if inl is not None else 0
    if Hm is None or n_in < 30:
        return float("inf"), n_in
    h, w = fa.shape
    c = np.float32([[0, 0], [w, 0], [0, h], [w, h]]).reshape(-1, 1, 2)
    return float(np.max(np.linalg.norm(cv2.perspectiveTransform(c, Hm) - c, axis=2))), n_in


def background_shift(a_path, b_path, at_a, at_b):
    """The camera's move between two takes at the same song moment (px at 1080 wide); inf when it cannot tell."""
    return camera_shift(_gray(a_path, at_a), _gray(b_path, at_b))[0]


def proxy(span, song_end_s, out_path):
    """A small proxy of one take on the song's clock (black padding before it starts), with its own mic audio."""
    pad = max(0.0, span["offset_s"])
    start = max(0.0, -span["offset_s"])
    vf = f"scale=-2:{PROXY_H},fps=15" + (f",tpad=start_duration={pad:.3f}:color=black" if pad > 0 else "")
    af = f"adelay={int(pad * 1000)}:all=1" if pad > 0 else "anull"
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", span["path"], "-t",
           f"{song_end_s - max(0.0, span['offset_s']):.3f}", "-vf", vf, "-af", af, "-c:v", "libx264", "-preset",
           "veryfast", "-crf", "30", "-c:a", "aac", "-b:a", "96k", "-ac", "1", str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path


def pick(backend, takes_spans, rec, work, effort="medium"):
    """One Wonder call with every take as video -> ({section: take index}, the reply). rec: a treatment record."""
    from realone.director import budget
    secs = rec["sections"]
    song_end = max(s["end_s"] for s in secs)
    brief = {"song_subtext": rec.get("song_subtext", ""),
             "sections": [{"index": s["index"], "role": s.get("role"), "start_s": s["start_s"], "end_s": s["end_s"],
                           "lyric": s.get("lyric", []), "subtext": s.get("subtext", ""),
                           "takes_that_cover_it": [k + 1 for k, sp in enumerate(takes_spans) if covers(sp, s)]}
                          for s in secs]}
    parts = [text_part("Sections, their lyric and subtext, and which takes cover each (JSON):\n"
                       + json.dumps(brief, indent=1))]
    for k, sp in enumerate(takes_spans):
        pv = proxy(sp, song_end, Path(work) / f"take{k + 1}_songclock.mp4")
        parts += [text_part(f"Take {k + 1}:"), {"type": "input_video", "file_id": backend.upload(pv, f"take {k + 1}")}]
    est = 1500 + len(json.dumps(brief)) // 4 + int(len(takes_spans) * song_end * budget.VIDEO_TOKENS_PER_S)
    turn = backend.respond(purpose="takes", instructions=INSTRUCTIONS, input=[user(*parts)], schema=PICKS_SCHEMA,
                           schema_name="take_picks", effort=effort, est_input_tokens=est)
    obj, errs = parse_json(turn.text)
    errs = errs or validate(obj, PICKS_SCHEMA)
    if errs:
        raise Refuse("takes: the reply did not validate: " + "; ".join(errs[:6]))
    return check(obj, secs, takes_spans), {"source": turn.source, "response_id": turn.id, "usd": round(turn.usd, 5),
                                           "picks": obj["picks"], "why": obj.get("why", "")}


def assemble(takes_spans, segs, out_path):
    """Cut the segments from the original takes (frame-accurate trims) and join them into one upright HEVC 10-bit
    HLG clip with each segment's own mic audio, so the render syncs and normalizes it like any phone take."""
    ins, fc, labels = [], [], []
    for j, (k, a, b, _) in enumerate(segs):
        ins += ["-i", takes_spans[k]["path"]]
        fc.append(f"[{j}:v]trim=start={a:.4f}:end={b:.4f},setpts=PTS-STARTPTS[v{j}]")
        fc.append(f"[{j}:a]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS[a{j}]")
        labels.append(f"[v{j}][a{j}]")
    fc.append("".join(labels) + f"concat=n={len(segs)}:v=1:a=1[v][a]")
    cmd = ["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]",
           "-c:v", "hevc_videotoolbox", "-profile:v", "main10", "-pix_fmt", "p010le", "-b:v", "60M", "-tag:v",
           "hvc1", "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc",
           "-c:a", "aac", "-b:a", "256k", str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path
