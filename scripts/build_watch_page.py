#!/usr/bin/env python3
"""Build the Real One watch page (web/) into a static folder a judge can open.

    .venv/bin/python scripts/build_watch_page.py --tag web_sample --section bars:11-20 \
        [--treatment out/TEST_LOCAL_director_integ02_treatment.json] [--credits FILE] [--out web/dist]

Reads one render's own files from out/ (`<tag>_<section>_run.json`, `_shots.json`, `_receipt.json` and the look
videos the run json names) and writes `<out>/index.html`, `watch.css`, `watch.js` and `<out>/data/`:
  finished.mp4, receipt.mp4  the two looks, re-encoded for the web (H.264 high, yuv420p, faststart, 1 s GOP, at most
                             1080x1920), checked after encoding: same frame count as each other and as the receipt
  poster.jpg                 a frame from inside the freeze (or 1 s in)
  receipt.json shots.json run.json timeline.json   the render's own files, local paths stripped
  treatment.json lyrics.json the director's treatment (only when --treatment is given and its sections line up)
  credits.json               which Meta model did what, with calls / frames / US$ (see build_credits)
  manifest.json              what the page loads, and whether this is a TEST build and why

Refuses (exit 2, one clear line) rather than guessing: a missing file, a render without a receipt look, a tag that
matches both a real and a TEST_LOCAL render, any input under footage/own/, a treatment for another song or other
section timings, videos whose frame counts disagree, a credits file that claims a Meta model ran on a test render,
and a real render with no credits file (the page will not invent calls or dollars).

--credits FILE (required for a real render; a test render gets TEST credits generated, every model ran=false):
  {"status": "REAL", "statement": "...", "total_usd": 0.0,
   "models": [{"name": "SAM 3.1", "did": "...", "ran": true, "calls": 0, "frames": 0, "usd": 0.0}, ...],
   "code": [{"name": "...", "did": "..."}]}
  A model that ran needs numeric calls and usd. On a test render the file must say TEST and no model may have run.

A TEST banner is written into index.html itself (not only by the page's script) whenever the render tag starts with
TEST_LOCAL_, the masks were cut locally, the worlds are the procedural test plates, or the treatment is a FIXTURE.
"""
import argparse
import datetime as dt
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from realone.common import LOCAL_PREFIX, Refuse  # noqa: E402

WEB = REPO / "web"
PAGE_FILES = ("index.html", "watch.css", "watch.js", "vercel.json")
BANNER_MARK = "<!-- TEST_BANNER -->"
DIST_MARK = ".realone-watch"
FINISHED_LOOKS = ("collage", "clean")
MAX_W, MAX_H = 1080, 1920
TIME_TOL_S = 0.05

META_MODELS = [
    {"id": "sam-3.1", "name": "SAM 3.1", "did": "Cuts the artist out of every frame. Real pixels, never generated."},
    {"id": "voice-transcribe", "name": "Voice Transcribe", "did": "Hears the lyrics in the master, section by "
                                                                   "section. The artist's own sheet outranks it."},
    {"id": "muse-wonder", "name": "Muse Wonder", "did": "Directs from what the song means: subtext, world and look "
                                                         "per section, the one hook that freezes. Watches its own "
                                                         "cut and revises once."},
    {"id": "muse-image", "name": "Muse Image", "did": "Builds one world per section, 9:16, no people in frame."},
]
OUR_CODE = [
    {"name": "Beat grid + sync", "did": "Downbeats and sections from the master; the phone mic synced to it."},
    {"name": "Compositor", "did": "Reframe, edge despill, colour match, grain; cuts land on downbeats."},
    {"name": "Hook Freeze", "did": "The world stops on the hook's downbeat while the artist keeps moving."},
    {"name": "Real Receipt", "did": "Measures the filmed share of every frame; generated pixels go grey."},
    {"name": "Literal-lyrics guard", "did": "Rejects any world that draws what the lyric says."},
]


# ---------------------------------------------------------------- paths

def sec_name(section):
    """The renderer's own section stem (realone/render.py): hook:2 -> hook2, bars:11-20 -> bars11_20, #3 -> s3."""
    return section.replace(":", "").replace("-", "_").replace("#", "s")


def _under(path, root):
    p, r = str(path).lower().rstrip("/"), str(root).lower().rstrip("/")
    return p == r or p.startswith(r + "/")


def refuse_own(path, repo=REPO):
    """Refuse a path under footage/own/, as given or wherever its symlinks lead (case-insensitive: APFS is)."""
    p = Path(path).expanduser()
    roots = {(repo / "footage" / "own").absolute(), (repo / "footage" / "own").resolve()}
    cands = {p.absolute(), p.resolve()}
    for c in cands:
        for r in roots:
            if _under(c, r):
                raise Refuse(f"{path} is under footage/own/ (the artist's own footage); the watch page never reads "
                             f"from there")
    return p


def need(path, what, repo=REPO):
    refuse_own(path, repo)
    if not Path(path).is_file():
        raise Refuse(f"missing {what}: {path}")
    return Path(path)


def find_render(out_dir, tag, section, repo=REPO):
    """-> the render's file stem. Tries the tag as given, then with TEST_LOCAL_; refuses when both exist."""
    refuse_own(out_dir, repo)
    sec = sec_name(section)
    stems = [f"{tag}_{sec}"]
    if not tag.startswith(LOCAL_PREFIX):
        stems.append(f"{LOCAL_PREFIX}{tag}_{sec}")
    found = [s for s in stems if (Path(out_dir) / f"{s}_run.json").exists()]
    if len(found) > 1:
        raise Refuse(f"tag {tag!r} matches a real render and a test render ({', '.join(found)}); pass the full tag")
    if not found:
        raise Refuse(f"no render for tag {tag!r} section {section!r}: no {Path(out_dir) / (stems[0] + '_run.json')}"
                     + (f" or {stems[1]}_run.json" if len(stems) > 1 else "")
                     + f". Render it first: python -m realone render ... --tag {tag} --section {section} "
                       f"--look collage,receipt")
    return found[0]


def load_json(path, what, repo=REPO):
    p = need(path, what, repo)
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise Refuse(f"{what} is not valid JSON: {p}: {e}")


# ---------------------------------------------------------------- truth

def test_reasons(stem, run, shots, treatment=None):
    """Why this build is TEST output (empty list = a real render). Any one reason puts the banner on the page."""
    reasons = []
    if stem.startswith(LOCAL_PREFIX):
        reasons.append("Tag TEST_LOCAL_")
    engine = str(run.get("engine", ""))
    if run.get("mode") == "local test" or "TEST ONLY" in engine or "Apple Vision" in engine:
        reasons.append("Masks: Apple Vision on a Mac, not SAM 3.1")
    if any("local plates" in str(w.get("from", "")) for w in shots.get("world_sources", [])):
        reasons.append("Worlds: procedural test plates, not Muse Image")
    if treatment is not None and (treatment.get("FIXTURE") or str(treatment.get("mode", "")).upper() == "DRY RUN"):
        reasons.append("Treatment: FIXTURE, hand-written, not model output")
    return reasons


def banner_html(reasons):
    items = "".join(f"<li>{_esc(r)}</li>" for r in reasons)
    return ('<aside class="test-banner" id="test-banner" role="note" aria-label="Test render">'
            '<strong>TEST RENDER</strong><span>Not the film. No Meta model ran.</span>'
            f'<ul>{items}</ul></aside>')


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def build_credits(reasons, run, credits_file=None, repo=REPO):
    """credits.json for the page. A test render gets TEST credits generated here (no model ran, US$0.00 billed);
    a real render must bring its own file with the real calls, frames and US$."""
    if credits_file is None:
        if not reasons:
            raise Refuse("a real render needs --credits FILE with the real calls, frames and US$ per Meta model; "
                         "the page will not invent them (shape: this script's docstring)")
        stand_in = {
            "sam-3.1": "Apple Vision on this Mac (TEST ONLY)",
            "voice-transcribe": "not run; the lyric beside the video is a FIXTURE",
            "muse-wonder": "not run; the treatment beside the video is a FIXTURE",
            "muse-image": "three procedural test plates (TEST ONLY)",
        }
        return {
            "status": "TEST",
            "statement": "No Meta model ran for this render. Each line below names what the model does in Real One "
                         "and what stood in for it here.",
            "models": [dict(m, ran=False, stand_in=stand_in[m["id"]], calls=0, frames=0, usd=0.0)
                       for m in META_MODELS],
            "code": OUR_CODE,
            "total_usd": float(run.get("billed_usd", 0.0) or 0.0),
        }
    doc = load_json(credits_file, "credits file", repo)
    check_credits(doc, reasons)
    return doc


def check_credits(doc, reasons):
    status = doc.get("status")
    if status not in ("TEST", "REAL"):
        raise Refuse(f"credits status must be TEST or REAL, not {status!r}")
    models = doc.get("models")
    if not isinstance(models, list) or not models:
        raise Refuse("credits need a non-empty models list")
    for m in models:
        if not m.get("name") or not m.get("did") or not isinstance(m.get("ran"), bool):
            raise Refuse(f"credits model entry needs name, did and ran (true/false): {m}")
    if reasons:
        if status != "TEST":
            raise Refuse("this is a test render; its credits must say TEST (" + " ".join(reasons) + ")")
        ran = [m["name"] for m in models if m["ran"]]
        if ran:
            raise Refuse(f"this is a test render, but the credits claim {', '.join(ran)} ran")
    elif status != "REAL":
        raise Refuse("this is a real render; credits marked TEST would undersell it. Fill in the real numbers")
    for m in models:
        if m["ran"]:
            for k in ("calls", "usd"):
                if not isinstance(m.get(k), (int, float)) or isinstance(m.get(k), bool):
                    raise Refuse(f"credits: {m['name']} ran, so it needs a numeric {k}")


def song_sections(shots, timeline=None):
    """The song's own sections that this render covers. A bars:A-B render's shot plan holds one pseudo-section
    (index -1), so the song's sections come from the timeline json the run names, clipped to the render's range."""
    if timeline:
        a, b = shots["start_s"], shots["end_s"]
        return [s for s in timeline.get("sections", []) if s["end_s"] > a + 1e-3 and s["start_s"] < b - 1e-3]
    return [s for s in shots.get("sections", []) if s.get("index", -1) >= 0]


def check_treatment(treatment, run, shots, timeline=None):
    """The treatment must be for this song and its sections must line up with the render's (index and timing)."""
    song_t, song_r = treatment.get("song"), run.get("song")
    if song_t and song_r and Path(song_t).name != Path(song_r).name:
        raise Refuse(f"the treatment is for {song_t}, the render is of {song_r}")
    secs = song_sections(shots, timeline)
    if not secs:
        raise Refuse("the render names no song sections (no timeline json), so the treatment cannot be lined up")
    by_index = {s.get("index"): s for s in treatment.get("sections", [])}
    for s in secs:
        t = by_index.get(s.get("index"))
        if t is None:
            raise Refuse(f"the treatment has no section #{s.get('index')} ({s.get('role')}) of this render")
        if abs(t["start_s"] - s["start_s"]) > TIME_TOL_S or abs(t["end_s"] - s["end_s"]) > TIME_TOL_S:
            raise Refuse(f"the treatment's section #{s['index']} is {t['start_s']:.3f}-{t['end_s']:.3f}s, the "
                         f"render's is {s['start_s']:.3f}-{s['end_s']:.3f}s: not the same cut of the song")


# ---------------------------------------------------------------- media

def encode(src, dst, audio=True):
    vf = (f"scale='min({MAX_W},iw)':'min({MAX_H},ih)':force_original_aspect_ratio=decrease,"
          "scale=trunc(iw/2)*2:trunc(ih/2)*2")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-map", "0:v:0"]
    cmd += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "160k"] if audio else ["-an"]
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-maxrate", "6M", "-bufsize", "12M",
            "-profile:v", "high", "-pix_fmt", "yuv420p", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
            "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                        "stream=codec_name,width,height,pix_fmt,nb_read_frames,r_frame_rate", "-of", "json",
                        str(path)], check=True, capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]
    return {"codec": s["codec_name"], "width": int(s["width"]), "height": int(s["height"]), "pix_fmt": s["pix_fmt"],
            "frames": int(s["nb_read_frames"]), "faststart": moov_first(path)}


def moov_first(path):
    """True when the moov atom comes before mdat (the browser can start playing before the whole file arrives)."""
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                return False
            size, kind = int.from_bytes(head[:4], "big"), head[4:8]
            if kind == b"moov":
                return True
            if kind == b"mdat":
                return False
            if size == 1:
                size = int.from_bytes(f.read(8), "big")
                f.seek(size - 16, 1)
            elif size < 8:
                return False
            else:
                f.seek(size - 8, 1)


def check_web_videos(specs, n_frames):
    for name, s in specs.items():
        bad = []
        if s["codec"] != "h264":
            bad.append(f"codec {s['codec']}")
        if s["width"] > MAX_W or s["height"] > MAX_H:
            bad.append(f"size {s['width']}x{s['height']}")
        if s["pix_fmt"] != "yuv420p":
            bad.append(f"pix_fmt {s['pix_fmt']}")
        if not s["faststart"]:
            bad.append("moov after mdat (no faststart)")
        if bad:
            raise Refuse(f"{name} is not fit for the web: {', '.join(bad)}")
    counts = {name: s["frames"] for name, s in specs.items()}
    if len(set(counts.values())) != 1:
        raise Refuse(f"the two looks have different frame counts {counts}; they cannot play in sync")
    got = next(iter(counts.values()))
    if abs(got - n_frames) > 1:
        raise Refuse(f"the videos have {got} frames, the receipt has {n_frames}: the counter would drift")


def poster(src, dst, t):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
                    "-q:v", "3", str(dst)], check=True)


# ---------------------------------------------------------------- json hygiene

def scrub(o, repo=REPO):
    """Strip local paths (repo -> relative, any other absolute path -> its name) and NaN (not valid JSON)."""
    if isinstance(o, dict):
        return {k: scrub(v, repo) for k, v in o.items()}
    if isinstance(o, list):
        return [scrub(v, repo) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, str):
        r = str(repo) + "/"
        if o.startswith(r):
            return o[len(r):]
        if o.startswith(("/Users/", "/private/", "/home/", "/var/", "/tmp/")):
            return Path(o).name
    return o


def dump(path, obj):
    Path(path).write_text(json.dumps(scrub(obj), indent=1, allow_nan=False))


# ---------------------------------------------------------------- build

def prepare_out(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not (out / DIST_MARK).exists():
        raise Refuse(f"{out} exists and was not made by this script (no {DIST_MARK}); pick an empty --out")
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / DIST_MARK).write_text("built by scripts/build_watch_page.py; safe to delete\n")
    return out


def build(tag, section, treatment_path=None, credits_path=None, out=None, look=None, out_dir=None, repo=REPO,
          web=WEB, encoder=encode, prober=probe, posterer=poster, log=print):
    out_dir = Path(out_dir or repo / "out")
    out = Path(out or web / "dist")
    refuse_own(out, repo)
    for p in (treatment_path, credits_path):
        if p:
            refuse_own(p, repo)

    stem = find_render(out_dir, tag, section, repo)
    run = load_json(out_dir / f"{stem}_run.json", "run json", repo)
    shots = load_json(out_dir / f"{stem}_shots.json", "shots json", repo)
    receipt = load_json(out_dir / f"{stem}_receipt.json", "receipt json", repo)

    looks = run.get("looks", {})
    if "receipt" not in looks:
        raise Refuse(f"{stem} has no receipt look ({', '.join(looks) or 'none'}); render it with "
                     f"--look collage,receipt so both looks share one cut")
    finished = look or next((k for k in FINISHED_LOOKS if k in looks), None)
    if finished not in looks or finished == "receipt":
        raise Refuse(f"{stem} has no finished look to pair with the receipt (wanted {look or 'collage or clean'}, "
                     f"have {', '.join(looks)})")
    videos = {}
    for k in (finished, "receipt"):
        v = Path(looks[k].get("video", ""))
        if not v.is_absolute():
            v = repo / v
        videos[k] = need(v, f"{k} video", repo)
        if k not in receipt.get("outputs", {}):
            raise Refuse(f"the receipt json has no '{k}' output")
    n_frames = int(shots["n_frames"])
    for k in (finished, "receipt"):
        n = len(receipt["outputs"][k]["frames"]["filmed_pct"])
        if n != n_frames:
            raise Refuse(f"the receipt has {n} frames for {k}, the shot plan has {n_frames}")

    timeline = None
    if run.get("timeline"):
        tp = Path(run["timeline"])
        timeline = load_json(tp if tp.is_absolute() else repo / tp, "timeline json (named by the run json)", repo)

    treatment = lyrics = None
    if treatment_path:
        treatment = load_json(treatment_path, "treatment", repo)
        check_treatment(treatment, run, shots, timeline)
        lp = Path(str(treatment_path).replace("_treatment.json", "_lyrics.json"))
        if lp != Path(treatment_path) and lp.is_file():
            refuse_own(lp, repo)
            lyrics = json.loads(lp.read_text())

    reasons = test_reasons(stem, run, shots, treatment)
    credits = build_credits(reasons, run, credits_path, repo)

    template = (web / "index.html").read_text()
    if BANNER_MARK not in template:
        raise Refuse(f"web/index.html has lost its {BANNER_MARK} marker; the TEST banner would have nowhere to go")

    out = prepare_out(out)
    data = out / "data"
    log(f"encode {videos[finished].name} -> data/finished.mp4")
    encoder(videos[finished], data / "finished.mp4", audio=True)
    log(f"encode {videos['receipt'].name} -> data/receipt.mp4")
    encoder(videos["receipt"], data / "receipt.mp4", audio=False)
    specs = {"finished.mp4": prober(data / "finished.mp4"), "receipt.mp4": prober(data / "receipt.mp4")}
    check_web_videos(specs, n_frames)
    fps = float(shots.get("fps", receipt.get("fps", 30)))
    fz = shots.get("freezes") or []
    t_poster = ((fz[0]["start_frame"] + min(fz[0]["frames"], 45)) / fps) if fz else min(1.0, n_frames / fps / 2)
    posterer(data / "finished.mp4", data / "poster.jpg", t_poster)

    dump(data / "receipt.json", receipt)
    dump(data / "shots.json", shots)
    dump(data / "run.json", run)
    if timeline is not None:
        dump(data / "timeline.json", {k: timeline[k] for k in ("song", "bpm", "beats_per_bar", "sections")
                                      if k in timeline})
    dump(data / "credits.json", credits)
    if treatment is not None:
        dump(data / "treatment.json", treatment)
    if lyrics is not None:
        dump(data / "lyrics.json", lyrics)
    manifest = {
        "built": dt.datetime.now().isoformat(timespec="seconds"),
        "render": stem, "section": section, "version": run.get("version"),
        "fps": fps, "frames": n_frames, "duration_s": round(n_frames / fps, 3), "start_s": shots.get("start_s"),
        "looks": {"finished": finished, "receipt": "receipt"},
        "videos": {"finished": "finished.mp4", "receipt": "receipt.mp4", "poster": "poster.jpg"},
        "video_specs": specs,
        "files": {"receipt": "receipt.json", "shots": "shots.json", "run": "run.json", "credits": "credits.json",
                  "timeline": "timeline.json" if timeline is not None else None,
                  "treatment": "treatment.json" if treatment is not None else None,
                  "lyrics": "lyrics.json" if lyrics is not None else None},
        "test": bool(reasons), "test_reasons": reasons,
    }
    dump(data / "manifest.json", manifest)

    page = template.replace(BANNER_MARK, banner_html(reasons) if reasons else "")
    page = page.replace('data-test="0"', f'data-test="{1 if reasons else 0}"')
    (out / "index.html").write_text(page)
    for f in PAGE_FILES[1:]:
        shutil.copy2(web / f, out / f)
    log(f"built {out} from {stem}: {n_frames} frames, looks {finished}+receipt, "
        f"{'TEST (' + str(len(reasons)) + ' reasons)' if reasons else 'REAL'}, "
        f"treatment {'yes' if treatment is not None else 'no'}")
    return out, manifest


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the Real One watch page from one render (static, no deploy).")
    p.add_argument("--tag", required=True, help="the render tag (TEST_LOCAL_ is tried too)")
    p.add_argument("--section", required=True, help="the section as passed to render, e.g. hook or bars:11-20")
    p.add_argument("--treatment", help="the director's _treatment.json; its sections must line up with the render")
    p.add_argument("--credits", help="credits.json with the real calls, frames and US$ (required for a real render)")
    p.add_argument("--look", choices=list(FINISHED_LOOKS), help="the finished look to pair with the receipt")
    p.add_argument("--out", default=str(WEB / "dist"))
    a = p.parse_args(argv)
    try:
        build(a.tag, a.section, a.treatment, a.credits, a.out, a.look)
    except Refuse as e:
        print(f"build_watch_page: REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
