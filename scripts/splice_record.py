"""Write a render record (run, shots, receipt, timeline) for a spliced cut, so the watch page and the README media
read it like any render.

A spliced video (scripts/splice_takes.py) is made of several renders' frames. Each part's numbers stay its own: the
shots are the parts' shots re-indexed onto the joined frames, the receipt's per-frame filmed share is concatenated
from the parts' own measurements (nothing is re-estimated), the freeze and the receipt reveal come from the part
that rendered them, and every check is the parts' checks plus the joined file's own frame count and audio offset.

    python scripts/splice_record.py --tag final --clean out/final_all_clean.mp4 --receipt out/final_all_receipt.mp4

Each MP4 must have its <stem>.splice.json beside it (written by splice_takes.py). Refuses when the two looks were
spliced from different frame ranges or the parts' records disagree on the song."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "out"
FPS = 30


def records(video):
    v = Path(video)
    stem = v.stem.rsplit("_", 1)[0]
    look = v.stem.rsplit("_", 1)[1]
    got = {}
    for kind in ("run", "shots", "receipt"):
        p = v.with_name(f"{stem}_{kind}.json")
        if not p.exists():
            raise SystemExit(f"splice_record: REFUSED: {p.name} missing for {v.name}")
        got[kind] = json.loads(p.read_text())
    return look, got


def frames_of(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    return int(out.stdout.strip())


def audio_error_ms(video, song, s0):
    sys.path.insert(0, str(REPO))
    from realone.media import load_audio
    from realone.sync import SR, measure_offset
    r = measure_offset(load_audio(song, SR), load_audio(str(video), SR), near_s=s0)
    return round((r["offset_s"] - s0) * 1000, 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--clean", required=True, help="the spliced finished look (…_clean.mp4)")
    ap.add_argument("--receipt", required=True, help="the same splice in the Real Receipt look (…_receipt.mp4)")
    a = ap.parse_args(argv)
    sp = {lk: json.loads(Path(v).with_suffix(".splice.json").read_text()) for lk, v in
          (("clean", a.clean), ("receipt", a.receipt))}
    ranges = {lk: [(p["section"], tuple(p["frames"])) for p in s["parts"]] for lk, s in sp.items()}
    if ranges["clean"] != ranges["receipt"] or sp["clean"]["s0"] != sp["receipt"]["s0"]:
        raise SystemExit("splice_record: REFUSED: the two looks were spliced from different frame ranges")
    s0 = sp["clean"]["s0"]
    parts = []
    for pc, pr in zip(sp["clean"]["parts"], sp["receipt"]["parts"]):
        lc, rc = records(pc["video"])
        lr, rr = records(pr["video"])
        parts.append({"section": pc["section"], "frames": pc["frames"], "clean": (lc, rc), "receipt": (lr, rr)})
    songs = {p["clean"][1]["run"]["song"] for p in parts} | {p["receipt"][1]["run"]["song"] for p in parts}
    if len(songs) != 1:
        raise SystemExit(f"splice_record: REFUSED: the parts were rendered against different songs: {songs}")
    n = sum(b - a for p in parts for a, b in [p["frames"]])
    # shots: each part's shots that fall inside its used range, re-indexed onto the joined frames
    shots, offset = [], 0
    freezes, reveals, moments = [], [], []
    for p in parts:
        sh = p["clean"][1]["shots"]
        fa, fb = p["frames"]
        for s in sh["shots"]:
            a, b = max(s["start_frame"], fa), min(s["end_frame"], fb)
            if a < b:
                shots.append({**s, "index": len(shots), "start_frame": a - fa + offset, "end_frame": b - fa + offset,
                              "frames": b - a, "from": Path(p["clean"][0] and "").name or None,
                              "part_section": p["section"]})
        for f in sh.get("freezes", []):
            if fa <= f["start_frame"] < fb:
                freezes.append({**f, "start_frame": f["start_frame"] - fa + offset,
                                "end_frame": f["end_frame"] - fa + offset})
        for r in sh.get("receipt_reveal", []) or []:
            fr = r.get("frames")
            if fr and fa <= fr[0] < fb:
                reveals.append({**r, "frames": [fr[0] - fa + offset, min(fr[1], fb - 1) - fa + offset]})
        for m in (sh.get("moments") or {}).get("moments", []):
            if fa <= m["frame"] < fb:
                moments.append({**m, "frame": m["frame"] - fa + offset})
        offset += fb - fa
    for j, s in enumerate(shots):
        s.pop("from", None)
    base = parts[-1]["clean"][1]["shots"]
    timeline = parts[-1]["clean"][1]["run"]["timeline"]
    tl = json.loads(Path(timeline).read_text())
    shots_rec = {**base, "section": "all", "sections": tl["sections"], "start_s": s0, "end_s": s0 + n / FPS,
                 "n_frames": n, "shots": shots, "freezes": freezes, "receipt_reveal": reveals,
                 "moments": {"source": (base.get("moments") or {}).get("source"), "moments": moments},
                 "spliced_from": [{"section": p["section"], "frames": p["frames"],
                                   "clean": p["clean"][1]["run"]["clip"], "receipt": p["receipt"][1]["run"]["clip"]}
                                  for p in parts]}
    # receipt: the parts' own per-frame measurements, concatenated
    outputs = {}
    for look in ("clean", "receipt"):
        per = []
        for p in parts:
            lk, rec = p[look]
            fa, fb = p["frames"]
            per += rec["receipt"]["outputs"][lk]["frames"]["filmed_pct"][fa:fb]
        if len(per) != n:
            raise SystemExit(f"splice_record: REFUSED: {look}: {len(per)} per-frame shares for {n} frames")
        secs = []
        for s in tl["sections"]:
            a = max(0, round((s["start_s"] - s0) * FPS))
            b = min(n, round((s["end_s"] - s0) * FPS))
            if a < b:
                v = sum(per[a:b]) / (b - a)
                secs.append({"section": s["index"], "role": s["role"], "frames": [a, b - 1],
                             "filmed_pct": round(v, 1), "generated_pct": round(100 - v, 1)})
        tot = sum(per) / n
        outputs[look] = {"video": {"filmed_pct": round(tot, 1), "generated_pct": round(100 - tot, 1)},
                         "sections": secs, "frames": {"filmed_pct": per}}
    receipt_rec = {**parts[-1]["clean"][1]["receipt"], "start_s": s0, "receipt_reveal": reveals, "outputs": outputs}
    # run: the parts' checks, plus the joined files' own frame count and audio offset
    song = songs.pop()
    looks = {}
    for look, video in (("clean", a.clean), ("receipt", a.receipt)):
        got = frames_of(video)
        err = audio_error_ms(video, REPO / song if not Path(song).is_absolute() else song, s0)
        part_ok = all(p[look][1]["run"]["looks"][p[look][0]]["ok"] for p in parts)
        looks[look] = {"video": str(Path(video).resolve()), "frames": got, "frames_planned": n,
                       "audio_vs_master_error_ms": err, "parts_ok": part_ok,
                       "ok": part_ok and got == n and abs(err) <= 1000 / FPS,
                       "parts": [{"section": p["section"], "run": p[look][1]["run"]["looks"][p[look][0]]["video"]}
                                 for p in parts]}
    base_run = parts[-1]["clean"][1]["run"]
    billed = sum(p[k][1]["run"].get("billed_usd", 0) for p in parts for k in ("clean", "receipt"))
    run_rec = {**base_run, "clip": [p["clean"][1]["run"]["clip"] for p in parts], "section": "all",
               "timeline": timeline, "shots": str(OUT / f"{a.tag}_all_shots.json"), "looks": looks,
               "billed_usd": round(billed, 4), "spliced": True,
               "receipt": receipt_rec["outputs"]["clean"]["video"]}
    for k in ("units", "render_s", "fps", "workers", "workers_why", "stages", "total_s"):
        run_rec.pop(k, None)
    for kind, rec in (("shots", shots_rec), ("receipt", receipt_rec), ("run", run_rec)):
        Path(OUT / f"{a.tag}_all_{kind}.json").write_text(json.dumps(rec, indent=1))
    ok = all(v["ok"] for v in looks.values())
    print(f"splice_record: {a.tag}_all: {n} frames, clean filmed {outputs['clean']['video']['filmed_pct']}%, "
          f"audio {looks['clean']['audio_vs_master_error_ms']} / {looks['receipt']['audio_vs_master_error_ms']} ms, "
          f"parts ok {all(v['parts_ok'] for v in looks.values())}, ok {ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
