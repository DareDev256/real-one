"""python -m realone.director {plan, direct, review, moments, takes} ...

  plan    timeline + lyrics + Muse Wonder treatment + literal-lyrics guard -> out/<..>director_<tag>_treatment.json
  direct  plan, then the agentic loop: worlds, renders, watch, one revision pass -> ..._log.json + the renders
  review  watch one existing render against a treatment -> ..._review.json
  moments one Wonder call picks the lyric moments (ignite, slam) of a treatment -> ..._moments_treatment.json
  takes   Wonder watches several takes and picks one per section -> one edited clip + ..._takes.json

Keyed runs need MODEL_API_KEY (refused before any work without it), quote US$ and CA$ first, and refuse a ceiling
over --cap-usd (US$2) without --yes-spend. --dry-run replays hand-written FIXTURE responses (never real model
output), spends nothing, and renders with --mask local --local-plate.
"""
import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from realone.common import LOCAL_PREFIX, OUT, Refuse, file_md5, write_json
from realone.director import budget, lyrics as lyr, loop, treatment as trt
from realone.director.backend import FIXTURE_LABEL, FixtureBackend, GlimmerBackend, MetaBackend, SparkOpenBackend
from realone.director.budget import WONDER_MODEL
from realone.director.tools import Renderer, State, Tools

DEFAULT_SCRIPT = "dry_run_script.json"


def build_parser():
    p = argparse.ArgumentParser(prog="python -m realone.director",
                                description="Real One director: Muse Wonder directs from what the song means.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--song", required=True, help="the master track")
        sp.add_argument("--lyrics", help="the artist's own lyrics: one block per section, blank line between "
                                         "(outranks the transcript); start a block with [#N] to place it")
        sp.add_argument("--tag", default="main", help="output stem: out/<TEST_LOCAL_>director_<tag>_*.json")
        sp.add_argument("--dry-run", action="store_true",
                        help="no key, no spend: FIXTURE responses from tests/fixtures/director/ (never model output)")
        sp.add_argument("--script", default=DEFAULT_SCRIPT, help="FIXTURE script for --dry-run")
        sp.add_argument("--treatment", help="reuse a treatment file (re-validated and re-guarded, no model call)")
        sp.add_argument("--no-audio", action="store_true", help="do not send the song to Muse Wonder as audio")
        sp.add_argument("--no-transcribe", action="store_true", help="skip Muse Voice Transcribe")
        sp.add_argument("--effort", default="high", choices=["minimal", "low", "medium", "high", "xhigh", "max"],
                        help="Muse Wonder reasoning effort for the treatment")
        sp.add_argument("--beats-per-bar", type=int)
        sp.add_argument("--sections", metavar="MARKS", help="the artist's section starts, ROLE@TIME comma list or a file (e.g. intro@0:00,hook@0:06.9,verse@20.4): replaces the heard sections when the beat does not change between them")
        sp.add_argument("--cap-usd", type=float, default=budget.DEFAULT_CAP_USD,
                        help="refuse a run whose spend ceiling is over this (US$)")
        sp.add_argument("--yes-spend", action="store_true", help="accept a quote over --cap-usd")
        sp.add_argument("--max-calls", type=int, default=budget.DEFAULT_MAX_CALLS, help="cap on paid API calls")

    pl = sub.add_parser("plan", help="lyrics + treatment + guard, no render")
    common(pl)
    pl.add_argument("--brain", choices=["wonder", "spark-open", "glimmer"], default="wonder",
                    help="who writes the treatment: Muse Wonder (Meta Model API) or Muse Spark Open on Baseten "
                         "(open weights; needs BASETEN_API_KEY and --no-transcribe)")
    pl.add_argument("--spark-model", default=budget.SPARK_MODEL, help="Baseten model slug for --brain spark-open")
    di = sub.add_parser("direct", help="plan, then Wonder drives worlds, renders, watch, one revision pass")
    common(di)
    di.add_argument("--clip", required=True, help="the artist's phone take")
    di.add_argument("--driver", choices=["model", "code"], default="model",
                    help="model = Wonder calls the tools (default); code = our loop, Wonder decides in JSON")
    di.add_argument("--max-turns", type=int, default=8, help="cap on Wonder turns in the model driver")
    di.add_argument("--sync-near", type=float, help="song time (s) the take starts near (passed to the renderer)")
    di.add_argument("--render-minutes", type=float, default=20.0, help="per-render time limit")
    di.add_argument("--stage", action="store_true",
                    help="keyed renders through the tracked stage (the world filmed through the take's camera; "
                         "needed for a dolly move)")
    mo = sub.add_parser("moments", help="Wonder picks the lyric moments (ignite, slam) of an existing treatment")
    mo.add_argument("--treatment", required=True, help="the treatment file (its lyric and subtext are read)")
    mo.add_argument("--tag", default="main")
    mo.add_argument("--effort", default="medium", choices=["minimal", "low", "medium", "high", "xhigh", "max"])
    mo.add_argument("--dry-run", action="store_true")
    mo.add_argument("--script", default=DEFAULT_SCRIPT)
    mo.add_argument("--cap-usd", type=float, default=0.25, help="refuse a ceiling over this (US$)")
    mo.add_argument("--yes-spend", action="store_true")
    tk = sub.add_parser("takes", help="Wonder watches several takes and picks one per section; the picks are cut "
                                      "together into one edited clip for the render")
    tk.add_argument("--take", action="append", required=True, help="a phone take of the song (repeat, in order)")
    tk.add_argument("--song", required=True)
    tk.add_argument("--treatment", required=True, help="the treatment: its sections, lyric and subtext")
    tk.add_argument("--out", required=True, help="the edited clip to write (.mov)")
    tk.add_argument("--picks", help="skip Wonder: one take number per section, comma list (e.g. 2,3,2,2)")
    tk.add_argument("--tag", default="main")
    tk.add_argument("--effort", default="medium", choices=["minimal", "low", "medium", "high", "xhigh", "max"])
    tk.add_argument("--dry-run", action="store_true")
    tk.add_argument("--script", default=DEFAULT_SCRIPT)
    tk.add_argument("--cap-usd", type=float, default=0.60, help="refuse a ceiling over this (US$)")
    tk.add_argument("--yes-spend", action="store_true")
    ed = sub.add_parser("edit", help="edit by sentence: Wonder turns what the artist says into tracked effects "
                                     "(an edit file for render --edits)")
    ed.add_argument("--treatment", required=True)
    ed.add_argument("--word-times", required=True, help="scripts/align_lyrics.py output for this treatment")
    ed.add_argument("--say", required=True, help='what to add, e.g. "fire from his hands on heat"')
    ed.add_argument("--out", required=True, help="the edit file to write (.json)")
    ed.add_argument("--effort", default="medium", choices=["minimal", "low", "medium", "high", "xhigh", "max"])
    ed.add_argument("--dry-run", action="store_true")
    ed.add_argument("--script", default=DEFAULT_SCRIPT)
    ed.add_argument("--cap-usd", type=float, default=0.25, help="refuse a ceiling over this (US$)")
    ed.add_argument("--yes-spend", action="store_true")
    rv = sub.add_parser("review", help="watch one render against its treatment")
    rv.add_argument("--render", required=True, help="an MP4 under out/")
    rv.add_argument("--treatment", required=True, help="the treatment file it was made from")
    rv.add_argument("--section", type=int, required=True)
    rv.add_argument("--tag", default="main")
    rv.add_argument("--dry-run", action="store_true")
    rv.add_argument("--script", default=DEFAULT_SCRIPT)
    rv.add_argument("--cap-usd", type=float, default=budget.DEFAULT_CAP_USD)
    rv.add_argument("--yes-spend", action="store_true")
    rv.add_argument("--max-calls", type=int, default=budget.DEFAULT_MAX_CALLS)
    return p


def _exists(*paths):
    for p in paths:
        if p and not Path(p).exists():
            raise Refuse(f"not found: {p}")


def _meta(a, mode_key):
    m = {"director": "realone.director", "created": dt.datetime.now().isoformat(timespec="seconds"),
         "mode": "DRY RUN" if a.dry_run else "keyed", "song": str(a.song) if hasattr(a, "song") else None}
    if a.dry_run:
        m["FIXTURE"] = FIXTURE_LABEL
    elif getattr(a, "brain", "wonder") == "spark-open":
        m["model"] = f"{a.spark_model} (Baseten)"
    elif getattr(a, "brain", "wonder") == "glimmer":
        m["model"] = f"{budget.GLIMMER_MODEL} ({budget.GLIMMER_HOST})"
    else:
        m["model"] = WONDER_MODEL
    m.update(mode_key)
    return m


def _song_mp3(song, work):
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"song_{file_md5(song)[:12]}.mp3"
    if not out.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(song), "-ac", "1", "-b:a", "96k",
                        "-map_metadata", "-1", str(out)], check=True)
    return out


def _print_treatment(rec):
    print(f"song subtext: {rec['song_subtext']}")
    for r in rec["sections"]:
        g = r["guard"]
        first = r.get("guard_first_pass")
        flag = (f"  (first pass REJECTED {first['score']:.2f}: {', '.join(first['concrete'] or first['shared'])})"
                if first else "")
        print(f"  #{r['index']} {r['role']:<7s} {r['look']:<8s} {'FREEZE' if r['freeze'] else '      '} "
              f"{r['camera_energy']:<4s} guard {g['verdict']:<8s} {g['score']:.2f}{flag}")
        print(f"      subtext: {r['subtext']}")
        print(f"      world:   {r['world_prompt']}")


def plan_stage(a, backend, meter, prefix, timeline):
    """lyrics + treatment -> (record, path, lyrics, result)."""
    stem = OUT / f"{prefix}director_{a.tag}"
    artist, warns = (lyr.read_lyrics_file(a.lyrics, timeline["sections"]) if a.lyrics else (None, []))
    for w in warns:
        print(f"director: lyrics warning: {w}")
    transcript = None
    if not a.no_transcribe:
        kw = lyr.keywords_from(artist) if artist else None
        transcript = lyr.transcribe_song(backend, a.song, OUT / "work" / "director", keywords=kw)
        print(f"director: transcript from {transcript['source']}: {len(transcript['turns'])} turns")
    ly = lyr.build(timeline["sections"], artist, transcript)
    ly["warnings"] = warns
    write_json(f"{stem}_lyrics.json", {**_meta(a, {}), **ly})
    if a.treatment:
        _exists(a.treatment)
        res = trt.from_record(json.loads(Path(a.treatment).read_text()), timeline, ly)
        print(f"director: treatment from {a.treatment} (re-validated, re-guarded, no model call)")
    else:
        fid = None
        if not a.no_audio:
            fid = backend.upload(_song_mp3(a.song, OUT / "work" / "director"), "song audio")
        res = trt.make(backend, timeline, ly, audio_file_id=fid, effort=a.effort)
    rec = trt.record(res, timeline, ly, _meta(a, {"timeline": str(stem) + "_timeline.json",
                                                   "lyrics": str(stem) + "_lyrics.json",
                                                   "spend": meter.summary() if meter else None}))
    path = write_json(f"{stem}_treatment.json", rec)
    return rec, path, ly, res


def edit_cmd(a):
    """One Wonder call: a sentence -> tracked effects placed on the sung words -> an edit file."""
    from realone import fx as FX
    from realone.director import edit as ED
    _exists(a.treatment)
    rec = json.loads(Path(a.treatment).read_text())
    if not rec.get("sections"):
        raise Refuse(f"{a.treatment}: not a treatment (no sections)")
    words = ED.load_word_times(a.word_times)
    if a.dry_run:
        backend, meter = FixtureBackend(a.script), None
    else:
        key = budget.require_key()
        est = len(json.dumps(rec["sections"])) // 4 + 900
        q = {"lines": [], "ceiling_usd": budget.wonder_ceiling("edit", est)}
        q["ceiling_cad"] = round(q["ceiling_usd"] * budget.USD_TO_CAD, 2)
        print(f"director: edit quote: up to US${q['ceiling_usd']:.3f} (about CA${q['ceiling_cad']:.2f}), one call")
        meter = budget.Meter(budget.gate({**q, "expected_usd": q["ceiling_usd"]}, a.cap_usd, a.yes_spend), 1)
        backend = MetaBackend(key, meter)
    out = ED.edit(backend, rec, words, a.say, sorted(FX.EFFECTS), effort=a.effort)
    out["spend"] = meter.summary() if meter else {"spent_usd": 0.0}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for e in out["fx"]:
        print(f"  {e['effect']} on '{e['track']}' at {e['at_s']:.2f} s ('{e['word']}', section {e['section']}): {e['why']}")
    for d in out["dropped"]:
        print(f"  dropped: {d['why']}")
    print(f"director: edit [{out['source']}] -> {a.out}; render it with --edits {a.out}")
    return 0


def takes_cmd(a):
    """Several takes -> one edited clip: sync each take, Wonder picks a take per section (or --picks), check the
    phone did not move between takes, cut on the section downbeats. Writes the clip and <tag>_takes.json."""
    import tempfile
    from realone.director import takes as TK
    for p in [a.song, a.treatment, *a.take]:
        _exists(p)
    rec = json.loads(Path(a.treatment).read_text())
    secs = sorted(rec.get("sections") or [], key=lambda s: s["start_s"])
    if not secs:
        raise Refuse(f"{a.treatment}: not a treatment (no sections)")
    spans = TK.spans(a.take, a.song)
    for k, sp in enumerate(spans):
        print(f"director: take {k + 1}: song {sp['song_start_s']:.2f}-{sp['song_end_s']:.2f} s "
              f"(sync r={sp['sync']['r_peak']:.2f}, margin {sp['sync']['margin']:.2f})")
    ref = spans[0]
    for k, sp in enumerate(spans[1:], 2):
        lo = max(ref["song_start_s"], sp["song_start_s"], 0.0) + 1.0
        hi = min(ref["song_end_s"], sp["song_end_s"]) - 1.0
        moments = [lo + (hi - lo) * f for f in (0.2, 0.5, 0.8)]  # the same song moments in both takes
        shift = sorted(TK.background_shift(ref["path"], sp["path"], t - ref["offset_s"], t - sp["offset_s"])
                       for t in moments)[1]
        if shift > TK.MAX_SHIFT_PX:
            raise Refuse(f"takes: the phone moved between take 1 and take {k} ({shift:.0f} px at 1080 wide); the "
                         f"tracked stage reads one camera. Cut these takes as separate videos.")
        print(f"director: take {k} background vs take 1: {shift:.1f} px")
    meter = None
    if a.picks:
        nums = [int(x) for x in a.picks.split(",")]
        if len(nums) != len(secs):
            raise Refuse(f"--picks has {len(nums)} numbers for {len(secs)} sections")
        obj = {"picks": [{"section": s["index"], "take": n, "why": "--picks (by hand)"} for s, n in zip(secs, nums)]}
        picks, info = TK.check(obj, secs, spans), {"source": "--picks (by hand)", **obj}
    else:
        if a.dry_run:
            backend, meter = FixtureBackend(a.script), None
        else:
            key = budget.require_key()
            song_end = max(s["end_s"] for s in secs)
            est = 2500 + int(len(spans) * song_end * budget.VIDEO_TOKENS_PER_S)
            q = {"lines": [], "ceiling_usd": budget.wonder_ceiling("takes", est)}
            q["ceiling_cad"] = round(q["ceiling_usd"] * budget.USD_TO_CAD, 2)
            print(f"director: takes quote: up to US${q['ceiling_usd']:.3f} (about CA${q['ceiling_cad']:.2f}), one "
                  f"call with {len(spans)} videos")
            # one Wonder call; each take's upload counts against the call cap too (it costs nothing)
            meter = budget.Meter(budget.gate({**q, "expected_usd": q["ceiling_usd"]}, a.cap_usd, a.yes_spend),
                                 1 + len(spans))
            backend = MetaBackend(key, meter)
        with tempfile.TemporaryDirectory() as work:
            picks, info = TK.pick(backend, spans, rec, work, effort=a.effort)
    segs = TK.segments(secs, picks, spans, max(s["end_s"] for s in secs))
    for (k, c0, c1, s0), sec in zip(segs, secs):
        why = next((p["why"] for p in info.get("picks", []) if p["section"] == sec["index"]), "")
        print(f"  #{sec['index']} {sec.get('role', ''):6s} take {k + 1}  (clip {c0:.2f}-{c1:.2f} s)  {why}")
    prefix = LOCAL_PREFIX if a.dry_run else ""
    OUT.mkdir(exist_ok=True)
    # the record first: a paid pick is kept even if the cut fails
    path = write_json(OUT / f"{prefix}director_{a.tag}_takes.json",
                      {"takes": spans, "picks": info, "segments": [
                          {"section": sec["index"], "take": k + 1, "clip_start_s": c0, "clip_end_s": c1,
                           "song_start_s": s0} for (k, c0, c1, s0), sec in zip(segs, secs)],
                       "spend": meter.summary() if not a.picks and meter else {"spent_usd": 0.0},
                       "out": str(a.out)})
    print(f"director: record -> {path}")
    TK.assemble(spans, segs, a.out)
    print(f"director: edited clip -> {a.out}")
    return 0


def moments_cmd(a):
    """One Wonder call: the treatment's lyric moments. Writes a copy of the treatment with `moments` per section."""
    from realone.director import moments as dmo
    _exists(a.treatment)
    rec = json.loads(Path(a.treatment).read_text())
    if not rec.get("sections"):
        raise Refuse(f"{a.treatment}: not a treatment (no sections)")
    if a.dry_run:
        backend, meter = FixtureBackend(a.script), None
        print("director: DRY RUN moments (FIXTURE reply, no call)")
    else:
        key = budget.require_key()
        est = len(json.dumps(dmo.brief(rec))) // 4 + 600
        q = {"lines": [], "ceiling_usd": budget.wonder_ceiling("moments", est)}
        q["ceiling_cad"] = round(q["ceiling_usd"] * budget.USD_TO_CAD, 2)
        print(f"director: moments quote: up to US${q['ceiling_usd']:.3f} (about CA${q['ceiling_cad']:.2f}), one call")
        meter = budget.Meter(budget.gate({**q, "expected_usd": q["ceiling_usd"]}, a.cap_usd, a.yes_spend), 1)
        backend = MetaBackend(key, meter)
    out = dmo.pick(backend, rec, effort=a.effort)
    prefix = LOCAL_PREFIX if a.dry_run else ""
    out["moments_pick"]["from_treatment"] = str(a.treatment)
    out["moments_pick"]["spend"] = meter.summary() if meter else {"spent_usd": 0.0}
    path = write_json(OUT / f"{prefix}director_{a.tag}_moments_treatment.json", out)
    for r in out["sections"]:
        for m in r["moments"]:
            print(f"  #{r['index']} {m['effect']:<6s} {m['word']!r}: {m.get('why', '')}")
    for d in out["moments_pick"]["dropped"]:
        print(f"  dropped {d['moment'].get('word')!r}: {d['why']}")
    print(f"director: {out['moments_pick']['kept']} moments [{out['moments_pick']['source']}] -> {path}")
    return 0


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "review":
            return review(a)
        if a.cmd == "moments":
            return moments_cmd(a)
        if a.cmd == "takes":
            return takes_cmd(a)
        if a.cmd == "edit":
            return edit_cmd(a)
        _exists(a.song, a.lyrics, getattr(a, "clip", None))
        brain = getattr(a, "brain", "wonder")
        if brain == "spark-open" and not a.no_transcribe:
            raise Refuse("--brain spark-open needs --no-transcribe (Baseten has no transcription endpoint); pass "
                         "--lyrics for the words. Nothing was called.")
        if brain == "glimmer" and not (a.no_transcribe and a.no_audio):
            raise Refuse("--brain glimmer needs --no-transcribe and --no-audio (Glimmer reads text and images, not "
                         "audio); pass --lyrics for the words. Nothing was called.")
        glimmer_url = None
        if a.dry_run:
            key = None
        elif brain == "glimmer":  # before any work
            key, glimmer_url = None, budget.require_glimmer_url()
        else:
            key = budget.require_baseten_key() if brain == "spark-open" else budget.require_key()
        prefix = LOCAL_PREFIX if a.dry_run else ""
        OUT.mkdir(exist_ok=True)
        stem = OUT / f"{prefix}director_{a.tag}"
        from realone.audio import analyze
        timeline = analyze(a.song, a.beats_per_bar, getattr(a, "sections", None))
        write_json(f"{stem}_timeline.json", timeline)
        print(f"director: {timeline['bpm']:.1f} BPM, {len(timeline['sections'])} sections: " +
              ", ".join(f"#{s['index']} {s['role']}{'*' if s.get('is_hook') else ''}" for s in timeline["sections"]))
        covered, sync, clip_s = [], None, None
        if a.cmd == "direct":
            covered, sync, clip_s = loop.coverage(timeline, a.clip, a.song, a.sync_near)
            print(f"director: take synced at {sync['offset_ms']:+.0f} ms, covers sections {covered or 'none'}")
            if not covered:
                raise Refuse("the phone take covers no whole section of the song; nothing can be rendered")
        brief_chars = len(json.dumps(trt.brief(timeline, {"sections": []}))) + 2500
        q = budget.quote(song_s=timeline["duration_s"], n_sections=len(timeline["sections"]),
                         brief_chars=brief_chars, audio_in=not a.no_audio and not a.treatment,
                         transcribe=not a.no_transcribe, direct=a.cmd == "direct",
                         covered_s=[s["end_s"] - s["start_s"] for s in timeline["sections"]
                                    if s["index"] in covered],
                         max_turns=getattr(a, "max_turns", 8), driver=getattr(a, "driver", "model"), brain=brain)
        if a.treatment:
            q["lines"] = [x for x in q["lines"] if not x["what"].startswith(f"{budget.BRAIN_NAME[brain]} treatment")
                          and "repair" not in x["what"] and "re-ask" not in x["what"]]
            for k in ("ceiling", "expected"):
                q[f"{k}_usd"] = round(sum(x[f"{k}_usd"] for x in q["lines"]), 4)
                q[f"{k}_cad"] = round(q[f"{k}_usd"] * budget.USD_TO_CAD, 2)
        if a.dry_run:
            print(budget.format_quote(q, "director: DRY RUN. A keyed run would quote:"))
            print("director: DRY RUN spend: US$0.00 (FIXTURE responses, no API calls)")
            backend, meter = FixtureBackend(a.script), None
        else:
            print(budget.format_quote(q, "director: spend quote before any call:"))
            cap = budget.gate(q, a.cap_usd, a.yes_spend)
            meter = budget.Meter(cap, a.max_calls)
            backend = (SparkOpenBackend(key, meter, a.spark_model) if brain == "spark-open"
                       else GlimmerBackend(meter, glimmer_url) if brain == "glimmer" else MetaBackend(key, meter))
        rec, tpath, ly, res = plan_stage(a, backend, meter, prefix, timeline)
        print(f"director: treatment -> {tpath}" + ("   [FIXTURE]" if a.dry_run else ""))
        _print_treatment(rec)
        still = rec["still_literal"]
        if a.cmd == "plan":
            if meter:
                print(f"director: spent US${meter.spent:.4f} (about CA${meter.spent * budget.USD_TO_CAD:.2f}) in "
                      f"{meter.calls} calls")
            if still:
                print(f"director: REFUSED: sections {still} are still literal after the one re-ask; edit their "
                      f"world_prompt in {tpath} and re-run with --treatment", file=sys.stderr)
                return 3
            return 0
        if still:
            raise Refuse(f"sections {still} are still literal after the one re-ask; edit their world_prompt in "
                         f"{tpath} and re-run direct with --treatment {tpath}")
        state = State(timeline=timeline, lyrics=lyr.lines_by_index(ly), treatment=res["treatment"],
                      covered=covered, tag=f"director_{a.tag}", dry_run=a.dry_run)
        renderer = Renderer(a.clip, a.song, dry_run=a.dry_run, max_minutes=a.render_minutes, sync_near=a.sync_near,
                            sections=getattr(a, "sections", None),
                            stage=getattr(a, "stage", False))
        tools = Tools(backend, state, renderer, meter)
        print(f"director: driver = {a.driver} ({'Muse Wonder calls the tools' if a.driver == 'model' else 'our loop'}"
              f"{'; FIXTURE turns' if a.dry_run else ''})")
        try:
            end = loop.run_model(backend, tools, state, a.max_turns) if a.driver == "model" else \
                loop.run_code(backend, tools, state)
        finally:
            log = {**_meta(a, {"treatment": str(tpath), "driver": a.driver, "clip": a.clip, "sync": sync,
                               "covered": covered}),
                   "ended": state.finished, "renders": state.renders, "watches": state.watches,
                   "revision": state.revision, "final_treatment": state.treatment, "steps": state.log,
                   "spend": meter.summary() if meter else {"spent_usd": 0.0, "note": "DRY RUN, FIXTURE"}}
            lpath = write_json(f"{stem}_log.json", log)
        print(f"\n=== director: {end} ===")
        for i, recs in sorted(state.renders.items()):
            for r in recs:
                print(f"  section {i} [{r['look']}] {r['video']}  cuts {r['measured']['cuts_on_plan']}, audio "
                      f"{r['measured']['audio_vs_master_error_ms']} ms, freeze: {r['freeze']}")
        for w in state.watches:
            print(f"  watched section {w['section']}: {w['verdict']} [{w['source']}]")
        for c in state.revision["changes"]:
            print(f"  revised section {c['section']} {c['field']}: {str(c['old'])[:50]} -> {str(c['new'])[:50]}")
        print(f"  summary: {state.finished}")
        print(f"  log -> {lpath}")
        if meter:
            print(f"  spent US${meter.spent:.4f} (about CA${meter.spent * budget.USD_TO_CAD:.2f}) in {meter.calls} "
                  f"calls; cap US${meter.cap_usd:.2f}")
        return 0
    except Refuse as e:
        print(f"realone.director: REFUSED: {e}", file=sys.stderr)
        return e.exit_code


def review(a):
    _exists(a.render, a.treatment)
    path = Path(a.render).resolve()
    if OUT.resolve() not in path.parents:
        raise Refuse("review only uploads renders under out/ (it never sends an arbitrary file)")
    key = None if a.dry_run else budget.require_key()
    rec = json.loads(Path(a.treatment).read_text())
    row = next((r for r in rec["sections"] if r["index"] == a.section), None)
    if row is None:
        raise Refuse(f"section {a.section} is not in {a.treatment}")
    dur = row["end_s"] - row["start_s"]
    if a.dry_run:
        backend, meter = FixtureBackend(a.script), None
        print("director: DRY RUN review (FIXTURE notes, no upload)")
    else:
        q = {"lines": [], "ceiling_usd": budget.wonder_ceiling("watch", 2000 + int(dur * budget.VIDEO_TOKENS_PER_S))}
        q["ceiling_cad"] = round(q["ceiling_usd"] * budget.USD_TO_CAD, 2)
        print(f"director: review quote: up to US${q['ceiling_usd']:.3f} (about CA${q['ceiling_cad']:.2f}); video "
              f"tokens ASSUMED {budget.VIDEO_TOKENS_PER_S}/s")
        meter = budget.Meter(budget.gate({**q, "expected_usd": q["ceiling_usd"]}, a.cap_usd, a.yes_spend),
                             a.max_calls)
        backend = MetaBackend(key, meter)
    secs = [{"index": r["index"], "role": r["role"], "is_hook": r["is_hook"], "start_s": r["start_s"],
             "end_s": r["end_s"]} for r in rec["sections"]]
    tr = {"song_subtext": rec["song_subtext"],
          "sections": [{k: r[k] for k in ("index", "subtext", "world_prompt", "palette", "camera_energy", "look",
                                           "freeze", "why")} for r in rec["sections"]]}
    run_json = next(iter(sorted(OUT.glob(path.name.rsplit("_", 1)[0] + "_run.json"))), None)
    measured = {}
    if run_json:
        lk = json.loads(run_json.read_text())["looks"].get(row["look"], {})
        measured = {k: lk.get(k) for k in ("spec_ok", "cuts_on_plan", "audio_vs_master_error_ms")}
    state = State(timeline={"sections": secs}, lyrics={r["index"]: r["lyric"] for r in rec["sections"]},
                  treatment=tr, covered=[a.section], tag=f"director_{a.tag}", dry_run=a.dry_run,
                  renders={a.section: [{"section": a.section, "video": str(path), "look": row["look"],
                                        "freeze": "as rendered", "duration_s": round(dur, 3), "measured": measured}]})
    tools = Tools(backend, state, renderer=None, meter=meter)
    out = tools.dispatch("watch", {"render_path": str(path)})
    prefix = LOCAL_PREFIX if a.dry_run else ""
    rp = write_json(OUT / f"{prefix}director_{a.tag}_review.json",
                    {**_meta(a, {"render": str(path), "treatment": a.treatment, "section": a.section}),
                     "result": out, "spend": meter.summary() if meter else {"spent_usd": 0.0}})
    if not out.get("ok"):
        raise Refuse(f"review failed: {out.get('error')}")
    print(f"director: {out['verdict']} [{out['source']}]")
    for n in out["notes"]:
        print(f"  {n['t_s']:6.2f}s {n['against']:<8s} {n['severity']:<4s} {n['note']}")
    print(f"  suggestion: {out['suggestion']['field']} -> {out['suggestion']['value'][:120]}")
    print(f"  review -> {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
