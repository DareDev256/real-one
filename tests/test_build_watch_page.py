"""scripts/build_watch_page.py: the refusals and the TEST banner. No ffmpeg: the encoder, prober and poster are fakes."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from realone.common import Refuse

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("build_watch_page", REPO / "scripts" / "build_watch_page.py")
bw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bw)

N = 12


def make_render(repo, stem="TEST_LOCAL_t_hook", local=True, plates=True, looks=("collage", "receipt"), n=N,
                song="footage/s.wav"):
    out = repo / "out"
    out.mkdir(parents=True, exist_ok=True)
    vids = {}
    for lk in looks:
        v = out / f"{stem}_{lk}.mp4"
        v.write_bytes(b"not really a video")
        vids[lk] = {"video": str(v)}
    sec = {"index": 2, "role": "hook", "start_s": 30.0, "end_s": 50.0, "start_bar": 12, "end_bar": 20}
    tl = out / f"{stem}_timeline.json"
    tl.write_text(json.dumps({"song": song, "bpm": 96.0, "beats_per_bar": 4, "sections": [sec]}))
    run = {"version": "0.2.0", "mode": "local test" if local else "demo",
           "engine": "Apple Vision (VNGenerateForegroundInstanceMaskRequest), TEST ONLY" if local else "SAM 3.1",
           "song": song, "looks": vids, "timeline": str(tl), "billed_usd": 0.0}
    shots = {"start_s": 30.0, "end_s": 30.0 + n / 30, "n_frames": n, "fps": 30, "start_bar": 12, "end_bar": 20,
             "bpm": 96.0, "beats_per_bar": 4, "sections": [sec], "shots": [{"start_frame": 0, "bar": 12}],
             "freezes": [{"start_frame": 0, "end_frame": 6, "frames": 6, "start_bar": 12, "end_bar": 14,
                          "section_index": 2}],
             "world_sources": [{"section": 2, "from": "local plates (TEST ONLY)" if plates else "Muse Image"}]}
    receipt = {"fps": 30, "outputs": {lk: {"video": {"filmed_pct": 20.0, "generated_pct": 80.0},
                                           "frames": {"filmed_pct": [20.0] * n}} for lk in looks}}
    for suffix, doc in (("run", run), ("shots", shots), ("receipt", receipt)):
        (out / f"{stem}_{suffix}.json").write_text(json.dumps(doc))
    return out


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "footage" / "own").mkdir(parents=True)
    web = r / "web"
    web.mkdir()
    for f in bw.PAGE_FILES:
        shutil.copy2(REPO / "web" / f, web / f)
    return r


def fake_encode(src, dst, audio=True):
    shutil.copy2(src, dst)


def fake_probe(frames=N):
    return lambda path: {"codec": "h264", "width": 1080, "height": 1920, "pix_fmt": "yuv420p", "frames": frames,
                         "faststart": True}


def build(repo, tag="t", section="hook", prober=None, **kw):
    return bw.build(tag, section, out=repo / "web" / "dist", repo=repo, web=repo / "web", encoder=fake_encode,
                    prober=prober or fake_probe(), posterer=lambda s, d, t: Path(d).write_bytes(b"jpg"),
                    log=lambda *a: None, **kw)


def treatment_file(repo, song="footage/s.wav", start=30.0, fixture=True, name="tr_treatment.json"):
    doc = {"song": song, "mode": "DRY RUN" if fixture else "KEYED",
           "sections": [{"index": 2, "role": "hook", "start_s": start, "end_s": 50.0, "lyric": ["a line"],
                         "subtext": "s", "world_prompt": "w"}]}
    if fixture:
        doc["FIXTURE"] = "FIXTURE (hand-written test response)"
    p = repo / "out" / name
    p.write_text(json.dumps(doc))
    return p


def real_credits(repo, status="REAL", ran=True):
    p = repo / "out" / "credits.json"
    p.write_text(json.dumps({"status": status, "statement": "s", "models": [
        {"name": "SAM 3.1", "did": "cut-out", "ran": ran, "calls": 3, "frames": 600, "usd": 0.12}]}))
    return p


# ---------------------------------------------------------------- refusals

def test_refuses_when_no_render(repo):
    (repo / "out").mkdir()
    with pytest.raises(Refuse, match="no render for tag"):
        build(repo)


@pytest.mark.parametrize("suffix", ["shots", "receipt"])
def test_refuses_each_missing_json(repo, suffix):
    out = make_render(repo)
    (out / f"TEST_LOCAL_t_hook_{suffix}.json").unlink()
    with pytest.raises(Refuse, match=f"missing {suffix} json"):
        build(repo)


def test_refuses_missing_timeline_named_by_run(repo):
    out = make_render(repo)
    (out / "TEST_LOCAL_t_hook_timeline.json").unlink()
    with pytest.raises(Refuse, match="missing timeline json"):
        build(repo)


def test_refuses_render_without_receipt_look(repo):
    make_render(repo, looks=("collage",))
    with pytest.raises(Refuse, match="no receipt look"):
        build(repo)


def test_refuses_missing_video(repo):
    out = make_render(repo)
    (out / "TEST_LOCAL_t_hook_receipt.mp4").unlink()
    with pytest.raises(Refuse, match="missing receipt video"):
        build(repo)


def test_refuses_tag_matching_real_and_test_render(repo):
    make_render(repo, stem="TEST_LOCAL_t_hook")
    make_render(repo, stem="t_hook", local=False, plates=False)
    with pytest.raises(Refuse, match="matches a real render and a test render"):
        build(repo)


def test_refuses_videos_with_different_frame_counts(repo):
    make_render(repo)
    counts = iter([N, N - 1])
    with pytest.raises(Refuse, match="different frame counts"):
        build(repo, prober=lambda p: dict(fake_probe()(p), frames=next(counts)))


def test_refuses_receipt_that_does_not_cover_every_frame(repo):
    out = make_render(repo)
    p = out / "TEST_LOCAL_t_hook_receipt.json"
    doc = json.loads(p.read_text())
    doc["outputs"]["receipt"]["frames"]["filmed_pct"] = [20.0] * (N - 3)
    p.write_text(json.dumps(doc))
    with pytest.raises(Refuse, match="the receipt has 9 frames for receipt"):
        build(repo)


def test_refuses_video_not_fit_for_web(repo):
    make_render(repo)
    with pytest.raises(Refuse, match="no faststart"):
        build(repo, prober=lambda p: dict(fake_probe()(p), faststart=False))


# ---------------------------------------------------------------- footage/own

def test_refuses_treatment_under_footage_own(repo):
    make_render(repo)
    t = repo / "footage" / "own" / "x_treatment.json"
    t.write_text("{}")
    with pytest.raises(Refuse, match="footage/own"):
        build(repo, treatment_path=t)


def test_refuses_out_under_footage_own(repo):
    make_render(repo)
    with pytest.raises(Refuse, match="footage/own"):
        bw.build("t", "hook", out=repo / "footage" / "own" / "dist", repo=repo, web=repo / "web",
                 encoder=fake_encode, prober=fake_probe(), posterer=lambda *a: None, log=lambda *a: None)


def test_refuses_video_symlinked_into_footage_own(repo):
    out = make_render(repo)
    real = repo / "footage" / "own" / "take.mp4"
    real.write_bytes(b"the artist's own take")
    link = out / "TEST_LOCAL_t_hook_collage.mp4"
    link.unlink()
    link.symlink_to(real)
    with pytest.raises(Refuse, match="footage/own"):
        build(repo)


def test_refuses_footage_own_in_other_case(repo):
    with pytest.raises(Refuse, match="footage/own"):
        bw.refuse_own(repo / "footage" / "OWN" / "x.json", repo)
    bw.refuse_own(repo / "footage" / "owner.json", repo)   # a sibling name is not under it


# ---------------------------------------------------------------- treatment

def test_refuses_treatment_for_another_song(repo):
    make_render(repo)
    with pytest.raises(Refuse, match="the treatment is for"):
        build(repo, treatment_path=treatment_file(repo, song="footage/other.wav"))


def test_refuses_treatment_with_other_section_timing(repo):
    make_render(repo)
    with pytest.raises(Refuse, match="not the same cut of the song"):
        build(repo, treatment_path=treatment_file(repo, start=31.0))


def test_treatment_that_lines_up_is_copied(repo):
    make_render(repo)
    out, m = build(repo, treatment_path=treatment_file(repo))
    assert m["files"]["treatment"] == "treatment.json"
    assert json.loads((out / "data" / "treatment.json").read_text())["sections"][0]["subtext"] == "s"


# ---------------------------------------------------------------- TEST banner + credits

def test_reasons_each_trigger_on_its_own():
    real_run = {"mode": "demo", "engine": "SAM 3.1"}
    real_shots = {"world_sources": [{"from": "Muse Image"}]}
    assert bw.test_reasons("t_hook", real_run, real_shots) == []
    assert bw.test_reasons("TEST_LOCAL_t_hook", real_run, real_shots) == ["Tag TEST_LOCAL_"]
    assert len(bw.test_reasons("t_hook", {"mode": "local test", "engine": "x"}, real_shots)) == 1
    assert len(bw.test_reasons("t_hook", {"mode": "demo", "engine": "Apple Vision"}, real_shots)) == 1
    assert len(bw.test_reasons("t_hook", real_run, {"world_sources": [{"from": "local plates (TEST ONLY)"}]})) == 1
    assert len(bw.test_reasons("t_hook", real_run, real_shots, {"FIXTURE": "yes"})) == 1
    assert len(bw.test_reasons("t_hook", real_run, real_shots, {"mode": "DRY RUN"})) == 1


def test_test_render_gets_banner_in_html_and_test_credits(repo):
    make_render(repo)
    out, m = build(repo)
    html = (out / "index.html").read_text()
    assert 'id="test-banner"' in html and "TEST RENDER" in html
    assert 'data-test="1"' in html and bw.BANNER_MARK not in html
    assert m["test"] is True and m["test_reasons"]
    c = json.loads((out / "data" / "credits.json").read_text())
    assert c["status"] == "TEST" and not any(x["ran"] for x in c["models"])
    assert c["total_usd"] == 0.0


def test_local_masks_alone_put_the_banner_up(repo):
    make_render(repo, stem="t_hook", local=True, plates=False)   # tag has no TEST_LOCAL_ prefix
    out, m = build(repo)
    assert 'id="test-banner"' in (out / "index.html").read_text()
    assert m["test_reasons"] == ["Masks: Apple Vision on a Mac, not SAM 3.1"]


def test_real_render_has_no_banner(repo):
    make_render(repo, stem="t_hook", local=False, plates=False)
    out, m = build(repo, credits_path=real_credits(repo))
    html = (out / "index.html").read_text()
    assert 'id="test-banner"' not in html and 'data-test="0"' in html and bw.BANNER_MARK not in html
    assert m["test"] is False


def test_real_render_without_credits_refuses(repo):
    make_render(repo, stem="t_hook", local=False, plates=False)
    with pytest.raises(Refuse, match="needs --credits"):
        build(repo)


def test_test_render_with_credits_claiming_a_model_ran_refuses(repo):
    make_render(repo)
    with pytest.raises(Refuse, match="claim SAM 3.1 ran"):
        build(repo, credits_path=real_credits(repo, status="TEST", ran=True))


def test_test_render_with_real_credits_refuses(repo):
    make_render(repo)
    with pytest.raises(Refuse, match="must say TEST"):
        build(repo, credits_path=real_credits(repo, status="REAL", ran=False))


def test_template_without_banner_marker_refuses(repo):
    make_render(repo)
    p = repo / "web" / "index.html"
    p.write_text(p.read_text().replace(bw.BANNER_MARK, ""))
    with pytest.raises(Refuse, match="TEST banner would have nowhere to go"):
        build(repo)


def test_out_dir_not_made_by_the_script_is_left_alone(repo):
    make_render(repo)
    dist = repo / "web" / "dist"
    dist.mkdir()
    (dist / "keep.txt").write_text("mine")
    with pytest.raises(Refuse, match="was not made by this script"):
        build(repo)
    assert (dist / "keep.txt").read_text() == "mine"


def test_local_paths_are_stripped(repo):
    make_render(repo)
    out, _ = build(repo)
    for f in (out / "data").glob("*.json"):
        assert str(repo) not in f.read_text() and "/Users/" not in f.read_text(), f.name


def test_a_test_render_credits_exactly_the_models_that_ran(repo):
    """Test masks, but the world is real Muse Image output and the treatment a keyed Muse Wonder run: the page
    says so (banner and credits), and still credits SAM 3.1 as stood in for."""
    make_render(repo, plates=False)
    shots_p = repo / "out" / "TEST_LOCAL_t_hook_shots.json"
    shots = json.loads(shots_p.read_text())
    shots["world_sources"] = [{"section": 2, "worlds": ["Vertical 9:16, deserted park"], "from": "#2"}]
    shots_p.write_text(json.dumps(shots))
    (repo / "out" / "worlds").mkdir(parents=True, exist_ok=True)
    (repo / "out" / "worlds" / "muse_abc.json").write_text(json.dumps(
        {"prompt": "Vertical 9:16, deserted parking structure. Light: soft", "model": "muse-image-1.0"}))
    tp = treatment_file(repo, fixture=False)
    t = json.loads(tp.read_text())
    t["mode"] = "keyed"
    t["spend"] = {"ledger": [{"kind": "voice:transcribe", "usd": 0.0045}, {"kind": "wonder:treatment", "usd": 0.1}]}
    tp.write_text(json.dumps(t))
    out, m = build(repo, treatment_path=tp)
    c = json.loads((out / "data" / "credits.json").read_text())
    ran = {x["name"]: x["ran"] for x in c["models"]}
    assert ran == {"SAM 3.1": False, "Voice Transcribe": True, "Muse Wonder": True, "Muse Image": True}
    assert next(x for x in c["models"] if x["name"] == "Muse Wonder")["usd"] == 0.1
    img = next(x for x in c["models"] if x["name"] == "Muse Image")
    assert (img["calls"], img["frames"], img["usd"]) == (1, None, 0.01)       # one world, priced per image
    assert next(x for x in c["models"] if x["name"] == "SAM 3.1")["calls"] == 0  # stood in: nothing ran
    html = (out / "index.html").read_text()
    assert ("Real Meta model output on this page: Muse Wonder (the direction and its review), Muse Image (the "
            "world).") in html                                          # Voice ran but no transcript is shown
    assert "No Meta model ran" not in html
    voice = next(x for x in c["models"] if x["name"] == "Voice Transcribe")
    assert voice["ran"] and voice["output"] is False and "heard no words" in voice["note"]
    assert "Voice Transcribe ran and returned nothing shown here" in c["statement"]
    (repo / "out" / "tr_lyrics.json").write_text(json.dumps(
        {"sections": [{"index": 2, "lines": ["a line"], "heard": ["a line"]}]}))
    out, m = build(repo, treatment_path=tp)
    html = (out / "index.html").read_text()
    assert "Voice Transcribe (the lyric timing), Muse Wonder" in html     # words heard: its output is on the page


def test_a_dry_run_treatment_is_never_credited_as_a_model_run(repo):
    make_render(repo)
    tp = treatment_file(repo, fixture=True)
    t = json.loads(tp.read_text())
    t["spend"] = {"ledger": [{"kind": "wonder:treatment", "usd": 0.0}]}   # a dry run can log its fixture calls
    tp.write_text(json.dumps(t))
    out, m = build(repo, treatment_path=tp)
    c = json.loads((out / "data" / "credits.json").read_text())
    assert not next(x for x in c["models"] if x["name"] == "Muse Wonder")["ran"]


def test_credits_count_what_ran_and_say_n_a_for_what_nobody_measured(repo):
    """The Nov 1 live page showed CALLS 0 and FRAMES 0 under models it marked RAN: YES (a blind critic read that as
    the Meta-tooling claim failing). Counts now come from the evidence: SAM 3.1 one request and one mask per frame
    on the page, priced per 1,000 frames; Wonder the ledger's calls; anything unmeasured is None (n/a), never 0."""
    make_render(repo, local=False, plates=False)
    tp = treatment_file(repo, fixture=False)
    t = json.loads(tp.read_text())
    t["mode"] = "keyed"
    t["spend"] = {"ledger": [{"kind": "wonder:loop", "usd": 0.01}, {"kind": "wonder:watch", "usd": 0.004},
                             {"kind": "files:upload", "usd": 0.0}]}
    tp.write_text(json.dumps(t))
    out, _ = build(repo, treatment_path=tp, stand_in="a stock clip")
    c = {x["name"]: x for x in json.loads((out / "data" / "credits.json").read_text())["models"]}
    n = json.loads((repo / "out" / "TEST_LOCAL_t_hook_shots.json").read_text())["n_frames"]
    assert (c["SAM 3.1"]["ran"], c["SAM 3.1"]["calls"], c["SAM 3.1"]["frames"]) == (True, 1, n)
    assert c["SAM 3.1"]["usd"] == round(max(n, bw.SAM_MIN_FRAMES) * bw.SAM_USD_PER_1K_FRAMES / 1000, 4)
    assert (c["Muse Wonder"]["calls"], c["Muse Wonder"]["frames"], c["Muse Wonder"]["usd"]) == (2, None, 0.014)
    assert not c["Voice Transcribe"]["ran"] and c["Voice Transcribe"]["calls"] == 0


def test_the_test_song_and_named_stand_in_footage_make_a_test_render(repo):
    """A keyed render can still be a stand-in: the repo's synthetic song (caught by content, so a renamed copy
    counts) and a stock clip named with --stand-in. Without the tag prefix, both reasons still put the banner up."""
    import shutil as sh
    (repo / "footage").mkdir(exist_ok=True)
    sh.copy2(REPO / bw.TEST_SONG, repo / bw.TEST_SONG)
    sh.copy2(REPO / bw.TEST_SONG, repo / "footage" / "renamed.wav")
    run = {"engine": "SAM 3.1", "mode": "demo", "song": "footage/renamed.wav"}
    r = bw.test_reasons("real_tag", run, {"world_sources": []}, None, repo=repo, stand_in="Mixkit stock clip")
    assert r == ["Footage: Mixkit stock clip", "Song: the repo's synthetic test song, not the artist's master"]
    (repo / "footage" / "master.wav").write_bytes(b"RIFF not the test song")
    run["song"] = "footage/master.wav"
    assert bw.test_reasons("real_tag", run, {"world_sources": []}, None, repo=repo) == []


def direct_log(repo, mode="keyed", section=2, watches=2):
    worlds = repo / "out" / "worlds"
    worlds.mkdir(parents=True, exist_ok=True)
    (worlds / "muse_a.png").write_bytes(b"png a")
    (worlds / "muse_b.png").write_bytes(b"png b")
    note = {"t_s": 0.5, "against": "camera", "severity": "high", "note": "the dolly does not read before 4.9 s"}
    log = {"mode": mode, "model": "muse-wonder-1.0", "ended": "2026-11-01T23:30:00",
           "watches": [{"section": section, "source": "muse-wonder-1.0", "verdict": v, "notes": [note],
                        "suggestion": {"field": "none", "value": "", "why": "x"}}
                       for v in ("revise", "keep")[:watches]],
           "renders": {str(section): [{"video": str(worlds / "muse_a.png"), "duration_s": 19.6},
                                      {"video": str(worlds / "muse_b.png"), "duration_s": 19.6}]},
           "revision": {"changes": [{"section": section, "field": "world_prompt", "old": "a garage",
                                     "new": "a garage before dawn", "because": "no refused exit"}]},
           "spend": {"calls": 19, "spent_usd": 0.52,
                     "ledger": [{"kind": "wonder:loop", "usd": 0.01}, {"kind": "wonder:watch", "usd": 0.02}]}}
    p = repo / "out" / "director_x_log.json"
    p.write_text(json.dumps(log))
    return p


def fake_thumb(src, dst, t, width=540):
    Path(dst).write_bytes(b"thumb of " + Path(src).read_bytes() + f" at {t}".encode())


def test_the_directors_own_review_goes_on_the_page_verbatim(repo):
    """The `direct` log's watch notes, its one revision and the world before and after reach the page as written;
    no local path leaks, and the second watch is kept."""
    make_render(repo, local=False, plates=False)
    out, m = build(repo, stand_in="stock", direct_log=direct_log(repo), thumber=fake_thumb)
    r = json.loads((out / "data" / "review.json").read_text())
    assert r["mode"] == "keyed" and [c["verdict"] for c in r["cuts"]] == ["revise", "keep"]
    assert r["cuts"][0]["notes"][0]["note"] == "the dolly does not read before 4.9 s"
    assert r["changes"] == [{"field": "world_prompt", "old": "a garage", "new": "a garage before dawn",
                             "because": "no refused exit"}]
    assert r["stills"] == ["cut1.jpg", "cut2.jpg"] and r["still_t_s"] == 1.5     # the high note's time + 1 s
    assert (out / "data" / "cut1.jpg").read_bytes() == b"thumb of png a at 1.5"
    assert (out / "data" / "cut2.jpg").read_bytes() == b"thumb of png b at 1.5"
    assert m["files"]["review"] == "review.json" and "/Users/" not in json.dumps(r)
    c = {x["name"]: x for x in json.loads((out / "data" / "credits.json").read_text())["models"]}
    assert (c["Muse Wonder"]["ran"], c["Muse Wonder"]["calls"], c["Muse Wonder"]["usd"]) == (True, 2, 0.03)


def test_a_review_that_is_not_there_or_not_real_is_refused_or_labelled(repo):
    make_render(repo, local=False, plates=False)
    with pytest.raises(bw.Refuse, match="no watch of section 2"):
        build(repo, stand_in="stock", direct_log=direct_log(repo, section=3), thumber=fake_thumb)
    out, m = build(repo, stand_in="stock", direct_log=direct_log(repo, mode="DRY RUN"), thumber=fake_thumb)
    assert "Director review: a FIXTURE (dry) run, not Muse Wonder output" in m["test_reasons"]
    c = {x["name"]: x for x in json.loads((out / "data" / "credits.json").read_text())["models"]}
    assert not c["Muse Wonder"]["ran"]                                 # a dry log's ledger credits nobody
    lp = direct_log(repo)
    (repo / "out" / "worlds" / "muse_b.png").unlink()
    with pytest.raises(bw.Refuse, match="not on disk"):
        build(repo, stand_in="stock", direct_log=lp, thumber=fake_thumb)


def test_the_footer_never_calls_stand_in_footage_the_artists_own(repo):
    """Until Nov 1 the footer said 'FOOTAGE AND SONG ARE THE ARTIST'S OWN.' on a stock clip and a synthetic song."""
    make_render(repo)
    out, _ = build(repo)
    html = (out / "index.html").read_text()
    assert bw.FOOT_TEST in html and bw.FOOT_REAL not in html and bw.FOOT_MARK not in html
    (repo / "web" / "index.html").write_text((repo / "web" / "index.html").read_text().replace(bw.FOOT_MARK, "x"))
    with pytest.raises(bw.Refuse, match="FOOT_SOURCE"):
        build(repo)
