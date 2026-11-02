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
    html = (out / "index.html").read_text()
    assert "Real model output here: Voice Transcribe, Muse Wonder, Muse Image." in html
    assert "No Meta model ran" not in html


def test_a_dry_run_treatment_is_never_credited_as_a_model_run(repo):
    make_render(repo)
    tp = treatment_file(repo, fixture=True)
    t = json.loads(tp.read_text())
    t["spend"] = {"ledger": [{"kind": "wonder:treatment", "usd": 0.0}]}   # a dry run can log its fixture calls
    tp.write_text(json.dumps(t))
    out, m = build(repo, treatment_path=tp)
    c = json.loads((out / "data" / "credits.json").read_text())
    assert not next(x for x in c["models"] if x["name"] == "Muse Wonder")["ran"]
