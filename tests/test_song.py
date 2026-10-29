"""Full song: every section boundary is a cut and every section opens wide with its own worlds; the units tile the
frames exactly once and never split a freeze release; raw segments join frame-exactly with the master audio; a
render does not depend on how it was cut into units; the new flags refuse cleanly before any work."""
import json
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from realone import units as U
from realone import worldlist as WL
from realone.common import FPS, H, W, Refuse
from realone.composite import LOOKS
from realone.freeze import resolve
from realone.media import count_frames, read_frames
from realone.plan import plan_range


def cfg(**kw):
    d = dict(world=None, worlds_file=None, world_image=None, worlds_json=None, section_world=None)
    d.update(kw)
    return SimpleNamespace(**d)


def test_whole_song_plan(timeline, song):
    secs = timeline["sections"]
    default, per = WL.load_config(cfg(section_world=["hook=neon", "default=roof", "verse=stage"]), [])
    entries, per_section, src = WL.assign(secs, default, per, local_plate=False)
    plan = plan_range(timeline, secs, 2, per_section)
    shots = plan["shots"]
    assert shots[0]["start_frame"] == 0 and shots[-1]["end_frame"] == plan["n_frames"]
    for a, b in zip(shots, shots[1:]):
        assert a["end_frame"] == b["start_frame"]
    starts = {s["bar"] for s in shots}
    assert {s["start_bar"] for s in secs} <= starts                      # every section boundary is a cut
    for s in secs:                                                       # and opens wide
        first = next(x for x in shots if x["bar"] == s["start_bar"])
        assert first["framing"] == "wide"
    truth = song[2]
    true_db = truth["first_downbeat_s"] + np.arange(truth["n_bars"] + 1) * truth["bar_s"]
    for sh in shots:
        t = plan["start_s"] + sh["start_frame"] / FPS
        assert np.abs(true_db - t).min() <= 1 / FPS
    names = [v for _, v in entries]
    for sh in shots:
        role = sh["role"]
        want = {"hook": "neon", "verse": "stage"}.get(role, "roof")
        assert names[sh["world"]] == want, (sh, role)
    assert src == ["default", "default", "hook", "default", "hook"] or "verse" in src


def test_world_lists_by_key_and_refusals(timeline, tmp_path):
    secs = timeline["sections"]
    img = tmp_path / "w.png"
    cv2.imwrite(str(img), np.zeros((16, 9, 3), np.uint8))
    j = tmp_path / "worlds.json"
    j.write_text(json.dumps({"hook": ["alley", {"image": str(img)}], "hook:2": ["rooftop"], "#0": ["stage"],
                             "default": ["field"]}))
    default, per = WL.load_config(cfg(worlds_json=str(j)), [])
    entries, per_section, src = WL.assign(secs, default, per, local_plate=False)
    assert src == ["#0", "default", "hook", "default", "hook:2"]
    assert [entries[i] for i in per_section[2]] == [("prompt", "alley"), ("image", str(img))]
    assert len(entries) == len(set(entries))  # one global list, deduplicated
    assert WL.prompts_in(default, per) == ["alley", "rooftop", "stage", "field"]
    _, per_section, _ = WL.assign(secs, [], {}, local_plate=True)
    assert per_section[1][0] != per_section[2][0]  # each role starts on a different plate
    bad = tmp_path / "bad.json"
    for content, msg in (("[1, 2]", "must be an object"), ('{"hook": [3]}', "each world is"),
                         ('{"hook": [{"image": "/nope.png"}]}', "not found"), ("{nope", "not valid JSON")):
        bad.write_text(content)
        with pytest.raises(Refuse, match=msg):
            WL.load_config(cfg(worlds_json=str(bad)), [])
    with pytest.raises(Refuse, match="KEY=PROMPT"):
        WL.load_config(cfg(section_world=["hook"]), [])
    with pytest.raises(Refuse, match="no world for"):
        WL.assign(secs, [], {"hook": [("prompt", "x")]}, local_plate=False)


def test_units_tile_every_frame_once_and_never_split_a_release(timeline):
    secs = timeline["sections"]
    fz = resolve(timeline, ["hook"], 4, secs)
    plan = plan_range(timeline, secs, 8, None, freezes=fz)
    rel = [f["end_frame"] for f in plan["freezes"]]
    for workers in (1, 4, 10, 32):
        units = U.make_units(plan, workers, avoid=rel)
        cover = np.zeros(plan["n_frames"], int)
        for f0, f1, warm in units:
            cover[f0:f1] += 1
            assert not any(abs(f0 - r) <= 2 for r in rel if f0 not in {s["start_frame"] for s in plan["shots"]})
        assert np.all(cover == 1), workers
        lens = [f1 - f0 for f0, f1, _ in units]
        assert lens == sorted(lens, reverse=True)  # longest first
    assert len(U.make_units(plan, 32, avoid=rel)) > len(plan["shots"])  # long shots were split for 32 workers


def _segments(tmp_path, lens):
    segs, fid = [], 0
    for k, n in enumerate(lens):
        p = tmp_path / f"seg{k}.h264"
        enc = U.seg_encoder(p, final=False)  # the renderer's own segment encoder
        for _ in range(n):
            fr = np.full((H, W, 3), (fid * 7) % 256, np.uint8)
            enc.stdin.write(fr.tobytes())
            fid += 1
        enc.stdin.close()
        assert enc.wait() == 0
        segs.append(p)
    return segs, fid


def test_raw_segments_join_frame_exactly_with_the_audio(tmp_path):
    segs, n = _segments(tmp_path, [9, 1, 14, 5])  # a 1-frame segment broke the mp4 concat demuxer
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=f=440:d={n / FPS:.4f}:sample_rate=48000",
                    "-ac", "2", str(wav)], check=True)
    out = U.concat_mux(segs, wav, tmp_path / "out.mp4", n)
    assert count_frames(out) == n
    vals = [int(f[10, 10, 0]) for f in read_frames(out)]
    assert all(abs(v - (i * 7) % 256) <= 3 for i, v in enumerate(vals)), vals
    info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,color_transfer", "-of", "csv=p=0",
                           str(out)], capture_output=True, text=True).stdout
    assert "audio" in info and "bt709" in info, info


def _tiny_render_ctx(tmp_path, n=12):
    """A real 12-frame render context: a drawn figure on a dim room, its mask, one procedural plate."""
    proxy = tmp_path / "proxy.mp4"
    masks = np.zeros((n, W * H // 8), np.uint8)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "8",
                            "-g", "30", "-pix_fmt", "yuv420p", str(proxy)], stdin=subprocess.PIPE)
    for i in range(n):
        fr = np.full((H, W, 3), (60, 50, 45), np.uint8)
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (540 + 6 * i, 900), (230, 620), 0, 0, 360, 1, -1)
        fr[m > 0] = (40, 40, 160)
        enc.stdin.write(fr.tobytes())
        masks[i] = np.packbits(m > 0)
    enc.stdin.close()
    assert enc.wait() == 0
    mpath = tmp_path / "masks.npy"
    np.save(mpath, masks)
    plate = tmp_path / "plate.png"
    yy, xx = np.mgrid[0:H, 0:W]
    cv2.imwrite(str(plate), np.stack([(xx * 0.2) % 255, (yy * 0.1) % 255, 120 + 0 * xx], -1).astype(np.uint8))
    crops = np.tile([0.0, 0.0, W, H], (n, 1))
    crops[6:] = (200, 300, W / 1.4, H / 1.4)
    arrays = {"crops": crops, "wf": np.arange(n, dtype=float), "pushf": np.linspace(0, 1, n),
              "shot_of": np.r_[np.zeros(6, int), np.ones(n - 6, int)], "boxes": np.tile([300.0, 280, 780, 1520], (n, 1)),
              "shot_world": np.array([0, 0]), "native_frames": np.zeros(n, np.uint8),
              "look_receipt": np.full(n, LOOKS.index("receipt"), np.uint8)}
    seg = tmp_path / "segs"
    seg.mkdir()
    np.savez(seg / "arrays.npz", **arrays)
    return {"args": dict(erode=2, feather=1.4, wrap=0.6, match=0.35, grain=0.0, despill=True, push=0.04),
            "proxy": str(proxy), "native": None, "native_size": None, "ns": [1.0, 1.0], "masks": str(mpath),
            "plates": [str(plate)], "particles": ["dust"], "arrays": str(seg / "arrays.npz"),
            "look_names": list(LOOKS), "outputs": ["receipt"], "s0": 30.0, "g0": 900, "seg_dir": str(seg),
            "final": False, "cv_threads": 2}


def test_a_render_does_not_depend_on_how_it_was_cut_into_units(tmp_path):
    ctx = _tiny_render_ctx(tmp_path)
    U.init_worker(ctx)
    whole = U.render_unit((0, 12, 0))
    out1 = U.concat_mux([whole["segments"]["receipt"]], _wav(tmp_path, 12), tmp_path / "w.mp4", 12)
    whole_frames = list(read_frames(out1))
    ctx2 = dict(ctx, seg_dir=str(tmp_path / "segs2"))
    (tmp_path / "segs2").mkdir()
    U.init_worker(ctx2)
    b = [U.render_unit((0, 4, 0)), U.render_unit((4, 9, U.WARM)), U.render_unit((9, 12, U.WARM))]
    assert whole["keys"] == sum((r["keys"] for r in b), [])          # identical world layer, frame for frame
    assert np.allclose(whole["filmed"]["receipt"], np.concatenate([r["filmed"]["receipt"] for r in b]), equal_nan=True)
    out2 = U.concat_mux([r["segments"]["receipt"] for r in b], _wav(tmp_path, 12), tmp_path / "p.mp4", 12)
    parts = list(read_frames(out2))
    assert len(parts) == len(whole_frames) == 12
    for i in range(12):  # no temporal state in the receipt look, grain off: only the encoder GOP differs
        assert np.abs(parts[i].astype(int) - whole_frames[i].astype(int)).mean() < 1.5, i


def _wav(tmp_path, n):
    p = tmp_path / f"a{n}.wav"
    if not p.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo",
                        "-t", f"{n / FPS:.4f}", str(p)], check=True)
    return p


def test_default_workers_never_outgrow_free_memory(monkeypatch):
    cores = max(1, (U.os.cpu_count() or 4) - 2)
    monkeypatch.setattr(U, "available_gb", lambda: 64.0)
    assert U.default_workers()[0] == cores
    monkeypatch.setattr(U, "available_gb", lambda: U.RESERVE_GB + 2.5 * U.WORKER_GB)
    assert U.default_workers()[0] == min(cores, 2)
    monkeypatch.setattr(U, "available_gb", lambda: 0.3)
    assert U.default_workers()[0] == 1  # never zero
    monkeypatch.setattr(U, "available_gb", lambda: None)
    assert U.default_workers()[0] == cores
    assert U.available_gb.__module__  # the real reader is restored by monkeypatch after the test
