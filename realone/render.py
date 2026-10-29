"""One section of the song, end to end: analyse -> sync -> plan -> mask -> worlds -> composite -> encode -> check."""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from realone import __version__
from realone.audio import analyze, pick_section
from realone.common import FPS, H, LOCAL_PREFIX, OUT, W, Refuse, file_md5, unpack_mask, write_json
from realone.composite import LOOKS, Compositor, lab_stats
from realone.encode import detect_cuts, start_encoder, verify_output
from realone.masks import sam_cost, sam_masks_cached, vision_masks
from realone.media import load_audio, load_audio_native, normalize_span, probe, read_frames, stream, write_wav
from realone.meta_api import IMAGE_USD, USD_TO_CAD, make_client, require_key
from realone.plan import bboxes, camera_tracks, plan_shots
from realone.sheets import contact_sheet, looks_sheet
from realone.sync import SR as SYNC_SR
from realone.sync import measure_offset, sync_clip
from realone.worlds import Particles, gen_world, load_plate, local_plates, muse_world_cached, world_view

PARALLAX = 0.25  # the world zooms a quarter as much as the artist: the artist feels closer than the world


def parse_looks(spec):
    looks = list(LOOKS) if spec == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    bad = [x for x in looks if x not in LOOKS]
    if bad or not looks:
        raise Refuse(f"unknown look {bad}; choose from {', '.join(LOOKS)} or 'all'")
    return looks


def world_prompts(a):
    prompts = list(a.world or [])
    if a.worlds_file:
        prompts += [ln.strip() for ln in Path(a.worlds_file).read_text().splitlines() if ln.strip()]
    return prompts


def gate(a):
    """Everything that can refuse cheaply, before any audio, video or network work."""
    for p in (a.clip, a.song):
        if not Path(p).exists():
            raise Refuse(f"not found: {p}")
    local = a.mask == "local"
    prompts = world_prompts(a)
    images = list(a.world_image or [])
    for p in images:
        if not Path(p).exists():
            raise Refuse(f"world image not found: {p}")
    if a.local_plate and not local:
        raise Refuse("--local-plate is test-only and needs --mask local")
    if not (prompts or images or a.local_plate):
        raise Refuse("no worlds: give --world '<prompt>' (repeatable, Muse Image), --world-image <path>, "
                     "or --local-plate (test only)")
    key = None
    if a.mask == "sam":
        key = require_key("--mask sam (SAM 3.1)")
    if prompts:
        key = key or require_key("--world (Muse Image)")
    return local, prompts, images, key


def render(a):
    t_start = time.time()
    looks = parse_looks(a.look)
    local, prompts, images, key = gate(a)
    prefix = LOCAL_PREFIX if local else ""
    base = prefix + a.tag
    OUT.mkdir(exist_ok=True)
    clip, song = Path(a.clip), Path(a.song)
    print(f"Real One {__version__}: {'LOCAL TEST (never the demo)' if local else 'DEMO path (SAM 3.1)'}; "
          f"MODEL_API_KEY {'present' if key else 'absent'}")

    # 1. audio
    tl = analyze(song, a.beats_per_bar)
    tl_path = write_json(OUT / f"{base}_timeline.json", tl)
    sec = pick_section(tl, a.section)
    sec_name = a.section.replace(":", "").replace("-", "_").replace("#", "s")
    tag = f"{base}_{sec_name}"
    print(f"song: {tl['bpm']:.2f} BPM ({tl['grid']} grid), {tl['beats_per_bar']}/4, {len(tl['downbeats'])} bars, "
          f"sections: " + ", ".join(f"{s['role']}[{s['start_bar']}-{s['end_bar']}]" for s in tl["sections"]))
    print(f"section '{a.section}': bars {sec['start_bar']}-{sec['end_bar']} = {sec['start_s']:.3f}-{sec['end_s']:.3f}s")

    # 2. sync (phone mic is only the clock; the render uses the master)
    sy = sync_clip(song, clip, a.sync_near)
    write_json(OUT / f"{tag}_sync.json", sy)
    off = sy["offset_s"]
    print(f"sync: offset {sy['offset_ms']:+.1f} ms (song = clip + offset), r={sy['r_peak']:.2f}, "
          f"margin {sy['margin']:.3f}, resolved by {sy['resolved_by']}")

    # 3. plan
    worlds_n = 3 if a.local_plate else len(prompts) + len(images)
    plan = plan_shots(tl, sec, a.cut_bars, worlds_n)
    s0, n = plan["start_s"], plan["n_frames"]
    dur = n / FPS
    if dur > a.max_seconds:
        raise Refuse(f"the section is {dur:.1f}s; the floor renders at most {a.max_seconds:.0f}s per run "
                     f"(--max-seconds, or pick bars with --section bars:A-B)")
    info = probe(clip)
    vlen = float((stream(info, "video") or {}).get("duration") or info["format"]["duration"])
    c0 = s0 - off
    if c0 < 0 or c0 + dur > vlen + 1e-3:
        raise Refuse(f"the take does not cover the section: it needs clip time {c0:.2f}-{c0 + dur:.2f}s, "
                     f"the clip runs 0-{vlen:.2f}s")

    # 4. normalize the span of the take that covers the section
    work = OUT / "work" / file_md5(clip)[:12]
    norm, n = normalize_span(clip, c0, dur, work)
    mtag = file_md5(norm)[:12] + f"_{n}f"

    # 5. spend quote, then masks
    client = make_client(key) if key else None
    usd = 0.0
    if not local and not any((OUT / "masks").glob(f"{mtag}__*.txt")):
        usd += sam_cost(n)
    usd += sum(IMAGE_USD for p in prompts if not muse_world_cached(p, not a.no_world_wrap))
    if usd:
        print(f"spend quote: US${usd:.3f} (about CA${usd * USD_TO_CAD:.2f}) for SAM {n} frames + "
              f"{sum(1 for p in prompts if not muse_world_cached(p, not a.no_world_wrap))} images")
        if usd > a.max_usd:
            raise Refuse(f"quote US${usd:.2f} is over --max-usd {a.max_usd:.2f}; nothing was called")
    billed = 0.0
    if local:
        (OUT / "masks").mkdir(parents=True, exist_ok=True)
        masks = vision_masks(norm, n, OUT / "masks" / f"local_{mtag}.npy")
        engine = "Apple Vision (VNGenerateForegroundInstanceMaskRequest), TEST ONLY"
    else:
        masks, engine, b = sam_masks_cached(client, norm, mtag, n)
        billed += b

    # 6. camera
    boxes = bboxes(masks)
    crops = camera_tracks(plan, boxes)

    # 7. worlds
    if a.local_plate:
        worlds = [(nm, p, kind) for nm, p, kind in local_plates()]
    else:
        worlds = [(Path(p).stem, Path(p), a.particles) for p in images]
        for p in prompts:
            path, b = gen_world(client, p, wrap=not a.no_world_wrap)
            billed += b
            worlds.append((p[:28], path, a.particles))
    plates = [load_plate(p) for _, p, _ in worlds]
    wstats = []
    for pl in plates:
        small = cv2.resize(pl, (270, 480), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        wstats.append(lab_stats(small))
    parts = [Particles(kind) for _, _, kind in worlds]
    names = [nm for nm, _, _ in worlds]
    for sh in plan["shots"]:
        sh["world_name"] = names[sh["world"]]
        c = crops[sh["start_frame"]]
        sh["crop_first_frame"] = [round(float(v), 1) for v in c]
    plan["sync"] = {k: sy[k] for k in ("offset_s", "r_peak", "margin", "resolved_by")}
    plan["clip_span_s"] = [round(c0, 4), round(c0 + dur, 4)]
    shots_path = write_json(OUT / f"{tag}_shots.json", plan)

    # 8. master audio for exactly this section (sample-accurate slice of the master, never the phone mic)
    master, sr = load_audio_native(song)
    i0 = int(round(s0 * sr))
    wav = write_wav(work / f"master_{s0:.4f}_{n}f.wav", master[i0:i0 + int(round(dur * sr))], sr)

    # 9. composite + encode
    outs = {look: OUT / f"{tag}_{look}.mp4" for look in looks}
    encs = {look: start_encoder(outs[look], wav, a.final) for look in looks}
    comp = Compositor(a)
    shot_of = np.zeros(n, int)
    for sh in plan["shots"]:
        shot_of[sh["start_frame"]:sh["end_frame"]] = sh["index"]
    g0 = int(round(s0 * FPS))
    bbox_out = np.zeros((n, 4))
    despill_stats = []
    t0 = time.time()
    try:
        for i, fr in enumerate(read_frames(norm)):
            if i >= n:
                break
            sh = plan["shots"][shot_of[i]]
            t = s0 + i / FPS
            x0, y0, cw, ch = crops[i]
            z = W / cw
            ccx, ccy = x0 + cw / 2, y0 + ch / 2
            zw = (1 + PARALLAX * (z - 1)) * (1 + a.push * (i - sh["start_frame"]) / max(1, sh["frames"]))
            centre = (W / 2 + (ccx - W / 2) * PARALLAX, H / 2 + (ccy - H / 2) * PARALLAX)
            wbase = world_view(plates[sh["world"]], zw, centre, dx=16 * np.sin(0.55 * t)).astype(np.float32)
            back, front = parts[sh["world"]].layers(t)
            world = np.clip(wbase + back * 255, 0, 255)
            m = unpack_mask(masks[i])
            frames, inf = comp.frame(fr, m, crops[i], world, front * 255, wstats[sh["world"]], i, g0 + i, looks)
            for look in looks:
                encs[look].stdin.write(frames[look].tobytes())
            bx = boxes[i] if not np.isnan(boxes[i, 0]) else (0, 0, W, H)
            bbox_out[i] = ((bx[0] - x0) * z, (bx[1] - y0) * z, (bx[2] - x0) * z, (bx[3] - y0) * z)
            despill_stats.append(inf[looks[0]].get("pocket_px", 0))
            el = time.time() - t0
            if i % 60 == 0:
                print(f"  frame {i}/{n}  {(i + 1) / max(el, 1e-6):.2f} fps  {el:.0f}s")
            if el > a.max_minutes * 60:
                raise Refuse(f"render passed {a.max_minutes:.0f} min at frame {i}/{n}; stopped (the Mac may be "
                             f"throttled). Nothing was claimed.")
    finally:
        for e in encs.values():
            e.stdin.close()
        rcs = {look: e.wait() for look, e in encs.items()}
    if any(rcs.values()):
        raise Refuse(f"ffmpeg encode failed: {rcs}")
    render_s = time.time() - t0

    # 10. checks on what was actually written
    master22 = load_audio(song, SYNC_SR)
    report = {"version": __version__, "mode": "local test" if local else "demo", "engine": engine,
              "clip": str(clip), "song": str(song), "section": sec, "timeline": str(tl_path),
              "shots": str(shots_path), "render_s": round(render_s, 1), "fps": round(n / max(render_s, 1e-6), 2),
              "billed_usd": round(billed, 4), "looks": {}}
    planned = [s["start_frame"] for s in plan["shots"][1:]]
    all_good = True
    for look in looks:
        rep, good = verify_output(outs[look], dur)
        cuts, _ = detect_cuts(outs[look])
        matched = [p for p in planned if any(abs(c - p) <= 1 for c in cuts)]
        extra = [c for c in cuts if not any(abs(c - p) <= 1 for p in planned)]  # a cut off the beat grid
        out_audio = load_audio(outs[look], SYNC_SR)
        av = measure_offset(master22, out_audio, near_s=s0)
        av_err_ms = (av["offset_s"] - s0) * 1000
        ok = good and len(matched) == len(planned) and not extra and abs(av_err_ms) <= 1000 / FPS
        all_good &= ok
        report["looks"][look] = {"video": str(outs[look]), "spec": rep, "spec_ok": good,
                                 "cuts_detected": cuts, "cuts_planned": planned,
                                 "cuts_on_plan": f"{len(matched)}/{len(planned)}", "cuts_off_plan": extra,
                                 "audio_vs_master_error_ms": round(av_err_ms, 2), "ok": ok}
    # 11. contact sheets (read back from the encoded files)
    title = (f"{'TEST_LOCAL (never the demo)  ' if local else ''}Real One {__version__}  {sec['role']}: "
             f"bar {sec['start_bar'] + 1} to {sec['end_bar']}  {tl['bpm']:.1f} BPM  cut every {a.cut_bars} bars  "
             f"sync {sy['offset_ms']:+.0f} ms")
    for look in looks:
        cs = OUT / f"{tag}_{look}_contact.jpg"
        contact_sheet(outs[look], plan, bbox_out, look, cs, title, names)
        report["looks"][look]["contact_sheet"] = str(cs)
    if len(looks) > 1:
        ls = OUT / f"{prefix}looks.jpg"
        looks_sheet(outs, plan, ls, title + "   rows: " + " / ".join(looks))
        report["looks_sheet"] = str(ls)
    report["frames_with_pocket_cut"] = int(sum(1 for v in despill_stats if v > 0))
    report["total_s"] = round(time.time() - t_start, 1)
    write_json(OUT / f"{tag}_run.json", report)

    print("\n=== result ===")
    for look, r in report["looks"].items():
        print(f"{look:8s} {r['video']}\n         {r['spec']}\n         cuts on plan {r['cuts_on_plan']}, "
              f"audio vs master {r['audio_vs_master_error_ms']:+.1f} ms, contact {r['contact_sheet']}")
    if "looks_sheet" in report:
        print(f"looks    {report['looks_sheet']}")
    print(f"render   {n} frames x {len(looks)} looks in {render_s:.0f}s ({report['fps']} fps); total {report['total_s']}s")
    print(f"masks    {engine}")
    print("cost     US$0.00 (local test, no API calls)" if local and not prompts else
          f"cost     billed this run US${billed:.4f} (about CA${billed * USD_TO_CAD:.2f})")
    if not all_good:
        raise Refuse("an output failed its checks (see above and the run json)", )
    return report
