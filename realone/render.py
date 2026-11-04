"""A section, or the whole song, end to end: analyse -> sync -> plan -> mask -> worlds -> composite (parallel units)
-> concat with the master -> check -> sheets."""
import json
import os
import resource
import shutil
import time
from multiprocessing import TimeoutError as PoolTimeout
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from realone import __version__
from realone import freeze as FZ
from realone import receipt as R
from realone import units as U
from realone import worldlist as WL
from realone.audio import analyze, pick_section
from realone.common import FPS, LOCAL_PREFIX, OUT, Refuse, file_md5, unpack_mask, write_json, W
from realone.composite import LIGHT_SHARE, LOOKS, border_ink, crop_matrix, is_identity
from realone.encode import detect_cuts, planned_cut_found, verify_output
from realone.masks import sam_cost, sam_masks_cached, sam_masks_union, sam_uncached, vision_masks
from realone.media import count_frames, load_audio, load_audio_native, probe, stream, write_wav
from realone.meta_api import IMAGE_USD, USD_TO_CAD, make_client, require_key
from realone.native import normalize as normalize_take
from realone.plan import bboxes, camera_tracks, plan_range, shot_index
from realone.sheets import contact_sheet, freeze_sheet, looks_sheet, song_sheet
from realone.sync import SR as SYNC_SR
from realone.sync import measure_offset, sync_clip
from realone.worlds import gen_world, load_plate, local_plates, make_particles, muse_world_cached, world_key, world_layer


def parse_looks(spec):
    looks = list(LOOKS) if spec == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    bad = [x for x in looks if x not in LOOKS]
    if bad or not looks:
        raise Refuse(f"unknown look {bad}; choose from {', '.join(LOOKS)} or 'all'")
    return looks


def world_prompts(a):
    prompts = list(a.world or [])
    if a.worlds_file:
        if not Path(a.worlds_file).exists():
            raise Refuse(f"--worlds-file not found: {a.worlds_file}")
        prompts += [ln.strip() for ln in Path(a.worlds_file).read_text().splitlines() if ln.strip()]
    return prompts


def gate(a):
    """Everything that can refuse cheaply, before any audio, video or network work."""
    for p in (a.clip, a.song):
        if not Path(p).exists():
            raise Refuse(f"not found: {p}")
    local = a.mask == "local"
    for p in list(a.world_image or []):
        if not Path(p).exists():
            raise Refuse(f"world image not found: {p}")
    default, per = WL.load_config(a, world_prompts(a))
    prompts = WL.prompts_in(default, per)
    if a.local_plate and not local:
        raise Refuse("--local-plate is test-only and needs --mask local")
    if getattr(a, "stage", "off") == "tracked" and a.local_plate:
        raise Refuse("--stage tracked films a world still through the take's camera; the procedural test plates "
                     "have no depth to rebuild (use --world-image or a Muse Image world)")
    if getattr(a, "sam_prompt", None) and local:
        raise Refuse("--sam-prompt is for --mask sam (SAM 3.1); the local test masks take no prompt")
    if not (default or per or a.local_plate):
        raise Refuse("no worlds: give --world '<prompt>' (repeatable, Muse Image), --world-image <path>, "
                     "--worlds-json, or --local-plate (test only)")
    if getattr(a, "workers", None) is not None and a.workers < 1:
        raise Refuse(f"--workers must be at least 1 (got {a.workers})")
    key = None
    if a.mask == "sam":
        key = require_key("--mask sam (SAM 3.1)")
    if prompts:
        key = key or require_key("--world (Muse Image)")
    return local, default, per, prompts, key


def run_units(ctx, units, workers, max_s, t0, n):
    """Render every unit; workers > 1 uses a spawn pool (one Compositor per process). -> records, by f0."""
    recs, done = [], 0
    if workers == 1:
        U.init_worker(ctx)
        for u in units:
            recs.append(U.render_unit(u))
            done += u[1] - u[0]
            _progress(recs[-1], done, n, t0, len(recs), len(units))
            if time.time() - t0 > max_s:
                raise Refuse(f"render passed the --max-minutes budget at {done}/{n} frames; stopped. "
                             f"Nothing was claimed.")
        return sorted(recs, key=lambda r: r["f0"])
    pool = get_context("spawn").Pool(workers, initializer=U.init_worker, initargs=(ctx,))
    try:
        it = pool.imap_unordered(U.render_unit, units)
        for _ in units:
            left = max_s - (time.time() - t0)
            try:
                r = it.next(timeout=max(1.0, left))
            except PoolTimeout:
                raise Refuse(f"render passed the --max-minutes budget at {done}/{n} frames; stopped (the Mac may "
                             f"be throttled). Nothing was claimed.") from None
            recs.append(r)
            done += r["f1"] - r["f0"]
            _progress(r, done, n, t0, len(recs), len(units))
        pool.close()
    finally:
        pool.terminate()
        pool.join()
    return sorted(recs, key=lambda r: r["f0"])


def _progress(r, done, n, t0, k, total):
    el = time.time() - t0
    print(f"  unit {k}/{total}: frames {r['f0']}-{r['f1'] - 1} in {r['seconds']:.0f}s "
          f"({(r['f1'] - r['f0']) / max(r['seconds'], 1e-6):.2f} fps, pid {r['pid']}, {r['maxrss_mb']:.0f} MB)   "
          f"{done}/{n} frames, {el:.0f}s, {done / max(el, 1e-6):.1f} fps overall", flush=True)


def collage_inks(plan, worlds, masks, crops, wf, pushf, s0, push, style="still"):
    """Collage legibility, decided once per shot (so the border never flickers): on the shot's first, middle and
    last frame, the share of the band under the paper border that is brighter than LIGHT_LUMA; over LIGHT_SHARE on
    average -> a dark outline for the whole shot. -> uint8 per shot (1 = dark), and the shots get
    `collage_ink` and `border_light_share`."""
    import cv2
    out = np.zeros(len(plan["shots"]), np.uint8)
    plates = {}
    for k, sh in enumerate(plan["shots"]):
        f0, f1 = sh["start_frame"], sh["end_frame"]
        wid = sh["world"]
        if wid not in plates:
            plates[wid] = (load_plate(worlds[wid][1]), make_particles(worlds[wid][2], style))
        plate, parts = plates[wid]
        shares = []
        for i in sorted({f0, (f0 + f1) // 2, f1 - 1}):
            world, _, _ = world_layer(plate, parts, crops[i], pushf[i], push, s0 + wf[i] / FPS)
            m = unpack_mask(masks[i])
            if not is_identity(crops[i]):
                M, _ = crop_matrix(crops[i])
                m = cv2.warpAffine(m.astype(np.float32), M, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR) > 0.5
            shares.append(border_ink(world, m)[1])
        share = float(np.mean(shares))
        sh["border_light_share"] = round(share, 3)
        sh["collage_ink"] = "dark" if share > LIGHT_SHARE else "white"
        out[k] = sh["collage_ink"] == "dark"
    return out


def apply_treatment(a):
    """--from-treatment: the director's treatment drives the render. Each section's world_prompt becomes its world
    (--section-world #i=...), its move goes to --moves-from, and the freeze and receipt-look sections are kept
    for after the analysis (only the ones inside the rendered range apply; see _treatment_specs)."""
    path = getattr(a, "from_treatment", None)
    if not path:
        if getattr(a, "word_times", None):
            raise Refuse("--word-times needs --from-treatment: the times belong to the treatment's lyric lines")
        return
    p = Path(path)
    if not p.exists():
        raise Refuse(f"--from-treatment not found: {p}")
    try:
        obj = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise Refuse(f"--from-treatment is not valid JSON: {e}") from None
    t = obj.get("treatment", obj)
    if not isinstance(t, dict) or not isinstance(t.get("sections"), list) or not t["sections"]:
        raise Refuse(f"--from-treatment {p}: not a treatment (no sections)")
    sw = list(getattr(a, "section_world", None) or [])
    for s in t["sections"]:
        if not isinstance(s.get("index"), int) or not str(s.get("world_prompt", "")).strip():
            raise Refuse(f"--from-treatment {p}: a section has no index or no world_prompt")
        sw.append(f"#{s['index']}={s['world_prompt']}")
    a.section_world = sw
    if not getattr(a, "moves_from", None):
        a.moves_from = str(p)
    a._treatment_freeze = [s["index"] for s in t["sections"] if s.get("freeze") is True]
    a._treatment_reveal = [s["index"] for s in t["sections"] if s.get("look") == "receipt"]
    a._treatment_particles = {s["index"]: s["particles"] for s in t["sections"] if s.get("particles")}
    a._treatment_lyrics = {s["index"]: list(s.get("lyric") or []) for s in t["sections"]}
    a._treatment_palettes = {s["index"]: list(s.get("palette") or []) for s in t["sections"]}
    a._treatment_moments = ({s["index"]: list(s.get("moments") or []) for s in t["sections"]}
                            if any("moments" in s for s in t["sections"]) else None)
    if getattr(a, "word_times", None):
        from realone import lyrictext as LT
        try:
            a._word_times = LT.load_word_times(a.word_times, a._treatment_lyrics)
        except LT.LyricError as e:
            raise Refuse(str(e)) from None
        a._word_source = json.loads(Path(a.word_times).read_text()).get("source", "word times")


def lyrics_mode(a):
    """forced (--lyrics-in-world: refuse without words), auto (on by default when the treatment carries lyric
    lines, James Nov 2: "when the text is there, it's a lot more interesting"), or off (--no-lyrics, no lyric, or a
    lyric with words on the violence list: those never go on screen unless forced)."""
    if getattr(a, "lyrics_in_world", False):
        return "forced"
    lines = [x for v in (getattr(a, "_treatment_lyrics", None) or {}).values() for x in v]
    if getattr(a, "no_lyrics", False) or not lines:
        return "off"
    from realone.moments import violent_words
    bad = sorted({w for x in lines for w in violent_words(x)})
    if bad:
        a._lyrics_off_why = (f"the lyric has words on the violence list ({', '.join(bad[:4])}); "
                             "--lyrics-in-world forces them on")
        return "off"
    return "auto"


def _treatment_specs(a, indices, secs):
    """Section indices from the treatment -> '#i' specs for the sections being rendered (others are skipped)."""
    inside = {s.get("index") for s in secs}
    return [f"#{i}" for i in indices if i in inside]


def render(a):
    t_start = time.time()
    stages = {}
    looks = parse_looks(a.look)
    apply_treatment(a)
    local, default, per, prompts, key = gate(a)
    prefix = LOCAL_PREFIX if local else ""
    base = prefix + a.tag
    OUT.mkdir(exist_ok=True)
    clip, song = Path(a.clip), Path(a.song)
    if getattr(a, "workers", None):
        workers, why = a.workers, "--workers"
    else:
        workers, why = U.default_workers()
    print(f"Real One {__version__}: {'LOCAL TEST (never the demo)' if local else 'DEMO path (SAM 3.1)'}; "
          f"MODEL_API_KEY {'present' if key else 'absent'}")

    # 1. audio
    t = time.time()
    tl = analyze(song, a.beats_per_bar, getattr(a, "sections", None))
    tl_path = write_json(OUT / f"{base}_timeline.json", tl)
    whole = a.section == "all"
    secs = list(tl["sections"]) if whole else [pick_section(tl, a.section)]
    sec_name = "all" if whole else a.section.replace(":", "").replace("-", "_").replace("#", "s")
    tag = f"{base}_{sec_name}"
    print(f"song: {tl['bpm']:.2f} BPM ({tl['grid']} grid), {tl['beats_per_bar']}/4, {len(tl['downbeats'])} bars, "
          f"sections: " + ", ".join(f"{s['role']}[{s['start_bar']}-{s['end_bar']}]" for s in tl["sections"]))
    print(f"section '{a.section}': bars {secs[0]['start_bar']}-{secs[-1]['end_bar']} = "
          f"{secs[0]['start_s']:.3f}-{secs[-1]['end_s']:.3f}s")
    stages["analyze_s"] = round(time.time() - t, 1)

    # 2. sync (phone mic is only the clock; the render uses the master)
    t = time.time()
    sy = sync_clip(song, clip, a.sync_near)
    write_json(OUT / f"{tag}_sync.json", sy)
    off = sy["offset_s"]
    print(f"sync: offset {sy['offset_ms']:+.1f} ms (song = clip + offset), r={sy['r_peak']:.2f}, "
          f"margin {sy['margin']:.3f}, resolved by {sy['resolved_by']}")
    stages["sync_s"] = round(time.time() - t, 1)

    # 3. plan: worlds per section, freezes, cuts
    if getattr(a, "_treatment_freeze", None):
        a.freeze = list(getattr(a, "freeze", None) or []) + _treatment_specs(a, a._treatment_freeze, secs)
    if getattr(a, "_treatment_reveal", None):
        a.receipt_reveal = (list(getattr(a, "receipt_reveal", None) or [])
                            + _treatment_specs(a, a._treatment_reveal, secs))
    tp = getattr(a, "_treatment_particles", None) or {}
    kinds = {tp[s["index"]] for s in secs if s.get("index") in tp}
    if len(kinds) == 1 and a.particles == "dust":  # the director revised the overlay for the rendered section(s)
        a.particles = kinds.pop()
        print(f"particles: {a.particles} (from the treatment)")
    freezes = FZ.resolve(tl, getattr(a, "freeze", None), getattr(a, "freeze_bars", 2), secs)
    style = getattr(a, "freeze_style", "bullet") if freezes else "still"  # no freeze: 0.2.0 unchanged
    lead_in = None
    if style == "bullet":
        secs, lead_in = FZ.lead_in(tl, secs, freezes, sy["offset_s"], None if whole else a.max_seconds)
        if lead_in:
            print(f"freeze lead-in: {lead_in['bars']} bars" + (f" (bars {lead_in['from_bar']}-{lead_in['freeze_bar']})"
                                                                if lead_in["bars"] else f": none, {lead_in['why']}"))
    entries, per_section, wsrc = WL.assign(secs, default, per, a.local_plate)
    reveal_secs = []
    for spec in getattr(a, "receipt_reveal", None) or []:
        reveal_secs += [x for x in FZ.select_sections(tl, spec, secs) if x not in reveal_secs]
    lo, hi = secs[0]["start_bar"], secs[-1]["end_bar"]
    reveal_bars = sorted({(max(lo, x["start_bar"]), min(hi, x["end_bar"])) for x in reveal_secs})
    plan = plan_range(tl, secs, a.cut_bars, per_section, freezes=freezes,
                      force_cuts=[b for pair in reveal_bars for b in pair], freeze_style=style)
    plan["freeze_style"] = style if plan["freezes"] else None
    plan["freeze_lead_in"] = lead_in
    from realone.stage import moves as MV
    move_specs = list(getattr(a, "move", None) or [])
    if getattr(a, "moves_from", None):
        move_specs += MV.specs_from_treatment(a.moves_from, secs)
    moves = MV.resolve(tl, move_specs, secs, plan, plan["freezes"])
    if any(m["kind"] == "dolly" for m in moves) and getattr(a, "stage", "off") != "tracked":
        raise Refuse("--move dolly needs --stage tracked: a dolly zoom stretches the world by its depth, and only "
                     "the tracked stage has the world's depth and the take's camera")
    plan["moves"] = [{k: v for k, v in m.items() if k != "downbeat_frames"} for m in moves]
    for m in moves:
        print(f"move: {m['kind']} on the {m['role']}, bars {m['start_bar']}-{m['end_bar']} = frames "
              f"{m['start_frame']}-{m['end_frame']}")
    for f in plan["freezes"]:
        print(f"freeze: {f['role']} bars {f['start_bar']}-{f['end_bar']} = frames {f['start_frame']}-"
              f"{f['end_frame'] - 1} ({f['frames'] / FPS:.2f}s): the world holds, the artist keeps moving"
              + ("; bullet: the camera keeps moving, the particles hang on 3 depth layers, hit + grade"
                 if style == "bullet" else "; still: the camera holds too"))
    s0, n = plan["start_s"], plan["n_frames"]
    dur = n / FPS

    def frame_of_bar(bar):
        t = tl["downbeats"][bar] if bar < len(tl["downbeats"]) else plan["end_s"]
        return int(np.clip(round((t - s0) * FPS), 0, n))

    plan["receipt_reveal"] = [{"start_bar": b0, "end_bar": b1, "frames": [frame_of_bar(b0), frame_of_bar(b1) - 1],
                               "roles": sorted({x["role"] for x in reveal_secs if x["start_bar"] <= b0 < x["end_bar"]})}
                              for b0, b1 in reveal_bars]
    for r_ in plan["receipt_reveal"]:
        print(f"receipt reveal: bars {r_['start_bar']}-{r_['end_bar']} = frames {r_['frames'][0]}-{r_['frames'][1]}: "
              f"generated pixels grey, the filmed artist in colour"
              + (" + counter" if getattr(a, "receipt_counter", False) else ""))
    if not whole and dur > a.max_seconds:
        raise Refuse(f"the section is {dur:.1f}s; the floor renders at most {a.max_seconds:.0f}s per run "
                     f"(--max-seconds, or pick bars with --section bars:A-B, or the whole song with --section all)")
    info = probe(clip)
    vlen = float((stream(info, "video") or {}).get("duration") or info["format"]["duration"])
    c0 = s0 - off
    if c0 < 0 or c0 + dur > vlen + 1e-3:
        raise Refuse(f"the take does not cover the section: it needs clip time {c0:.2f}-{c0 + dur:.2f}s, "
                     f"the clip runs 0-{vlen:.2f}s")

    # 4. normalize the span of the take that covers the section(s)
    t = time.time()
    work = OUT / "work" / file_md5(clip)[:12]
    take = normalize_take(clip, c0, dur, work, keep_native=not getattr(a, "no_native", False))
    norm, n = take["proxy"], take["n"]
    print(f"normalize: {take['route']}  source {take['coded'][0]}x{take['coded'][1]} {take['codec']} "
          f"{take['pix_fmt']} {take['transfer']} rotation {take['rotation']} -> proxy 1080x1920"
          + (f" + native {take['native_size'][0]}x{take['native_size'][1]} for punch-ins" if take["native"] else ""))
    mtag = file_md5(norm)[:12] + f"_{n}f"
    stages["normalize_s"] = round(time.time() - t, 1)

    # 5. spend quote, then masks
    wrap = not a.no_world_wrap
    new_prompts = [v for kind, v in entries if kind == "prompt" and not muse_world_cached(v, wrap)]
    client = make_client(key) if key else None
    usd = 0.0
    sam_prompts = getattr(a, "sam_prompt", None)
    if not local and sam_prompts:
        usd += sam_cost(n) * len(sam_uncached(mtag, sam_prompts))  # one call per prompt not yet cached
    elif not local and not any((OUT / "masks").glob(f"{mtag}__*.txt")):
        usd += sam_cost(n)
    usd += IMAGE_USD * len(new_prompts)
    fx_list = None
    if getattr(a, "edits", None):  # tracked effects (realone/fx.py): one SAM 3.1 call per part named, cached
        from realone import fx as FX
        fx_list = FX.load_edits(a.edits)
        if local:
            raise Refuse("--edits: tracked effects need SAM 3.1 (a keyed render); the local test masks have no parts")
        usd += sam_cost(n) * len(sam_uncached(mtag, sorted({e["track"] for e in fx_list})))
    if usd:
        print(f"spend quote: US${usd:.3f} (about CA${usd * USD_TO_CAD:.2f}) for SAM {n} frames + "
              f"{len(new_prompts)} images")
        if usd > a.max_usd:
            raise Refuse(f"quote US${usd:.2f} is over --max-usd {a.max_usd:.2f}; nothing was called")
    billed = 0.0
    t = time.time()
    if local:
        (OUT / "masks").mkdir(parents=True, exist_ok=True)
        mpath, req = vision_masks(norm, n, OUT / "masks" / f"local_{mtag}.npy", workers=min(workers, 4))
        engine = f"Apple Vision ({req}), TEST ONLY"
    else:
        if sam_prompts:
            mpath, engine, b = sam_masks_union(client, norm, mtag, n, sam_prompts)
        else:
            mpath, engine, b = sam_masks_cached(client, norm, mtag, n)
        billed += b
    masks = np.load(mpath, mmap_mode="r")
    stages["masks_s"] = round(time.time() - t, 1)

    fx_boxes = {}
    if fx_list:
        from realone.masks import _decoded, sam_text_one, slug
        for track in sorted({e["track"] for e in fx_list}):
            text, b = sam_text_one(client, norm, mtag, n, track)
            billed += b
            if not text.strip():
                raise Refuse(f"--edits: SAM 3.1 found no '{track}' in this take")
            fx_boxes[track] = bboxes(np.load(_decoded(text, n, OUT / "masks" / f"{mtag}__{slug(track)}.masks.npy"),
                                             mmap_mode="r"))

    # 6. camera
    t = time.time()
    boxes = bboxes(masks)
    shot_of = shot_index(plan)
    wf = FZ.world_clock(n, plan["freezes"])  # the world clock in frames (holds inside a freeze)
    crops = FZ.hold_camera(camera_tracks(plan, boxes), plan["freezes"], shot_of)
    pushf = FZ.push_fraction(wf, plan)

    # 7. worlds
    worlds = []
    light_txt = ""
    if (getattr(a, "stage", "off") == "tracked" and not getattr(a, "no_world_light", False)
            and any(kind not in ("plate", "image") for kind, _ in entries)):
        from realone.stage import light as LIGHT  # light every generated world the way the take was lit
        info, light_txt = LIGHT.from_take(norm, masks, n)
        if info:
            print(f"world light: the take reads {info['kind']} (background L* {info['l_mean']:.0f}, sky grey "
                  f"{info['sky_grey']:.2f} / blue {info['sky_blue']:.2f}); appended to every world prompt:{light_txt}")
            plan["world_light"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in info.items()}
    plates_local = local_plates() if any(kind == "plate" for kind, _ in entries) else []
    from realone import reflect as RF
    wet_worlds = []  # worlds whose ground is wet: they mirror him (realone/reflect.py)
    for kind, v in entries:
        wet_worlds.append(kind == "prompt" and RF.is_wet(v))
        if kind == "plate":
            worlds.append(plates_local[v])
        elif kind == "image":
            worlds.append((Path(v).stem, Path(v), a.particles))
        else:
            path, b = gen_world(client, v + light_txt, wrap=wrap)
            billed += b
            worlds.append((v[:28], path, a.particles))
    names = [nm for nm, _, _ in worlds]
    plan["wet_worlds"] = [nm for nm, wet in zip(names, wet_worlds) if wet]
    for sh in plan["shots"]:
        sh["world_name"] = names[sh["world"]]
        c = crops[sh["start_frame"]]
        sh["crop_first_frame"] = [round(float(v), 1) for v in c]
    plan["sync"] = {k: sy[k] for k in ("offset_s", "r_peak", "margin", "resolved_by")}
    plan["clip_span_s"] = [round(c0, 4), round(c0 + dur, 4)]
    plan["world_sources"] = [{"section": s.get("index", -1), "role": s["role"], "worlds": [names[i] for i in ids],
                              "from": src} for s, ids, src in zip(secs, per_section, wsrc)]
    shots_path = write_json(OUT / f"{tag}_shots.json", plan)

    # 8. master audio for exactly this span (sample-accurate slice of the master, never the phone mic)
    master, sr = load_audio_native(song)
    i0 = int(round(s0 * sr))
    wav = write_wav(work / f"master_{s0:.4f}_{n}f.wav", master[i0:i0 + int(round(dur * sr))], sr)
    del master
    stages["prepare_s"] = round(time.time() - t, 1)

    # 9. composite in parallel units -> raw segments -> one file per look
    seg_dir = OUT / "work" / f"segments_{tag}"
    shutil.rmtree(seg_dir, ignore_errors=True)
    seg_dir.mkdir(parents=True)
    native_frames = np.array([take["native"] is not None and not is_identity(c) for c in crops], np.uint8)
    arrays = {"crops": crops, "wf": wf, "pushf": pushf, "shot_of": shot_of, "boxes": boxes,
              "shot_world": np.array([sh["world"] for sh in plan["shots"]]), "native_frames": native_frames}
    rid = LOOKS.index("receipt")
    for look in looks:
        lk = np.full(n, LOOKS.index(look), np.uint8)
        for r_ in plan["receipt_reveal"]:
            lk[r_["frames"][0]:r_["frames"][1] + 1] = rid
        arrays[f"look_{look}"] = lk
    on_receipt = np.any([arrays[f"look_{look}"] == rid for look in looks], axis=0)
    counter = np.zeros(n, np.uint8)
    if getattr(a, "receipt_counter", False):
        for sh in plan["shots"]:  # one place per shot, the one the artist covers least
            f0, f1 = sh["start_frame"], sh["end_frame"]
            if not on_receipt[f0:f1].any():
                continue
            outb = []
            for i in range(f0, f1, 3):
                bx = boxes[i] if not np.isnan(boxes[i, 0]) else (0, 0, 1080, 1920)
                x0, y0, cw, ch = crops[i]
                z = 1080 / cw
                outb.append(((bx[0] - x0) * z, (bx[1] - y0) * z, (bx[2] - x0) * z, (bx[3] - y0) * z))
            sh["counter_place"] = R.place_for(outb)
            counter[f0:f1] = np.where(on_receipt[f0:f1], sh["counter_place"], 0)
    arrays["counter"] = counter
    shot_ink = np.zeros(len(plan["shots"]), np.uint8)
    if any(LOOKS[int(v)] == "collage" for look in looks for v in np.unique(arrays[f"look_{look}"])):
        shot_ink = collage_inks(plan, worlds, masks, crops, wf, pushf, s0, a.push, style)
    arrays["shot_ink"] = shot_ink
    if style == "bullet":  # Hook Freeze v2: camera truck, motion-blur clock rate, the hit and the grade
        arrays["cam_dx"], arrays["cam_zoom"] = FZ.bullet_camera(n, plan, plan["freezes"], shot_of)
        arrays["vel"] = FZ.clock_rate(n, plan["freezes"], shot_of)
        arrays["grade"], arrays["punch"], arrays["flash"] = FZ.hit_profile(n, plan["freezes"])
    if any(m["kind"] == "swap" for m in moves):
        shot_w = np.array([sh["world"] for sh in plan["shots"]])[shot_of]
        def swap_ids(f):  # the section's own worlds; with only one, every world in the render (a montage)
            ids = per_section[plan["shots"][shot_of[f]]["section_pos"]]
            return ids if len(set(ids)) >= 2 else sorted(set(range(len(worlds))))
        arrays["frame_world"] = MV.swap_worlds(n, moves, shot_w, swap_ids)
    if any(m["kind"] == "crash" for m in moves):
        sc, fx, fy = MV.crash_profile(n, moves, boxes)
        ox, oy = np.empty(n), np.empty(n)
        for i in range(n):  # the face point in OUTPUT space (after the shot's reframe)
            x0, y0, cw, ch = crops[i]
            ox[i], oy[i] = (fx[i] - x0) * W / cw, (fy[i] - y0) * W / cw
        arrays["crash_scale"], arrays["crash_cx"], arrays["crash_cy"] = sc, ox, oy
    beat_designed = set()
    ent_g = None
    if getattr(a, "entrance", "auto") == "auto":  # the world stays dark until he steps in (realone/entrance.py)
        from realone import entrance as EN
        ent_g, plan["entrance"] = EN.plan(boxes, n, FPS)
        if plan["entrance"]["on"]:
            arrays["ent_gain"] = ent_g
            beat_designed = set(range(plan["entrance"]["entry_frame"], plan["entrance"]["lit_frame"] + 1))
            print(f"entrance: the world stays dark until he steps in at frame {plan['entrance']['entry_frame']} "
                  f"({plan['entrance']['entry_s']:.2f} s), lit by frame {plan['entrance']['lit_frame']}")
        else:
            ent_g = None
    if getattr(a, "hits", False):  # the frame kicks on the beat (realone/hits.py)
        from realone import hits as HT
        hp = HT.profile(tl, s0, n, FPS)
        arrays["hit_punch"], arrays["hit_flash"], arrays["hit_dx"], arrays["hit_dy"] = (
            hp["punch"], hp["flash"], hp["dx"], hp["dy"])
        beat_designed = set(beat_designed) | set(hp["designed"])
        plan["hits"] = {"frames_punched": int((hp["punch"] > 1.0).sum()), "flashes": int((hp["flash"] > 0).sum()),
                        "jolts": int((np.abs(hp["dx"]) > 0).sum())}
        print(f"hits on the beat: {plan['hits']}")
    lyric_lines = None
    mode = lyrics_mode(a)
    if mode != "off":  # the lyric lines written into the world, a word on each beat
        from realone import lyrictext as LT
        try:
            lyric_lines, lyr_arr, plan["lyrics_in_world"] = LT.for_render(getattr(a, "_treatment_lyrics", None),
                                                                          tl, s0, n, FPS,
                                                                          getattr(a, "_word_times", None),
                                                                          getattr(a, "_word_source", None))
        except LT.LyricError as e:
            if mode == "forced":
                raise Refuse(str(e)) from None
            plan["lyrics_in_world"] = {"off": str(e)}  # on by default only where there are words to show
            print(f"lyrics in the world: off ({e})")
        else:
            arrays.update(lyr_arr)
            plan["lyrics_in_world"]["mode"] = mode
            framing = [sh.get("framing") for sh in plan["shots"]]
            from realone import moments as MO
            # a line that starts while the world is still dark (the entrance) is a front card: a sign in a dark
            # world would be dark too
            # and while the dolly runs a sign standing in the stretching world is pushed out of frame ("PROM",
            # "T", "ART" fragments behind him on the verse, gauntlet rounds 1 and 4): those lines are cards too
            dolly_on = (MV.dolly_profile(n, moves, [sh["start_frame"] for sh in plan["shots"]]) > 0
                        if getattr(a, "stage", "off") == "tracked" else np.zeros(n, bool))
            arrays["lyr_card"] = LT.card_places(
                lyr_arr["lyr_line"], len(lyric_lines),
                lambda i: "close" if (ent_g is not None and ent_g[i] < 1) or dolly_on[i] else framing[int(shot_of[i])],
                lambda i: None if np.isnan(boxes[i, 0]) else MO.box_out(boxes[i], crops[i]))
            plan["lyrics_in_world"]["cards"] = sorted({int(g) for g in lyr_arr["lyr_line"][arrays["lyr_card"] > 0]})
            # a front card appearing, changing line or leaving changes a band of the frame on purpose: the cut
            # check must not read it as a cut (Nov 4: the intro's card leaving at frame 149 did)
            beat_designed = set(beat_designed) | LT.card_changes(lyr_arr["lyr_line"], arrays["lyr_card"])
            print(f"lyrics in the world ({mode}): {len(lyric_lines)} lines, {plan['lyrics_in_world']['words']} "
                  f"words, on {plan['lyrics_in_world']['frames_with_words']} of {n} frames")
    elif getattr(a, "_lyrics_off_why", None):
        plan["lyrics_in_world"] = {"off": a._lyrics_off_why}
        print(f"lyrics in the world: off ({a._lyrics_off_why})")
    slam_words, slam_colors = [], []
    from realone import lyrictext as LTX
    # the type takes each section's accent from the director's palette (realone/lyrictext.accent_of)
    accents = {i: LTX.accent_of(pal) for i, pal in (getattr(a, "_treatment_palettes", None) or {}).items()}
    line_sec = (plan.get("lyrics_in_world") or {}).get("line_section") or []
    lyric_colors = [accents.get(sec) for sec in line_sec] if lyric_lines is not None else None
    if getattr(a, "moments", False) or getattr(a, "moment", None):  # lyric moments (realone/moments.py)
        from realone import lyrictext as LT
        from realone import moments as MO
        if lyric_lines is not None:
            mo_lines, mo_line, mo_shown = lyric_lines, arrays["lyr_line"], arrays["lyr_shown"]
            mo_sec = plan["lyrics_in_world"].get("line_section")
        else:  # each word's beat is needed even when the words are not drawn
            try:
                mo_lines, lyr_arr, mo_rep0 = LT.for_render(getattr(a, "_treatment_lyrics", None), tl, s0, n, FPS,
                                                           getattr(a, "_word_times", None),
                                                           getattr(a, "_word_source", None))
            except LT.LyricError as e:
                raise Refuse(str(e).replace("--lyrics-in-world", "--moments")) from None
            mo_line, mo_shown = lyr_arr["lyr_line"], lyr_arr["lyr_shown"]
            mo_sec = mo_rep0.get("line_section")
        if getattr(a, "moment", None):
            chosen, source = [], "--moment (hand-picked)"
            for spec in a.moment:
                word, _, eff = spec.rpartition(":")
                if eff not in MO.EFFECTS or not word.strip():
                    raise Refuse(f"--moment {spec}: expected WORD:EFFECT with effect one of {', '.join(MO.EFFECTS)}")
                chosen.append({"word": word, "effect": eff})
        elif getattr(a, "_treatment_moments", None) is not None:
            t1 = s0 + n / FPS  # only the sections this render covers: a verse pick must not land on a hook word
            inside = [sec["index"] for sec in tl["sections"] if sec["start_s"] < t1 and sec["end_s"] > s0]
            chosen = [dict(m, section=i) for i in inside for m in a._treatment_moments.get(i, [])]
            source = "the director's pick (treatment `moments`)"
        else:
            chosen, source = None, "no director pick: words about fire ignite"
        picked = MO.pick(mo_lines, chosen, line_section=mo_sec)
        mo_arr, slam_words, mo_designed, mo_rep = MO.profile(picked, mo_line, mo_shown, n, boxes, line_section=mo_sec)
        slam_colors = [accents.get(m.get("section")) for m in mo_rep if m["effect"] == "slam"]
        arrays.update(mo_arr)
        beat_designed = set(beat_designed) | mo_designed  # a slam or an ignite flash changes the frame on purpose
        plan["moments"] = {"source": source, "moments": mo_rep}
        print(f"lyric moments ({source}): " + (", ".join(f"{m['word']} {m['effect']} @{m['frame']}" for m in mo_rep)
                                                or "none in this render"))
    if fx_list:  # the tracked effects (realone/fx.py): each drawn from its part's box, frame by frame
        from realone import fx as FX
        fx_arr, fx_designed, plan["fx"] = FX.plan(fx_list, s0, n, lambda tr: fx_boxes[tr], FPS)
        arrays.update(fx_arr)
        beat_designed = set(beat_designed) | fx_designed
        print("tracked effects: " + ", ".join(f"{e['effect']} on '{e['track']}' @{e['at_s']:.2f}s"
                                             + ("" if e["placed"] else f" ({e['why']})") for e in plan["fx"]))
    stage_rep = None
    if getattr(a, "stage", "off") == "tracked":
        from realone.stage import pipeline as STG
        lens_mm, lens_src = STG.lens_for(STG.clip_tags(clip), getattr(a, "lens_mm", None))
        st_arr, stage_rep = STG.prepare(norm, n, masks, worlds, plan, shot_of, lens_mm, OUT / "work" / "depth",
                                        figure_m=getattr(a, "figure_m", 1.75),
                                        locked_camera_m=getattr(a, "camera_m", 1.3),
                                        world_of=arrays.get("frame_world"),
                                        dolly_dz=MV.dolly_profile(n, moves, [sh["start_frame"] for sh in
                                                                             plan["shots"]]))
        stage_rep["lens_source"] = lens_src
        arrays.update(st_arr)
        # the real floor out of the matte at his feet (clean plate from the tracked floor): every later stage,
        # the Receipt included, uses the refined matte
        from realone.stage.cleanplate import refine_masks_cached
        mpath, removed = refine_masks_cached(norm, n, masks, mpath, st_arr["stage_H"], st_arr["stage_contact"],
                                             capture_shadow=getattr(a, "contact", "synthetic") == "real")
        masks = np.load(mpath, mmap_mode="r")
        stage_rep["feet_px_removed"] = {"frames": int((removed > 0).sum()), "total": int(removed.sum())}
        if getattr(a, "contact", "synthetic") == "real":
            stage_rep["shadow_path"] = str(Path(str(mpath)).with_name(Path(str(mpath)).stem + "_shadow.npy"))
        plan["stage"] = {k: v for k, v in stage_rep.items() if k != "K"}
    plan["living_world"] = ("off (--still-world)" if getattr(a, "still_world", False) else
                            "on: the world's lights flicker and swell on the kick, haze drifts (world clock)")
    write_json(shots_path, plan)  # again: now with each shot's counter place and collage ink
    np.savez(seg_dir / "arrays.npz", **arrays)
    units = U.make_units(plan, workers, avoid=[f["end_frame"] for f in plan["freezes"]])
    workers = min(workers, len(units))
    pockets = {"on": True, "off": False}.get(getattr(a, "pockets", "auto"), local)  # auto: only the gap-filling
    ctx = {"args": dict({k: getattr(a, k) for k in ("erode", "feather", "wrap", "match", "grain", "despill", "push")},
                        pockets=pockets),
           "proxy": str(norm), "native": str(take["native"]) if take["native"] else None,
           "native_size": take["native_size"], "ns": list(take["ns"]), "masks": str(mpath),
           "plates": [str(p) for _, p, _ in worlds], "particles": [k for _, _, k in worlds], "freeze_style": style,
           "wet_worlds": wet_worlds,
           "arrays": str(seg_dir / "arrays.npz"), "look_names": list(LOOKS), "outputs": looks,
           "s0": s0, "g0": int(round(s0 * FPS)), "seg_dir": str(seg_dir), "final": a.final,
           "cv_threads": max(1, (os.cpu_count() or 4) // workers),
           "stage": stage_rep, "stage_cache": str(OUT / "work" / "depth"), "lyric_lines": lyric_lines,
           "slam_words": slam_words, "slam_colors": slam_colors, "lyric_colors": lyric_colors,
           "beat_grid": None if getattr(a, "still_world", False) else
           {"beats": tl.get("beats") or [], "downbeats": tl.get("downbeats") or [],
            "beats_per_bar": tl.get("beats_per_bar") or 4}}
    if stage_rep is not None:
        ctx["args"]["stage"] = "tracked"
    if os.environ.get("REALONE_KEEP_WORK"):  # debug: keep what a lab needs to rebuild any frame without a render
        keep = OUT / "work" / f"keep_{tag}"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seg_dir / "arrays.npz", keep / "arrays.npz")
        (keep / "ctx.json").write_text(json.dumps(ctx, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        print(f"kept {keep}")
    print(f"render: {n} frames, {len(plan['shots'])} shots, {len(units)} units on {workers} worker(s) ({why})")
    t0 = time.time()
    recs = run_units(ctx, units, workers, a.max_minutes * 60 - (t0 - t_start), t0, n)
    render_s = time.time() - t0
    t = time.time()
    outs = {look: OUT / f"{tag}_{look}.mp4" for look in looks}
    for look in looks:
        U.concat_mux([r["segments"][look] for r in recs], wav, outs[look], n)
    stages["render_s"] = round(render_s, 1)
    stages["concat_s"] = round(time.time() - t, 1)
    keys, plate_steps = [], np.full(n, np.nan)
    state_keys, states, cam_used = [], [], np.zeros(n)
    bbox_out, pocket = np.zeros((n, 4)), np.zeros(n, int)
    sources = {}
    for r in recs:
        keys += r["keys"]
        state_keys += r["state_keys"]
        states += r["states"]
        cam_used[r["f0"]:r["f1"]] = r["cam"]
        plate_steps[r["f0"]:r["f1"]] = r["plate_steps"]
        bbox_out[r["f0"]:r["f1"]] = r["bbox"]
        pocket[r["f0"]:r["f1"]] = r["pocket"]
        for k, v in r["sources"].items():
            sources[k] = sources.get(k, 0) + v
    if len(keys) != n:
        raise Refuse(f"the units rendered {len(keys)} frames, the plan has {n}")
    filmed = {look: np.concatenate([r["filmed"][look] for r in recs]) for look in looks}
    look_runs = {}
    for look in looks:
        lk, runs = arrays[f"look_{look}"], []
        for i in range(n):
            if not runs or runs[-1]["look"] != LOOKS[lk[i]]:
                runs.append({"look": LOOKS[lk[i]], "frames": [i, i]})
            runs[-1]["frames"][1] = i
        look_runs[look] = runs
    receipt_path = R.write_json(OUT / f"{tag}_receipt.json", plan, filmed, look_runs, plan["receipt_reveal"],
                                bool(getattr(a, "receipt_counter", False)), s0, write_json)

    # 10. checks on what was actually written
    t = time.time()
    master22 = load_audio(song, SYNC_SR)
    report = {"version": __version__, "mode": "local test" if local else "demo", "engine": engine,
              "normalize": {k: take[k] for k in ("route", "coded", "rotation", "display", "transfer", "codec",
                                                 "pix_fmt", "fps", "native_size", "ns")},
              "clip": str(clip), "song": str(song), "section": plan["section"], "timeline": str(tl_path),
              "shots": str(shots_path), "render_s": round(render_s, 1), "fps": round(n / max(render_s, 1e-6), 2),
              "workers": workers, "workers_why": why, "units": [{k: r[k] for k in ("f0", "f1", "seconds", "pid", "maxrss_mb")}
                                            for r in recs],
              "billed_usd": round(billed, 4), "looks": {}}
    planned = [s["start_frame"] for s in plan["shots"][1:]]
    # a planned cut while the world is still dark (the entrance) changes a near-black frame: nothing a viewer or the
    # detector can see, so it is reported, not required (Nov 4: take 2 steps in at 6.1 s, after the intro's cut)
    from realone import entrance as EN
    in_dark = EN.dark_cuts(planned, ent_g)
    planned = [p for p in planned if p not in in_dark]
    all_good = True
    for look in looks:
        rep, good = verify_output(outs[look], dur)
        frames_out = count_frames(outs[look])
        cuts, diffs = detect_cuts(outs[look], quiet=None if ent_g is None else np.nonzero(ent_g < 1)[0])
        # the bullet freeze's hit (punch + flash on its downbeat, decaying over the 3 frames after; the downbeat
        # is not a cut in `bullet`) changes the whole frame like a cut does: a detection there is the hit
        hit = ({f["start_frame"] + j for f in plan["freezes"] for j in range(0, len(FZ.FLASH) + 1)}
               | {f["end_frame"] for f in plan["freezes"]}) if style == "bullet" else set()
        hit |= MV.designed_changes(moves)  # a crash snap or a world swap changes the whole frame on purpose
        hit |= beat_designed  # so does a punch on the beat
        # a planned cut is confirmed by the strict global rule, or (busy handheld takes) as a clear local peak;
        # where a crash snap lands on the same downbeat the neighbourhood is busier still, and 2x is the bar
        matched = [p for p in planned if any(abs(c - p) <= 1 for c in cuts)
                   or planned_cut_found(diffs, p, ratio=2.0 if p in hit else 2.5)]
        # (the release downbeat is designed too: the freeze's cool grade snaps off there, a whole-frame change the
        # cut detector reads as a cut when the release falls inside a shot rather than on a section cut)
        extra = [c for c in cuts if not any(abs(c - p) <= 1 for p in planned)
                 and c not in hit]  # a cut off the beat grid
        out_audio = load_audio(outs[look], SYNC_SR)
        av = measure_offset(master22, out_audio, near_s=s0)
        av_err_ms = (av["offset_s"] - s0) * 1000
        ok = (good and frames_out == n and len(matched) == len(planned) and not extra
              and abs(av_err_ms) <= 1000 / FPS)
        all_good &= ok
        report["looks"][look] = {"video": str(outs[look]), "spec": rep, "spec_ok": good, "frames": frames_out,
                                 "frames_planned": n, "cuts_detected": cuts, "cuts_planned": planned,
                                 "cuts_in_the_dark": in_dark,
                                 "cuts_on_plan": f"{len(matched)}/{len(planned)}", "cuts_off_plan": extra,
                                 "detections_on_the_freeze_hit": sorted(c for c in cuts if c in hit),
                                 "audio_vs_master_error_ms": round(av_err_ms, 2), "ok": ok}
    # 10b. the freeze, proven on the world layer of every rendered frame
    if plan["freezes"]:
        art, ctl = [], []
        for f in plan["freezes"]:
            f0, f1 = f["start_frame"], f["end_frame"]
            art.append(not np.array_equal(masks[f0], masks[f1 - 1]))
            sh = plan["shots"][shot_of[f0]]
            span = max(1.0, wf[sh["end_frame"] - 1] - wf[sh["start_frame"]] + 1)
            plate, parts = load_plate(worlds[sh["world"]][1]), make_particles(worlds[sh["world"]][2], style)
            if style == "bullet" and parts.n:  # the particles one clock step on (what the freeze holds back)
                ctl.append(not np.array_equal(parts.state(s0 + wf[f0] / FPS), parts.state(s0 + (wf[f0] + 1) / FPS)))
                continue
            held = world_layer(plate, parts, crops[f0], pushf[f0], a.push, s0 + wf[f0] / FPS)
            free = world_layer(plate, parts, crops[f0], pushf[f0] + 1 / span, a.push, s0 + (wf[f0] + 1) / FPS)
            ctl.append(world_key(*held[:2]) != world_key(*free[:2]))
        if style == "bullet":
            sec_starts = sorted({sh["start_frame"] for k, sh in enumerate(plan["shots"])
                                 if k == 0 or sh["section_pos"] != plan["shots"][k - 1]["section_pos"]})
            if len(state_keys) != n:
                raise Refuse(f"the units recorded {len(state_keys)} world states, the plan has {n}")
            frep, fok = FZ.checks_bullet(keys, state_keys, FZ.particle_steps(states), cam_used, plate_steps, wf,
                                         plan["freezes"], shot_of, art, ctl,
                                         world_of=np.array([plan["shots"][k]["world"] for k in shot_of]),
                                         section_starts=sec_starts)
        else:
            frep, fok = FZ.checks(keys, plate_steps, wf, plan["freezes"], shot_of, art, ctl)
        report["freeze"] = frep
        report["freeze"]["lead_in"] = lead_in
        all_good &= fok
    stages["checks_s"] = round(time.time() - t, 1)

    # 11. contact sheets (read back from the encoded files)
    t = time.time()
    title = (f"{'TEST_LOCAL (never the demo)  ' if local else ''}Real One {__version__}  "
             f"{plan['section']['role']}: bar {plan['start_bar'] + 1} to {plan['end_bar']}  {tl['bpm']:.1f} BPM  "
             f"cut every {a.cut_bars} bars  sync {sy['offset_ms']:+.0f} ms")
    for look in looks:
        cs = OUT / f"{tag}_{look}_contact.jpg"
        if len(plan["shots"]) <= 12:
            contact_sheet(outs[look], plan, bbox_out, look, cs, title, names)
        else:
            song_sheet(outs[look], plan, cs, title + f"   look: {look}", names)
        report["looks"][look]["contact_sheet"] = str(cs)
        for k, f in enumerate(plan["freezes"]):
            fs = OUT / f"{tag}_{look}_freeze{k + 1}.jpg"
            freeze_sheet(outs[look], plan, f, wf, fs, title + f"   look: {look}   freeze: {style}", style=style)
            report["looks"][look].setdefault("freeze_sheets", []).append(str(fs))
    if len(looks) > 1 and len(plan["shots"]) <= 12:
        ls = OUT / f"{prefix}looks.jpg"
        looks_sheet(outs, plan, ls, title + "   rows: " + " / ".join(looks))
        report["looks_sheet"] = str(ls)
    stages["sheets_s"] = round(time.time() - t, 1)
    report["receipt"] = {"json": str(receipt_path), **{look: dict(zip(("filmed_pct", "generated_pct"),
                                                                    R.split(float(np.nanmean(filmed[look])))))
                                                        for look in looks}}
    report["frames_with_pocket_cut"] = int((pocket > 0).sum())
    report["artist_pixels_from"] = sources  # proxy | native | native+refined, frames each
    report["peak_rss_mb_python"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1)
    report["peak_rss_mb_worker"] = max(r["maxrss_mb"] for r in recs)
    report["stages"] = stages
    report["total_s"] = round(time.time() - t_start, 1)
    write_json(OUT / f"{tag}_run.json", report)
    if all_good:
        shutil.rmtree(seg_dir, ignore_errors=True)

    print("\n=== result ===")
    for look, r in report["looks"].items():
        print(f"{look:8s} {r['video']}\n         {r['spec']}\n         frames {r['frames']}/{n}, cuts on plan "
              f"{r['cuts_on_plan']}, audio vs master {r['audio_vs_master_error_ms']:+.1f} ms, "
              f"contact {r['contact_sheet']}")
    if "looks_sheet" in report:
        print(f"looks    {report['looks_sheet']}")
    for k, fr_ in enumerate(report.get("freeze", {}).get("freezes", [])):
        print(f"freeze   {fr_['role']} frames {fr_['frames']}: world {'state ' if 'camera_moves' in fr_ else ''}"
              f"identical on all {fr_['n_frames']} frozen frames: "
              f"{fr_['frozen_identical']}; moves before: {fr_['moves_before']}; resumes: {fr_['resumes']}; "
              f"no jump: {fr_['no_jump']}; artist moves: {fr_['artist_moves']}; "
              f"world would move unfrozen: {fr_['control_world_would_move']}"
              + (f"; camera moves {fr_['camera_moves']} ({fr_['camera_travel_px']} px), rendered world moves in the "
                 f"freeze {fr_['rendered_world_moves_in_freeze']}; no cut on the downbeat {fr_['no_cut_on_downbeat']}"
                 f"; same world through {fr_['same_world_through']}" if "camera_moves" in fr_ else "")
              + f"  -> {'OK' if fr_['ok'] else 'FAILED'}")
        for look in looks:
            print(f"         sheet {report['looks'][look].get('freeze_sheets', [''])[k]}")
    if "freeze" in report and not report["freeze"]["world_moves_outside_freezes"]:
        print(f"freeze   FAILED: the world stood still outside a freeze on frames "
              f"{report['freeze']['frames_that_should_move_but_did_not']}")
    print(f"render   {n} frames x {len(looks)} looks in {render_s:.0f}s ({report['fps']} fps) on {workers} worker(s); "
          f"total {report['total_s']}s  stages {stages}")
    for look in looks:
        rc = report["receipt"][look]
        print(f"receipt  {look}: filmed {rc['filmed_pct']}%  generated {rc['generated_pct']}% of the pixels over the "
              f"video; per frame in {receipt_path}")
    print(f"memory   python {report['peak_rss_mb_python']:.0f} MB, largest worker {report['peak_rss_mb_worker']:.0f} MB")
    print(f"masks    {engine}")
    print("cost     US$0.00 (local test, no API calls)" if local and not prompts else
          f"cost     billed this run US${billed:.4f} (about CA${billed * USD_TO_CAD:.2f})")
    if not all_good:
        raise Refuse("an output failed its checks (see above and the run json)")
    return report
