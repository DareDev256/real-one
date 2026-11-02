"""Parallel render units (0.2.0): the frame range is cut into units (one per shot; a shot much longer than its
share is split, with a short colour-match warm-up), each unit is rendered by a worker process into one raw H.264
segment per output, and the segments are concatenated byte for byte and muxed once with the master audio.

Why it is frame-exact: every unit starts with an IDR frame and has no B-frames, so a segment is a self-contained
run of exactly (f1 - f0) frames; the muxer stamps the joined stream at a constant 30 fps (tested: 514 of 514
frames in order across five segments, one of them a single frame; mp4 segments through the concat demuxer
drifted after a 1-frame segment, and raw segments WITH B-frames reordered their first frames).

Why it is deterministic whatever the worker count: everything a frame needs is a function of its index (world
clock, camera crop, grain, collage boil step) except the colour-match EMA, which restarts at every unit; a cut
changes the world anyway, and a split unit warms the EMA on the frames before it.
"""
import hashlib
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from realone.common import FPS, H, W, unpack_mask
from realone.composite import INK, Compositor, lab_stats
from realone.media import read_frames
from realone.freeze import apply_hit
from realone.worlds import grade_world, load_plate, make_particles, plate_thumb, world_key, world_layer

WARM = 10          # frames of colour-match warm-up before a split unit
SEG_THREADS = 2    # x264 threads per segment encoder (the compositor, not the encoder, is the bottleneck)

_CTX = None


WORKER_GB = 0.95   # measured peak RSS of one worker: 0.67 to 0.95 GB (1080x1920; the 0.95 on the full song, 2026-10-29)
RESERVE_GB = 1.5   # left for everything else on the machine


def available_gb():
    """Memory the OS could hand out now without swapping: macOS free + inactive + speculative + purgeable pages
    (vm_stat); Linux MemAvailable. None if it cannot be read."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(re.search(r"page size of (\d+)", out).group(1))
            pages = {m.group(1): int(m.group(2)) for m in re.finditer(r"^Pages ([a-z ]+):\s+(\d+)\.", out, re.M)}
            return sum(pages.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable")) * page / 2 ** 30
        with open("/proc/meminfo") as f:
            return int(re.search(r"MemAvailable:\s+(\d+)", f.read()).group(1)) / 2 ** 20
    except Exception:
        return None


def default_workers():
    """CPU cores minus 2, and no more than free memory holds (a pool that pushes a shared 16 GB Mac into swap
    renders slower than a smaller one: measured 2026-10-29, 10 workers at 8 to 20% CPU each, kernel_task at
    275%, 12.5 of 13.3 GB swap used). -> (workers, why)."""
    cpu = max(1, (os.cpu_count() or 4) - 2)
    av = available_gb()
    if av is None:
        return cpu, f"cores-2 = {cpu} (free memory unreadable)"
    mem = max(1, int((av - RESERVE_GB) / WORKER_GB))
    return min(cpu, mem), f"cores-2 = {cpu}; memory allows {mem} ({av:.1f} GB reclaimable, {WORKER_GB} GB each)"


def make_units(plan, workers, avoid=()):
    """-> list of (f0, f1, warm). One unit per shot; a shot longer than 1.6x the per-worker share (and over 240
    frames) is split into near-equal pieces, never within 2 frames of a frame in `avoid` (a freeze release: its
    no-jump check compares the release with the frame before, which must come from the same worker). Sorted
    longest first (better packing for the pool)."""
    n = plan["n_frames"]
    share = max(1, n / max(1, workers))
    units = []
    for sh in plan["shots"]:
        f0, f1 = sh["start_frame"], sh["end_frame"]
        L = f1 - f0
        k = int(np.ceil(L / (1.6 * share))) if (L > 1.6 * share and L > 240) else 1
        edges = [f0 + round(L * j / k) for j in range(k + 1)]
        for j in range(1, k):
            while any(abs(edges[j] - r) <= 2 for r in avoid) and edges[j] + 3 < edges[j + 1]:
                edges[j] += 3
        for j in range(k):
            units.append((edges[j], edges[j + 1], WARM if j > 0 else 0))
    return sorted(units, key=lambda u: (-(u[1] - u[0]), u[0]))


def seg_encoder(path, final):
    """Raw H.264 (Annex B), no B-frames, no audio: one self-contained segment."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-an", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264",
           "-preset", "slow" if final else "veryfast", "-crf", "17" if final else "20", "-bf", "0",
           "-threads", str(SEG_THREADS), "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
           "-color_trc", "bt709", "-f", "h264", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def concat_mux(segments, audio, out, n_frames):
    """Join raw segments in order, stamp at 30 fps into a video-only mp4 (copy; BT.709 written into the SPS), then
    mux with the master audio (copy video, AAC audio). Two steps because a raw stream copied straight next to an
    audio input reaches the muxer without timestamps and -shortest then drops the audio (seen on 1080p)."""
    out = Path(out)
    joined, video = out.with_suffix(".joined.h264"), out.with_suffix(".video.mp4")
    with open(joined, "wb") as dst:
        for s in segments:
            with open(s, "rb") as src:
                shutil.copyfileobj(src, dst, 1 << 22)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-f", "h264", "-i", str(joined),
                    "-c:v", "copy", "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:"
                    "matrix_coefficients=1:video_full_range_flag=0", "-frames:v", str(n_frames), str(video)],
                   check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(audio), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(out)],
                   check=True)
    joined.unlink(missing_ok=True)
    video.unlink(missing_ok=True)
    return out


class Ctx:
    """Everything a worker needs, built once per process from a picklable dict."""

    def __init__(self, d):
        self.d = d
        cv2.setNumThreads(d.get("cv_threads", 2))
        self.args = type("A", (), d["args"])()
        self.comp = Compositor(self.args)
        self.masks = np.load(d["masks"], mmap_mode="r")
        self.plates = [load_plate(p) for p in d["plates"]]
        self.style = d.get("freeze_style", "still")
        self.parts = [make_particles(k, self.style) for k in d["particles"]]
        self.wstats = []
        for pl in self.plates:
            small = cv2.resize(pl, (270, 480), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            self.wstats.append(lab_stats(small))
        arr = np.load(d["arrays"])
        self.crops, self.wf, self.pushf = arr["crops"], arr["wf"], arr["pushf"]
        self.shot_of, self.shot_world = arr["shot_of"], arr["shot_world"]
        self.boxes = arr["boxes"]
        self.look_of = {k[5:]: arr[k] for k in arr.files if k.startswith("look_")}  # output -> per-frame look id
        self.look_names = list(d["look_names"])
        self.native_frames = arr["native_frames"]  # per frame: 1 if the punch-in uses the native file
        self.counter = arr["counter"] if "counter" in arr.files else np.zeros(len(self.wf), np.uint8)
        self.shot_ink = arr["shot_ink"] if "shot_ink" in arr.files else np.zeros(len(self.shot_world), np.uint8)
        # the tracked stage (realone/stage): the world filmed through the take's own solved camera
        self.stage = None
        if d.get("stage"):
            from realone.stage.pipeline import StageWorlds
            self.stage = StageWorlds(d["stage"], d["plates"], d["stage_cache"],
                                     {k: arr[k] for k in ("stage_R", "stage_t", "stage_corr", "stage_contact",
                                                          "stage_G", "stage_dz", "stage_zs") if k in arr.files})
        # camera moves on the beat (realone/stage/moves.py): a world per frame (swap), a crash zoom per frame
        self.frame_world = arr["frame_world"] if "frame_world" in arr.files else None
        # hits on the beat (realone/hits.py): punch, flash and jolt per frame, on the finished frame
        self.beat = ((arr["hit_punch"], arr["hit_flash"], arr["hit_dx"], arr["hit_dy"])
                     if "hit_punch" in arr.files else None)
        # lyric moments (realone/moments.py): ignite (frames since, which, the artist's box) and slam (which, age)
        self.mo = ((arr["mo_ign"], arr["mo_ign_id"], arr["mo_ign_box"], arr["mo_slam"], arr["mo_slam_age"])
                   if "mo_ign" in arr.files else None)
        self.slam_words = d.get("slam_words") or []
        self.bursts, self.slam_painter = {}, None
        # lyrics in the world (realone/lyrictext.py): which line, how many words, how long since the last landed
        self.lyrics = None
        if "lyr_line" in arr.files and d.get("lyric_lines"):
            self.lyrics = (arr["lyr_line"], arr["lyr_shown"], arr["lyr_age"])
            self.lyric_lines = d["lyric_lines"]
            self.painters = {}
            ln = self.lyrics[0]  # each line's middle frame: every worker anchors a line's sign at the same place,
            # placed for the middle of the line so the camera's drift over the line splits both ways
            self.lyric_start = {int(k): int((np.nonzero(ln == k)[0][0] + np.nonzero(ln == k)[0][-1]) // 2)
                                for k in np.unique(ln[ln >= 0])}
        self.crash = (arr["crash_scale"], arr["crash_cx"], arr["crash_cy"]) if "crash_scale" in arr.files else None
        # Hook Freeze v2 (`bullet`): the camera truck, the motion-blur clock rate and the hit, per frame
        self.bullet = "cam_dx" in arr.files
        if self.bullet:
            self.cam_dx, self.cam_zoom, self.vel = arr["cam_dx"], arr["cam_zoom"], arr["vel"]
            self.grade, self.punch, self.flash = arr["grade"], arr["punch"], arr["flash"]


def _with_lyrics(c, i, wid, world):
    """Blend frame i's lyric words into the world layer. On the tracked stage they are painted on the backdrop (the
    still's own coordinates, warped by the backdrop's homography and the shot's punch-in), so they move like the far
    scene; without the stage they sit in screen space. Either way the artist is composited over them."""
    from realone import lyrictext as LT
    if c.stage is not None:  # the stage already painted the sign for this frame (pipeline.StageWorlds._sign)
        got = c.stage.last_text
    else:  # no stage: the words sit in screen space, still behind the artist
        if "screen" not in c.painters:
            c.painters["screen"] = LT.LyricPainter(c.lyric_lines, W, H)
        got = c.painters["screen"].rgba(*(int(a[i]) for a in c.lyrics))
    return world if got is None else LT.blend(world, *got)


def init_worker(d):
    global _CTX
    _CTX = Ctx(d)


def _looks_for(c, i):
    return {out: c.look_names[int(c.look_of[out][i])] for out in c.d["outputs"]}


def render_unit(u):
    """Render frames [f0, f1) of every output into raw segments. -> per-frame records for the checks."""
    c = _CTX
    d = c.d
    f0, f1, warm = u
    t0 = time.time()
    s0, g0 = d["s0"], d["g0"]
    c.comp.match = {k: type(v)(c.args.match) for k, v in c.comp.match.items()}  # a unit starts its own EMA
    if c.comp.counter is not None:
        c.comp.counter.reset()  # and picks its counter ink afresh (a unit starts on a cut)
    need_native = d["native"] is not None and bool(c.native_frames[max(0, f0 - warm):f1].any())
    start = max(0, f0 - warm)
    prox = read_frames(d["proxy"], start=start)
    nat = read_frames(d["native"], tuple(d["native_size"]), start=start, scale=False) if need_native else None
    segs = {out: Path(d["seg_dir"]) / f"{out}_{f0:06d}.h264" for out in d["outputs"]}
    encs = {out: seg_encoder(p, d["final"]) for out, p in segs.items()}
    n = f1 - f0
    rec = {"f0": f0, "f1": f1, "keys": [], "plate_steps": np.full(n, np.nan), "bbox": np.zeros((n, 4)),
           "pocket": np.zeros(n, int), "filmed": {out: np.zeros(n) for out in d["outputs"]}, "sources": {},
           "segments": {k: str(v) for k, v in segs.items()}, "state_keys": [], "states": [], "cam": np.zeros(n)}
    prev_thumb, wcache = None, None
    try:
        for i in range(start, f1):
            fr = next(prox)
            nf = next(nat) if nat is not None else None
            sh_world = int(c.shot_world[c.shot_of[i]] if c.frame_world is None else c.frame_world[i])
            wt = s0 + c.wf[i] / FPS
            crop = c.crops[i]
            cam = (float(c.cam_dx[i]), float(c.vel[i]), float(c.cam_zoom[i])) if c.bullet else None
            wkey = (sh_world, float(wt), float(c.pushf[i]), tuple(crop), cam)  # exactly world_layer's inputs
            m = unpack_mask(c.masks[i])
            if c.stage is not None:  # every frame has its own camera pose: nothing to cache
                lyr = None
                if c.lyrics is not None and c.lyrics[0][i] >= 0:
                    c.stage.lyric_lines = c.lyric_lines
                    c.stage.lyric_start = c.lyric_start
                    c.stage.crops = c.crops
                    c.stage.boxes = c.boxes
                    lyr = tuple(int(x[i]) for x in c.lyrics)
                world, front, wbase = c.stage.layer(sh_world, i, m, c.parts[sh_world], wt, cam, crop, lyric=lyr)
                c.comp.key = c.stage.key_for(sh_world)
            else:
                if wcache is None or wcache[0] != wkey:
                    wcache = (wkey,) + world_layer(c.plates[sh_world], c.parts[sh_world], crop, c.pushf[i],
                                                   c.args.push, wt, cam=cam)
                _, world, front, wbase = wcache
            if c.bullet and c.grade[i] > 0:  # the frozen world grades cool; the artist is composited in colour
                world, front = grade_world(world, float(c.grade[i])), grade_world(front, float(c.grade[i]))
            wk = world_key(world, front)  # the freeze proof is the world itself, taken before any lyrics
            if c.lyrics is not None and c.lyrics[0][i] >= 0:
                world = _with_lyrics(c, i, sh_world, world)
            if c.mo is not None and c.mo[0][i] >= 0:  # an ignite: embers rise behind him, the world takes the heat
                from realone import moments as MO
                j = int(c.mo[1][i])
                burst = c.bursts.setdefault(j, MO.EmberBurst(j))
                world, front = MO.ignite(world, front, int(c.mo[0][i]), MO.box_out(c.mo[2][i], crop), burst)
            per_out = _looks_for(c, i)
            looks = sorted(set(per_out.values()))
            use_nat = nf if c.native_frames[i] else None
            frames, inf = c.comp.frame(fr, m, crop, world, front, c.wstats[sh_world], i, g0 + i, looks,
                                       native=use_nat, ns=tuple(d["ns"]), counter=int(c.counter[i]),
                                       ink=INK if c.shot_ink[c.shot_of[i]] else None)
            if i < f0:
                continue  # warm-up: the colour-match EMA only
            j = i - f0
            rec["keys"].append(wk)
            if c.bullet:  # the world-space state this frame was drawn from: particle positions, plate clock, push
                st = c.parts[sh_world].state(wt)
                rec["states"].append(st.astype(np.float32))
                rec["state_keys"].append(hashlib.md5(st.tobytes() + np.float64([sh_world, wt, c.pushf[i]]).tobytes())
                                         .hexdigest())
                rec["cam"][j] = cam[0]
            th = plate_thumb(wbase)
            if prev_thumb is not None:
                rec["plate_steps"][j] = float(np.abs(th - prev_thumb).mean())
            prev_thumb = th
            if c.mo is not None and c.mo[3][i] >= 0:  # a slam: the word in front of him, never on the Real Receipt
                from realone import moments as MO
                if c.slam_painter is None:
                    c.slam_painter = MO.SlamPainter(c.slam_words)
                sj, sa = int(c.mo[3][i]), int(c.mo[4][i])
                frames = {lk: fr_ if lk == "receipt" else c.slam_painter.apply(fr_, sj, sa)
                          for lk, fr_ in frames.items()}
            if c.bullet:  # the punch and the flash on the freeze downbeat, once per look
                frames = {lk: apply_hit(fr_, float(c.punch[i]), float(c.flash[i])) for lk, fr_ in frames.items()}
            if c.crash is not None and c.crash[0][i] > 1.0001:  # the crash zoom: the whole finished frame
                from realone.stage.moves import apply_crash
                sc, cx, cy = c.crash
                frames = {lk: apply_crash(fr_, float(sc[i]), float(cx[i]), float(cy[i]),
                                          float(sc[i - 1]) if i > 0 else 1.0) for lk, fr_ in frames.items()}
            if c.beat is not None:  # the frame kicks on the beat, after everything else (Receipt share unchanged)
                from realone import hits as HT
                bp, bf, bx_, by_ = (float(a_[i]) for a_ in c.beat)
                frames = {lk: HT.apply(fr_, bp, bf, bx_, by_) for lk, fr_ in frames.items()}
            for out, look in per_out.items():
                encs[out].stdin.write(frames[look].tobytes())
                rec["filmed"][out][j] = inf[look].get("filmed", np.nan)
            x0, y0, cw, ch = crop
            z = W / cw
            bx = c.boxes[i] if not np.isnan(c.boxes[i, 0]) else (0, 0, W, H)
            rec["bbox"][j] = ((bx[0] - x0) * z, (bx[1] - y0) * z, (bx[2] - x0) * z, (bx[3] - y0) * z)
            first = inf[looks[0]]
            rec["pocket"][j] = first.get("pocket_px", 0)
            src = first.get("source", "proxy")
            rec["sources"][src] = rec["sources"].get(src, 0) + 1
    finally:
        prox.close()
        if nat is not None:
            nat.close()
        for e in encs.values():
            e.stdin.close()
        rcs = [e.wait() for e in encs.values()]
    if any(rcs):
        raise RuntimeError(f"segment encode failed for frames {f0}-{f1}: {rcs}")
    rec["seconds"] = round(time.time() - t0, 2)
    rec["maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1)  # bytes on macOS
    rec["pid"] = os.getpid()
    return rec
