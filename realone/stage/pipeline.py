"""Stage in the render: track the take's floor once, solve its camera, and give every frame of every shot the pose
and floor pin it needs to film that shot's world (realone/stage/world3d.py). Workers rebuild each World3D from the
still, its cached depth and the numbers here, so nothing large crosses a process boundary.

`--stage tracked` refuses when the floor cannot be tracked or the still cannot cover the camera; nothing falls
back silently. The world clock (Hook Freeze) never stops the tracked camera: holding the camera while the artist's
real camera moves would slide the floor under his feet, the exact tell the stage exists to remove. In a freeze the
particles hang and the world's camera keeps following the phone."""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from realone.common import H, W, Refuse
from realone.media import read_frames
from realone.stage import camera as CAM
from realone.stage.depth import world_disparity
from realone.stage.world3d import World3D, fit_still_lens

# 35 mm-equivalent focal length of the main (1x) camera in VIDEO, by model prefix; stabilisation crops a few %
MAIN_LENS_MM = {"iPhone 1": 26.0}   # iPhone 11 to 17 main cameras: 24 to 26 mm stills, ~26 mm in video
DEFAULT_LENS_MM = 26.0


def vfov_916(lens_mm):
    """Vertical FOV of an upright 9:16 frame from a 35 mm-equivalent focal length (long side = 36 mm)."""
    return float(np.degrees(2 * np.arctan(18.0 / lens_mm)))


def clip_tags(path):
    """The container's format tags (an iPhone writes com.apple.quicktime.model); {} when there are none."""
    import subprocess
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(path)],
                             check=True, capture_output=True, text=True).stdout
        return json.loads(out).get("format", {}).get("tags", {}) or {}
    except (subprocess.CalledProcessError, ValueError):
        return {}


def lens_for(take_tags, override_mm=None):
    if override_mm:
        return float(override_mm), "--lens-mm"
    model = (take_tags or {}).get("com.apple.quicktime.model", "")
    for k, mm in MAIN_LENS_MM.items():
        if model.startswith(k):
            return mm, f"{model} main camera"
    return DEFAULT_LENS_MM, "default (phone main camera)"


def _mask(masks, i):
    return np.unpackbits(masks[i])[: W * H].reshape(H, W).astype(bool)


GROUND_CACHE_V = 1  # bump when CAM.ground_track changes what it computes


def take_key(proxy, masks, n):
    """A short key for "this take's proxy with these masks": the proxy file's name, size and mtime, plus a sample of
    64 mask frames (a re-cut matte changes the key)."""
    import hashlib
    st = Path(proxy).stat()
    h = hashlib.blake2b(digest_size=8)
    h.update(f"v{GROUND_CACHE_V}|{Path(proxy).name}|{st.st_size}|{st.st_mtime_ns}|{n}|{tuple(masks.shape)}".encode())
    for i in range(0, n, max(1, n // 64)):
        h.update(np.ascontiguousarray(masks[i]).tobytes())
    return h.hexdigest()


def ground_track_cached(proxy, n, masks, cache_dir, log=print):
    """The floor track depends only on the take and its masks, never on the treatment, so it is computed once per
    take and reused by every later render (Nov 4: the stage prep of a 1,646-frame take took 43 min under memory
    pressure before a single frame rendered, and a gauntlet re-renders the same take all day)."""
    p = Path(cache_dir) / f"ground_{take_key(proxy, masks, n)}.npz"
    if p.exists():
        z = np.load(p)
        log(f"stage: floor track reused ({p.name})")
        return {k: z[k] for k in z.files}
    track = CAM.ground_track(read_frames(proxy), lambda i: _mask(masks, i))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, **track)
    tmp.rename(p)
    return track


def _measure_look(proxy, frames, masks, world_full, R, t, corr, contact):
    """On up to 5 of the world's frames: the take's own background/performer focus ratio (the depth of field the
    world must match), its grain, and the world's key light read off the still.
    -> dict(k_dof, focus_take, focus_comp, grain, sun_dir, sun_len, sun_drop)."""
    from realone.stage import integrate as IG
    pick = set(frames[:: max(1, len(frames) // 5)].tolist())
    samples = []
    for i, fr in enumerate(read_frames(proxy)):
        if i > max(pick):
            break
        if i not in pick:
            continue
        m = _mask(masks, i)
        img, zb, _ = world_full.render(R[i], t[i], corr[i])
        cx, cy = contact[i]
        zf = (1.0 / max(float(zb[int(np.clip(cy - 4, 0, H - 1)), int(np.clip(cx, 0, W - 1))]), 1e-3)
              if np.isfinite(cx) else 3.0)
        samples.append((cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255, m, img, zb, zf))
    k, ft, fc = IG.calibrate_defocus(samples)
    g0, m0 = samples[0][0], samples[0][1]
    region = ~(cv2.dilate(m0.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0)
    region[: int(H * 0.35)] = False
    sun_dir, sun_len, sun_drop = IG.sun_from_still(np.clip(world_full.S, 0, 255).astype(np.uint8),
                                                   world_full.horizon_still)
    kcol = IG.key_colour(np.clip(world_full.S, 0, 255).astype(np.uint8), world_full.horizon_still)
    return dict(k_dof=int(k), focus_take=round(ft, 3), focus_comp=round(fc, 3), key_colour=list(kcol),
                grain=float(IG.estimate_grain(g0, region)), sun_dir=[float(x) for x in sun_dir],
                sun_len=float(sun_len), sun_drop=float(sun_drop))


def _figure_rows(masks, contact, n):
    """(feet row, head row) on frames where the whole figure is inside the frame, sampled ~24 times."""
    out = []
    for i in range(0, n, max(1, n // 24)):
        m = _mask(masks, i)
        ys = np.nonzero(m.any(1))[0]
        if len(ys) and ys.max() < H - 8 and ys.min() > 8:  # feet and head both inside the frame
            out.append((float(contact[i][1]), float(ys.min())))
    if not out:
        raise Refuse("stage: the artist is never seen head to feet on the floor; cannot size the camera "
                     "(film one wide moment with the feet in frame)")
    return out


def prepare(proxy, n, masks, worlds, plan, shot_of, lens_mm, cache_dir, figure_m=1.75, log=print, world_of=None,
            dolly_dz=None, locked_camera_m=1.3):
    """-> (arrays dict for arrays.npz, report dict). worlds: [(name, still path, particles)]."""
    t0 = time.time()
    track = ground_track_cached(proxy, n, masks, cache_dir, log)
    Hs = track["H_to0"]
    if len(Hs) != n:
        raise Refuse(f"stage: tracked {len(Hs)} frames, the take has {n}")
    K = CAM.intrinsics(vfov_916(lens_mm))
    kind, drift = CAM.classify_motion(Hs, K)
    rows = _figure_rows(masks, track["contact"], n)
    try:
        if kind == "locked":  # the frame-centre floor moved under 3 px all take: any solved normal would be noise
            raise Refuse("stage solve: the camera never moved enough (locked off)")
        n0 = CAM.solve_camera(Hs, K, 1.0)["n0"]
        l = np.linalg.inv(K).T @ n0
        horizon = float(-(l[0] * W / 2 + l[2]) / l[1])
        hs = []
        for feet, head in rows:
            try:
                hs.append(CAM.height_from_figure(feet, head, horizon, figure_m))
            except Refuse:
                pass
        if not hs:
            raise Refuse("stage: the artist's feet are never below the solved horizon")
        height = float(np.median(hs))
        R, t, res = CAM.solve_on_floor(Hs, K, n0, height)
    except Refuse as e:
        if "moved the camera enough" not in str(e) and "never moved enough" not in str(e):
            raise
        # a tripod: nothing to solve. The camera is level (no roll), `locked_camera_m` above the floor (a chest-high
        # tripod by default, --camera-m), and the horizon comes from the artist standing on the floor. The world
        # holds still under the floor track (the homographies are ~identity) and virtual moves run on top.
        kind = "locked"
        height = float(locked_camera_m)
        horizon = float(np.median([CAM.horizon_from_figure(f, hd, height, figure_m) for f, hd in rows]))
        n0 = CAM.level_floor_normal(horizon, K)
        R = np.repeat(np.eye(3)[None], n, 0)
        t = np.zeros((n, 3))
    log(f"stage: floor tracked on {n} frames (median {int(np.median(track['step_inliers'][1:])) if n > 1 else 0} "
        f"points), {kind} camera, lens {lens_mm:g} mm, horizon row {horizon:.0f}, camera {height:.2f} m above the "
        f"floor" + (" (tripod: height assumed, horizon from the figure)" if kind == "locked" else ""))

    if world_of is None:
        world_of = np.array([sh["world"] for sh in plan["shots"]])[shot_of]
    world_of = np.asarray(world_of)
    st_R = np.zeros((n, 3, 3))
    st_t = np.zeros((n, 3))
    st_corr = np.zeros((n, 3, 3))
    per_world = {}
    for wid in sorted(set(world_of.tolist())):
        frames = np.nonzero(world_of == wid)[0]
        C = np.array([-R[i].T @ t[i] for i in frames])
        fwd = np.array([0, 0, 1.0]) - n0 * n0[2]
        fwd /= np.linalg.norm(fwd)
        a = int(frames[int(np.argmin(C @ fwd))])  # the frame furthest back sees the most: the still is its view
        Ra, ta = R[a], t[a]
        poses = [(R[i] @ Ra.T, t[i] - (R[i] @ Ra.T) @ ta) for i in frames]
        still = cv2.imread(str(worlds[wid][1]))
        if still is None:
            raise Refuse(f"stage: cannot read the world still {worlds[wid][1]}")
        disp = world_disparity(still, cache_dir)

        def fe(g, j, a=a):
            return np.linalg.inv(Hs[frames[j]]) @ Hs[a] @ g.floor_h(np.eye(3), np.zeros(3))

        still_f, rep = fit_still_lens(still, disp, K, height, Ra @ n0, poses, fe)
        g = World3D(still, disp, K, height, Ra @ n0, still_f=still_f, geometry_only=True)
        for j, i in enumerate(frames):
            Rr, tr = poses[j]
            st_R[i], st_t[i] = Rr, tr
            st_corr[i] = fe(g, j) @ np.linalg.inv(g.floor_h(Rr, tr))
        full = World3D(still, disp, K, height, Ra @ n0, still_f=still_f)
        look = _measure_look(proxy, frames, masks, full, st_R, st_t, st_corr, track["contact"])
        per_world[str(int(wid))] = dict(anchor=a, still_f=still_f, n_anchor=(Ra @ n0).tolist(),
                                        horizon_still=round(g.horizon_still, 1), floor_fit_r=round(g.floor_r, 4),
                                        **look)
        log(f"stage: world {wid} '{worlds[wid][0]}': anchor frame {a}, still lens {still_f / K[0, 0]:.2f}x, "
            f"floor fit r={g.floor_r:.3f}, focus take {look['focus_take']} -> defocus k={look['k_dof']} "
            f"({look['focus_comp']}), shadow {np.round(look['sun_dir'], 2).tolist()}")
    # the floor's own motion between frames: the world is motion-blurred along it (exposure, not a filter)
    st_G = np.array([np.eye(3)] + [np.linalg.inv(Hs[i]) @ Hs[i - 1] for i in range(1, n)])
    arrays = dict(stage_R=st_R, stage_t=st_t, stage_corr=st_corr, stage_contact=track["contact"], stage_G=st_G,
                  stage_H=Hs)
    dolly_info = None
    if dolly_dz is not None and np.any(dolly_dz > 0):
        # the subject's distance per frame: where the ray through his contact point meets the solved floor,
        # smoothed like a focus pull (the lowest matte pixel hops between feet; a raw distance makes the zoom pop)
        from scipy.ndimage import gaussian_filter1d, median_filter
        Ki = np.linalg.inv(K)
        zs = np.full(n, np.nan)
        for i in range(n):
            cx, cy = track["contact"][i]
            if np.isfinite(cx):
                zs[i] = height / float((R[i] @ n0) @ (Ki @ np.array([cx, cy, 1.0])))
        good = np.isfinite(zs) & (zs > 0.3)
        if not good.any():
            raise Refuse("--move dolly: the artist's distance could not be read off the floor on any frame")
        zs = np.where(good, zs, np.median(zs[good]))
        arrays["stage_zs"] = gaussian_filter1d(median_filter(zs, size=31, mode="nearest"), 8, mode="nearest")
        arrays["stage_dz"] = np.asarray(dolly_dz, float)
        dolly_info = dict(max_m=float(np.max(dolly_dz)), subject_m=[round(float(arrays["stage_zs"].min()), 2),
                                                                   round(float(arrays["stage_zs"].max()), 2)])
    report = dict(lens_mm=lens_mm, vfov_deg=round(vfov_916(lens_mm), 2), motion=kind, drift_px=round(drift, 1),
                  horizon_row=round(horizon, 1), camera_height_m=round(height, 3), figure_m=figure_m,
                  track_points_median=int(np.median(track["step_inliers"][1:])) if n > 1 else 0,
                  floor_reproj_px_before_pin=[round(x, 2) for x in CAM.floor_reproj_err(Hs, K, R, t, n0, height)]
                  if kind != "locked" else None,
                  worlds=per_world, K=K.tolist(), height=height, seconds=round(time.time() - t0, 1))
    if dolly_info:
        report["dolly"] = dolly_info
    return arrays, report


class StageWorlds:
    """Worker side: one World3D per world id, built lazily from the prepared numbers, and the per-frame layer
    (world filmed through the take's camera + integration + particles + the shot's crop), the stage's stand-in
    for realone.worlds.world_layer with the same return shape."""

    def __init__(self, report, plates, cache_dir, arrays):
        self.rep, self.plates, self.cache = report, plates, cache_dir
        self.K = np.array(report["K"])
        self.built = {}
        self.R, self.t, self.corr = arrays["stage_R"], arrays["stage_t"], arrays["stage_corr"]
        self.contact = arrays["stage_contact"]
        self.dz = arrays.get("stage_dz")
        self.zs = arrays.get("stage_zs")
        self.G = arrays.get("stage_G")
        self.shadow = np.load(report["shadow_path"], mmap_mode="r") if report.get("shadow_path") else None

    def key_for(self, wid):
        """(dx, colour) for the artist's key-light ramp when the world shows a clear sun, else None."""
        w = self.rep["worlds"][str(wid)]
        return (w["sun_dir"][0], tuple(w["key_colour"])) if w.get("sun_drop", 0) >= 0.3 else None

    def layer(self, wid, i, mask, parts, wt, cam, crop, lyric=None):
        from realone.composite import crop_matrix, is_identity
        from realone.stage import integrate as IG
        g = self.get(wid)
        w = self.rep["worlds"][str(wid)]
        virt = None
        if self.dz is not None and self.dz[i] > 0:  # the dolly zoom: back dz, lens x (zs + dz) / zs
            dz, zs = float(self.dz[i]), float(self.zs[i])
            virt = (np.array([0.0, 0.0, dz]), (zs + dz) / zs)
        img, zb, fv = g.render(self.R[i], self.t[i], self.corr[i], virtual=virt)
        self.last_text = None
        if lyric is not None:  # the words on a sign standing in the world, warped like the world, hidden behind it
            self.last_text = self._sign(g, wid, i, zb, virt, *lyric)
        # contact: the synthetic occlusion + drop (the blind gauntlet's winner, rounds 4 to 7), or with
        # --contact real his real shadow lifted from the take (right under a LIFTED foot, but it loses the dense
        # pocket under a planted one: gauntlet r7 scored it 5 vs 6)
        ao = 0.0 if self.shadow is not None else 0.55
        drop = w["sun_drop"] if w.get("sun_drop", 0) >= 0.3 or self.shadow is None else 0.0
        sh = IG.contact_shadow(mask, fv, light_dir=tuple(w["sun_dir"]), length=w["sun_len"], ao=ao, drop=drop)
        img = img * (1 - sh[..., None])
        if self.shadow is not None:  # his real shadow from the take, laid on the world's floor (pinned to his)
            real = cv2.resize(self.shadow[i].astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_LINEAR)
            img = img * real[..., None]
        cx, cy = self.contact[i]
        if np.isfinite(cx):
            zf = 1.0 / max(float(zb[int(np.clip(cy - 4, 0, H - 1)), int(np.clip(cx, 0, W - 1))]), 1e-3)
        else:
            zf = 3.0
        if w["k_dof"]:
            img = IG.depth_defocus(img, zb, zf, coc_px_per_dioptre=w["k_dof"], max_coc=14.0, n_layers=8)
        if self.G is not None and i > 0:
            img = IG.motion_blur(img, self.G[i])
        if not is_identity(crop):
            M, _ = crop_matrix(crop)
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            if self.last_text is not None:
                rgb, al = self.last_text
                self.last_text = (cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR),
                                  cv2.warpAffine(al, M, (W, H), flags=cv2.INTER_LINEAR))
        img = IG.add_grain(img, w.get("grain", 0.0), i)  # after the punch-in: grain is the sensor's, not the world's
        back, front = parts.layers(wt) if cam is None else parts.layers(wt, cam=cam[0], vel=cam[1])
        base = np.clip(img, 0, 255).astype(np.float32)
        return np.clip(base + back * 255, 0, 255), front * 255, base

    def _sign(self, g, wid, i, zb, virt, line, shown, age):
        """-> (rgb, alpha) of frame i's words in frame coordinates, or None. Each line's sign stands in the world
        LT.SIGN_BEHIND_M behind where the artist stands at the middle of the line (his feet found on the tracked
        floor), facing that frame's camera, and fixed in the world from then on: the camera moves past it like
        anything else in the scene. The depth test hides it wherever the world is nearer (a pillar)."""
        from realone import lyrictext as LT
        if not hasattr(self, "sign_paint"):
            self.sign_paint, self.sign_depth = LT.sign_painter(self.lyric_lines), {}
        got = self.sign_paint.rgba(line, shown, age)
        if got is None:
            return None
        key = (wid, line)
        if key not in self.sign_depth:  # where this line's sign stands: behind him, where he is as the line starts
            self.sign_depth[key] = self._sign_spot(g, self.lyric_start.get(line, i), line)
        foot, facing, height_m, width_m, side = self.sign_depth[key]
        H_still, (m, dm), zc = g.sign_plane(foot, facing, height_m, LT.sign_yaw(side, shown, age), width_m,
                                            LT.SIGN_CANVAS)
        Hf = g.plane_frame_h(self.R[i], self.t[i], m, dm, self.corr[i], virt) @ H_still
        rgb, al = got
        rgb = cv2.warpPerspective(rgb, Hf, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        al = cv2.warpPerspective(al, Hf, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        zt = cv2.warpPerspective(zc, Hf, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        return rgb, LT.occlude(al, zb, zt)  # where the world is nearer, the sign is behind it

    def _sign_spot(self, g, i0, line=0):
        """-> (floor point, facing, height m, width m) for the sign of a line starting at frame i0. It stands on a
        vertical plane SIGN_BEHIND_M behind the artist's feet (the contact point on the tracked floor), facing that
        frame's camera. Lines change on the same downbeats as the shots, so each line is placed for its own shot:
        its centre where the ray through SIGN_TARGET of that shot's framing (after the punch-in) meets the plane,
        and SIGN_SHARE of that framing wide. From then on it is fixed in the world."""
        from realone import lyrictext as LT
        cx, cy = self.contact[i0]
        px = (cx, cy) if np.isfinite(cx) else (W / 2, H * 0.92)
        feet = g.floor_point(self.R[i0], self.t[i0], px)
        cam = g.camera_centre(self.R[i0], self.t[i0])
        if feet is None:  # the ray misses the floor (a frame looking up): a spot 3 m ahead along the floor
            n = g.nS / np.linalg.norm(g.nS)
            fwd = (self.R[i0] @ g.R_al).T @ np.array([0.0, 0.0, 1.0])
            fwd = fwd - (fwd @ n) * n
            feet = cam + (g.d - float(n @ cam)) * n + 3.0 * fwd / np.linalg.norm(fwd)
        facing = feet - cam
        n = g.nS / np.linalg.norm(g.nS)
        facing = facing - (facing @ n) * n
        facing /= max(np.linalg.norm(facing), 1e-6)
        q = feet + LT.SIGN_BEHIND_M * facing                         # a point on the sign's plane, on the floor
        crops = getattr(self, "crops", None)
        if crops is None:
            side = LT.sign_target(line)[0]
            return q, facing, LT.SIGN_HEIGHT_M, LT.SIGN_WIDTH_M, side
        x0, y0, cw, ch = (float(v) for v in crops[i0])
        boxes = getattr(self, "boxes", None)
        body = None
        if boxes is not None and np.isfinite(boxes[i0, 0]):
            body = ((float(boxes[i0, 0]) - x0) / cw, (float(boxes[i0, 2]) - x0) / cw)
        side, tx, ty, share = LT.sign_target(line, body)
        R = self.R[i0] @ g.R_al
        ray = R.T @ np.linalg.inv(self.K) @ np.array([x0 + tx * cw, y0 + ty * ch, 1.0])
        den = float(facing @ ray)
        if den <= 1e-6:
            return q, facing, LT.SIGN_HEIGHT_M, LT.SIGN_WIDTH_M, side
        P0 = cam + (float(facing @ (q - cam)) / den) * ray            # the sign's centre, on its plane
        height = float(np.clip(g.d - float(n @ P0), 0.6, 3.0))
        Z = float((R @ P0 + self.t[i0])[2])                           # its depth from that frame's camera
        width = float(np.clip(share * cw * Z / self.K[0, 0], 0.3, 3.0))
        foot = P0 + (g.d - float(n @ P0)) * n
        # measure it through the same warp the render uses (the floor pin can scale a shot by a lot), then size it
        # so it really spans SIGN_SHARE of that shot's width
        Hs, (m, dm), _ = g.sign_plane(foot, facing, height, LT.sign_yaw(side, 2, 99), width, LT.SIGN_CANVAS)
        Hf = g.plane_frame_h(self.R[i0], self.t[i0], m, dm, self.corr[i0]) @ Hs
        ends = Hf @ np.array([[0.0, LT.SIGN_CANVAS[0]], [LT.SIGN_CANVAS[1] / 2] * 2, [1.0, 1.0]])
        px = abs(ends[0, 1] / ends[2, 1] - ends[0, 0] / ends[2, 0])
        if np.isfinite(px) and px > 1:
            width = float(np.clip(width * share * cw / px, 0.3, 3.0))
        return foot, facing, height, width, side

    def get(self, wid):
        if wid not in self.built:
            w = self.rep["worlds"][str(wid)]
            still = cv2.imread(str(self.plates[wid]))
            disp = world_disparity(still, self.cache)
            self.built[wid] = World3D(still, disp, self.K, self.rep["height"], np.array(w["n_anchor"]),
                                      still_f=w["still_f"])
        return self.built[wid]


def save_report(path, report):
    Path(path).write_text(json.dumps(report, indent=1))
