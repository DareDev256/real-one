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


def prepare(proxy, n, masks, worlds, plan, shot_of, lens_mm, cache_dir, figure_m=1.75, log=print):
    """-> (arrays dict for arrays.npz, report dict). worlds: [(name, still path, particles)]."""
    t0 = time.time()
    track = CAM.ground_track(read_frames(proxy), lambda i: _mask(masks, i))
    Hs = track["H_to0"]
    if len(Hs) != n:
        raise Refuse(f"stage: tracked {len(Hs)} frames, the take has {n}")
    K = CAM.intrinsics(vfov_916(lens_mm))
    kind, drift = CAM.classify_motion(Hs, K)
    n0 = CAM.solve_camera(Hs, K, 1.0)["n0"]
    l = np.linalg.inv(K).T @ n0
    horizon = float(-(l[0] * W / 2 + l[2]) / l[1])
    # camera height from the artist standing on the floor, measured on the frames where the whole figure is in
    hs = []
    for i in range(0, n, max(1, n // 24)):
        m = _mask(masks, i)
        ys = np.nonzero(m.any(1))[0]
        if len(ys) and ys.max() < H - 8 and ys.min() > 8:  # feet and head both inside the frame
            try:
                hs.append(CAM.height_from_figure(float(track["contact"][i][1]), float(ys.min()), horizon, figure_m))
            except Refuse:
                pass
    if not hs:
        raise Refuse("stage: the artist is never seen head to feet on the floor; cannot size the camera height "
                     "(film one wide moment with the feet in frame)")
    height = float(np.median(hs))
    R, t, res = CAM.solve_on_floor(Hs, K, n0, height)
    log(f"stage: floor tracked on {n} frames (median {int(np.median(track['step_inliers'][1:]))} points), "
        f"{kind} camera, lens {lens_mm:g} mm, horizon row {horizon:.0f}, camera {height:.2f} m above the floor")

    shot_world = np.array([sh["world"] for sh in plan["shots"]])
    world_of = shot_world[shot_of]
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
    from realone.stage.integrate import planted_from_track
    st_planted = planted_from_track(track["contact"], Hs)
    arrays = dict(stage_R=st_R, stage_t=st_t, stage_corr=st_corr, stage_contact=track["contact"], stage_G=st_G,
                  stage_H=Hs, stage_planted=st_planted)
    report = dict(lens_mm=lens_mm, vfov_deg=round(vfov_916(lens_mm), 2), motion=kind, drift_px=round(drift, 1),
                  horizon_row=round(horizon, 1), camera_height_m=round(height, 3), figure_m=figure_m,
                  track_points_median=int(np.median(track["step_inliers"][1:])) if n > 1 else 0,
                  floor_reproj_px_before_pin=[round(x, 2) for x in CAM.floor_reproj_err(Hs, K, R, t, n0, height)],
                  worlds=per_world, K=K.tolist(), height=height, seconds=round(time.time() - t0, 1))
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
        self.G = arrays.get("stage_G")
        self.planted = arrays.get("stage_planted")

    def key_for(self, wid):
        """(dx, colour) for the artist's key-light ramp when the world shows a clear sun, else None."""
        w = self.rep["worlds"][str(wid)]
        return (w["sun_dir"][0], tuple(w["key_colour"])) if w.get("sun_drop", 0) >= 0.3 else None

    def layer(self, wid, i, mask, parts, wt, cam, crop):
        from realone.composite import crop_matrix, is_identity
        from realone.stage import integrate as IG
        g = self.get(wid)
        w = self.rep["worlds"][str(wid)]
        img, zb, fv = g.render(self.R[i], self.t[i], self.corr[i])
        sh = IG.contact_shadow(mask, fv, light_dir=tuple(w["sun_dir"]), length=w["sun_len"], drop=w["sun_drop"],
                               planted=float(self.planted[i]) if self.planted is not None else 0.0)
        img = img * (1 - sh[..., None])
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
        img = IG.add_grain(img, w.get("grain", 0.0), i)  # after the punch-in: grain is the sensor's, not the world's
        back, front = parts.layers(wt) if cam is None else parts.layers(wt, cam=cam[0], vel=cam[1])
        base = np.clip(img, 0, 255).astype(np.float32)
        return np.clip(base + back * 255, 0, 255), front * 255, base

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
