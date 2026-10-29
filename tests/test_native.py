"""Native-resolution punch-ins and the normalize routes: a close shot cut from a 4K frame has measurably more
detail and a truer mask edge than the 0.1.0 upscaled proxy; the native crop lands exactly where the proxy crop
does; the proxy and native files are frame-aligned; seeks are frame-exact; an HDR HLG take with a -90 rotation
comes out upright, SDR BT.709, 30 fps."""
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import make_test_clip as mtc
from realone import composite as C
from realone import native as N
from realone.common import FPS, H, W
from realone.media import count_frames, probe, read_frames, stream


def args(**kw):
    d = dict(erode=2, feather=1.4, wrap=0.6, match=0.35, grain=0.0, despill=True)
    d.update(kw)
    return SimpleNamespace(**d)


def make_4k(t=0.8, weave=None, noise=None):
    """One 2160x3840 frame of the test figure, its 1080 proxy, the proxy mask and the 4K truth mask."""
    old = mtc.WEAVE_AMP, mtc.NOISE_SIGMA
    try:
        mtc.WEAVE_AMP = old[0] if weave is None else weave
        mtc.NOISE_SIGMA = old[1] if noise is None else noise
        f4 = mtc.figure(t, None, (2160, 3840))
    finally:
        mtc.WEAVE_AMP, mtc.NOISE_SIGMA = old
    bg = mtc.background(t, (2160, 3840))
    a = f4[..., 3:4].astype(np.float32) / 255
    frame4 = np.clip(f4[..., :3] * a + bg * (1 - a), 0, 255).astype(np.uint8)
    proxy = cv2.resize(frame4, (W, H), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(f4[..., 3], (W, H), interpolation=cv2.INTER_AREA) > 127
    return frame4, proxy, mask, f4[..., 3] > 127


def close_crop(mask, z=1.9, ty=0.17):
    ys, xs = np.nonzero(mask)
    cw, ch = W / z, H / z
    cx = (xs.min() + xs.max()) / 2
    y = ys.min() + ty * (ys.max() - ys.min())
    return (float(np.clip(cx - cw / 2, 0, W - cw)), float(np.clip(y - ch / 2, 0, H - ch)), cw, ch)


def render_receipt(frame4, proxy, mask, crop, native):
    comp = C.Compositor(args())
    world = np.full((H, W, 3), 128, np.float32)
    small = cv2.resize(world, (270, 480)) / 255
    outs, info = comp.frame(proxy, mask, crop, world, np.zeros_like(world), C.lab_stats(small), 0, 0, ["receipt"],
                            native=frame4 if native else None, ns=(2.0, 2.0))
    return outs["receipt"], info["receipt"], comp


def lapvar(img, region):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F)[region].var())


def truth_in_crop(truth4, crop):
    t = N.native_crop(np.repeat(truth4[..., None].astype(np.uint8) * 255, 3, 2), crop, (2.0, 2.0))
    return t[..., 0] > 0.5


@pytest.fixture(scope="module")
def scene4k():
    return make_4k()


def test_close_shot_from_native_has_more_detail_than_the_upscaled_proxy(scene4k):
    frame4, proxy, mask, truth4 = scene4k
    crop = close_crop(mask)
    gt = truth_in_crop(truth4, crop)
    inner = cv2.erode(gt.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0  # the artist, away from the matte
    old, info_old, _ = render_receipt(frame4, proxy, mask, crop, native=False)
    new, info_new, _ = render_receipt(frame4, proxy, mask, crop, native=True)
    assert info_old["source"] == "proxy" and info_new["source"] == "native+refined"
    v_old, v_new = lapvar(old, inner), lapvar(new, inner)
    assert v_new > 20 * v_old, (v_old, v_new)  # measured 2026-10-29: 6.8 vs ~1280 (weave + sensor noise)


def test_native_detail_without_the_native_only_texture_is_still_sharper():
    """Weave and sensor noise off: only the drawn edges (eyes, mouth, hair, hand) differ."""
    frame4, proxy, mask, truth4 = make_4k(weave=0.0, noise=0.0)
    crop = close_crop(mask)
    inner = cv2.erode(truth_in_crop(truth4, crop).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    ref = cv2.cvtColor((N.native_crop(frame4, crop, (2.0, 2.0)) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    grad = np.hypot(cv2.Sobel(ref.astype(np.float32), cv2.CV_32F, 1, 0), cv2.Sobel(ref.astype(np.float32), cv2.CV_32F, 0, 1))
    near_edges = inner & (cv2.dilate((grad > 40).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    assert near_edges.sum() > 5000
    old, _, _ = render_receipt(frame4, proxy, mask, crop, native=False)
    new, _, _ = render_receipt(frame4, proxy, mask, crop, native=True)
    v_old, v_new = lapvar(old, near_edges), lapvar(new, near_edges)
    print(f"edges only: whole artist {lapvar(old, inner):.1f} -> {lapvar(new, inner):.1f}; "
          f"near the drawn edges {v_old:.1f} -> {v_new:.1f}")
    assert v_new > 1.2 * v_old, (v_old, v_new)  # measured 2026-10-29: 39.0 -> 53.0 (1.36x)


def test_close_shot_mask_edge_is_closer_to_the_4k_truth(scene4k):
    frame4, proxy, mask, truth4 = scene4k
    crop = close_crop(mask)
    gt = truth_in_crop(truth4, crop)
    z = W / crop[2]
    M = np.float32([[z, 0, -z * crop[0]], [0, z, -z * crop[1]]])
    up = cv2.warpAffine(mask.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR) > 0.5
    fg = N.native_crop(frame4, crop, (2.0, 2.0))
    ref = N.refine_mask_in_crop(fg, mask, crop)
    e_up, e_ref = int((up ^ gt).sum()), int((ref ^ gt).sum())
    assert e_ref < 0.85 * e_up, (e_up, e_ref)  # measured: 0.55% -> 0.38% of the artist's area


@pytest.mark.parametrize("ns", [(2.0, 2.0), (1214 / W, 2160 / H)])
def test_native_crop_lands_where_the_proxy_crop_does(ns):
    """A dot at a known proxy position must come out at the same output pixel through either path (portrait 4K
    and the landscape 4K centre crop, whose x and y scales differ by 0.1%)."""
    nw, nh = int(round(W * ns[0])), int(round(H * ns[1]))
    px, py = 612.3, 940.7
    native = np.zeros((nh, nw, 3), np.uint8)
    cv2.circle(native, (int(round(px * ns[0] * 16)), int(round(py * ns[1] * 16))), int(6 * ns[0] * 16),
               (255, 255, 255), -1, cv2.LINE_AA, shift=4)
    proxy = cv2.resize(native, (W, H), interpolation=cv2.INTER_AREA)
    crop = (500.0, 800.0, W / 1.9, H / 1.9)
    z = 1.9
    a = N.native_crop(native, crop, ns)[..., 0]
    b = cv2.warpAffine(proxy.astype(np.float32) / 255, np.float32([[z, 0, -z * crop[0]], [0, z, -z * crop[1]]]),
                       (W, H), flags=cv2.INTER_CUBIC)[..., 0]

    def centroid(img):
        ys, xs = np.mgrid[0:H, 0:W]
        w = np.clip(img, 0, None)
        return (xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()

    (ax, ay), (bx, by) = centroid(a), centroid(b)
    assert abs(ax - bx) < 0.6 and abs(ay - by) < 0.6, ((ax, ay), (bx, by))
    assert abs(ax - (px - crop[0]) * z) < 1.2, ax


def test_seek_reads_exactly_the_frame_asked_for(tmp_path):
    """Workers start mid-file; the first frame read must be frame `start`, not its neighbour (B-frames, 1 s GOP)."""
    p = tmp_path / "count.mp4"
    n = 75
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "64x64",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "1",
                            "-g", "30", "-pix_fmt", "yuv420p", str(p)], stdin=subprocess.PIPE)
    for i in range(n):
        fr = np.zeros((64, 64, 3), np.uint8)
        fr[:, :] = (i * 3) % 256
        cv2.putText(fr, str(i), (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255 - (i * 3) % 256,) * 3, 2)
        enc.stdin.write(fr.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0
    seq = list(read_frames(p, (64, 64), scale=False))
    assert len(seq) == n
    for k in (1, 29, 30, 31, 47, 74):
        got = next(read_frames(p, (64, 64), start=k, scale=False))
        diffs = [np.abs(got.astype(int) - s.astype(int)).mean() for s in seq]
        assert int(np.argmin(diffs)) == k and diffs[k] == 0, (k, int(np.argmin(diffs)))


def _has_videotoolbox():
    out = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True).stdout
    return "videotoolbox" in out


@pytest.mark.skipif(not _has_videotoolbox(), reason="needs ffmpeg with VideoToolbox (macOS)")
def test_hdr_hlg_take_with_rotation_comes_out_upright_sdr_at_30fps(tmp_path):
    """A 10-bit HLG HEVC take, 59.94 fps, coded landscape with display rotation -90 (how an iPhone stores a
    portrait clip). A red block in the CODED bottom-left must end up top-left of the upright output."""
    raw = tmp_path / "hlg.mov"
    rot = tmp_path / "hlg_rot.mov"
    cw, chh, n_in = 640, 360, 72
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                            f"{cw}x{chh}", "-r", "60000/1001", "-i", "-", "-vf", "format=yuv420p10le",
                            "-c:v", "libx265", "-preset", "ultrafast", "-x265-params",
                            "log-level=error:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
                            "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc",
                            "-tag:v", "hvc1", str(raw)], stdin=subprocess.PIPE)
    for i in range(n_in):
        fr = np.full((chh, cw, 3), 90, np.uint8)
        fr[chh - 80:, :120] = (0, 0, 230)  # red, coded bottom-left
        cv2.circle(fr, (320 + i, 180), 20, (240, 240, 240), -1)
        enc.stdin.write(fr.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-display_rotation", "-90", "-i", str(raw), "-c", "copy",
                    str(rot)], check=True)
    r = N.probe_route(rot)
    assert r["hdr"] and r["rotation"] == 270 and r["display"] == (chh, cw)
    take = N.normalize(rot, 0.0, 1.0, tmp_path / "work")
    assert take["route"] == "hdr-vt" and take["native"] is None
    v = stream(probe(take["proxy"]), "video")
    assert v["color_transfer"] == "bt709" and v["pix_fmt"] == "yuv420p" and v["r_frame_rate"] == "30/1"
    assert count_frames(take["proxy"]) == 30
    fr = next(read_frames(take["proxy"]))
    tl = fr[:300, :300].astype(int)   # top-left of the upright output
    br = fr[-300:, -300:].astype(int)
    assert tl[..., 2].mean() > tl[..., 1].mean() + 60, tl.mean(axis=(0, 1))   # red is up top-left
    assert abs(br[..., 2].mean() - br[..., 1].mean()) < 25                     # and not bottom-right


def test_native_route_keeps_a_frame_aligned_native_file(tmp_path):
    """A 1440x2560 SDR take (33% more pixels across than the output): proxy and native come from one graph,
    so proxy frame k is the area-downscale of native frame k, not of a neighbour."""
    src = tmp_path / "big.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1440x2560:r=30:d=2",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "14", "-pix_fmt", "yuv420p", str(src)],
                   check=True)
    take = N.normalize(src, 0.5, 1.0, tmp_path / "work")
    assert take["route"] == "sdr+native" and take["native_size"] == (1440, 2560)
    assert count_frames(take["native"]) == count_frames(take["proxy"]) == 30
    prox = list(read_frames(take["proxy"]))
    nat = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
           for f in read_frames(take["native"], take["native_size"], scale=False)]
    for k in (3, 17):
        d = [np.abs(prox[k].astype(int) - nat[j].astype(int)).mean() for j in (k - 1, k, k + 1)]
        assert d[1] < min(d[0], d[2]) and d[1] < 3, (k, d)
    small = tmp_path / "small.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1080x1920:r=30:d=1",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(small)], check=True)
    assert N.normalize(small, 0.0, 0.5, tmp_path / "work2")["route"] == "sdr"  # the 0.1.0 path, unchanged
