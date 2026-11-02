"""Looks and despill on a synthetic frame: a red-and-skin figure on a bluish source background, with an enclosed
pocket of that background inside the mask (the chin / hand / collar case), on a colourful world."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from realone import composite as C
from realone.common import FPS, H, W

SRC_BG = (0.27, 0.22, 0.18)  # BGR: the bluish room of the test clip


def args(**kw):
    d = dict(erode=2, feather=1.4, wrap=0.6, match=0.35, grain=5.0, despill=True)
    d.update(kw)
    return SimpleNamespace(**d)


@pytest.fixture(scope="module")
def scene():
    src = np.zeros((H, W, 3), np.float32)
    src[:] = SRC_BG
    mask = np.zeros((H, W), np.uint8)
    cv2.rectangle(mask, (330, 700), (750, 1500), 1, -1)          # torso
    cv2.circle(mask, (540, 520), 150, 1, -1)                     # head
    src[mask > 0] = (0.16, 0.16, 0.62)                           # red jacket
    head = np.zeros_like(mask)
    cv2.circle(head, (540, 520), 150, 1, -1)
    src[head > 0] = (0.47, 0.59, 0.80)                           # skin
    pocket = np.zeros_like(mask)
    cv2.circle(pocket, (640, 760), 22, 1, -1)                    # enclosed gap: source bg inside the mask
    src[pocket > 0] = SRC_BG
    shirt = np.zeros_like(mask)
    cv2.rectangle(shirt, (380, 1100), (700, 1450), 1, -1)        # big bg-coloured panel: clothing, not a gap
    src[shirt > 0] = SRC_BG
    src = cv2.GaussianBlur(src, (0, 0), 1.0)                     # soft edges, so the fringe is a real mix
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    world = np.stack([120 + 100 * np.sin(xx / 90), 60 + 50 * np.cos(yy / 130), 200 - 60 * np.sin((xx + yy) / 200)], -1)
    return {"src": (src * 255).astype(np.uint8), "mask": mask > 0, "pocket": pocket > 0, "shirt": shirt > 0,
            "world": world.astype(np.float32), "front": np.zeros((H, W, 3), np.float32)}


def sat(img):
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    return hsv[..., 1].astype(np.float32) / 255


def render(scene, look, gframe=0, **kw):
    comp = C.Compositor(args(**kw))
    small = cv2.resize(scene["world"], (270, 480)).astype(np.float32) / 255
    outs, info = comp.frame(scene["src"], scene["mask"], (0, 0, W, H), scene["world"], scene["front"],
                            C.lab_stats(small), gframe, gframe, [look])
    return outs[look], info[look], comp


def test_receipt_world_grey_artist_in_colour(scene):
    out, _, _ = render(scene, "receipt")
    m = scene["mask"].astype(np.uint8)
    world_px = cv2.dilate(m, np.ones((61, 61), np.uint8)) == 0
    artist_px = (cv2.erode(m, np.ones((41, 41), np.uint8)) > 0) & ~cv2.dilate(
        (scene["pocket"] | scene["shirt"]).astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    assert sat(out)[world_px].mean() < 0.02, sat(out)[world_px].mean()
    assert sat(scene["world"])[world_px].mean() > 0.3  # the world really was colourful before
    ratio = sat(out)[artist_px].mean() / sat(scene["src"].astype(np.float32))[artist_px].mean()
    assert ratio > 0.92, ratio


def test_collage_border_ring_is_near_white(scene):
    out, _, comp = render(scene, "collage", grain=0.0)
    m = scene["mask"].astype(np.uint8)
    d = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    ring = (d >= 2) & (d <= 0.5 * comp.collage.border - 0.5)  # always paper: the radius never drops below half
    assert ring.sum() > 2000
    px = out[ring].astype(np.float32) / 255
    assert px.mean() > 0.88, px.mean()
    assert sat(out)[ring].mean() < 0.08


def test_collage_boil_changes_at_12fps_not_30(scene):
    """Same artist pixels every frame: the border may only change when the 12 fps boil step changes."""
    m = scene["mask"]
    d = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
    ring = (d > 0) & (d < 2.5 * C.Collage().border)
    sigs = []
    comp = C.Compositor(args(grain=0.0, match=0.0))
    small = cv2.resize(scene["world"], (270, 480)).astype(np.float32) / 255
    for g in range(FPS):  # one second of 30 fps frames
        outs, _ = comp.frame(scene["src"], m, (0, 0, W, H), scene["world"], scene["front"], C.lab_stats(small),
                             g, g, ["collage"])
        sigs.append(outs["collage"][ring].tobytes())
    distinct = len(set(sigs))
    assert distinct == C.BOIL_FPS, distinct
    steps = [C.boil_step(g) for g in range(FPS)]
    for g in range(1, FPS):
        same_step = steps[g] == steps[g - 1]
        assert (sigs[g] == sigs[g - 1]) == same_step, g


def test_despill_removes_an_enclosed_pocket_and_leaves_clothing(scene):
    fg = scene["src"].astype(np.float32) / 255
    target = np.zeros_like(fg)
    target[:] = (0.1, 0.8, 0.1)
    out, cut, st = C.despill(fg, scene["mask"], target)
    inner_pocket = cv2.erode(scene["pocket"].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert cut[inner_pocket].mean() > 0.8, cut[inner_pocket].mean()
    inner_shirt = cv2.erode(scene["shirt"].astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    assert cut[inner_shirt].max() < 0.05  # 320x350 px of bg-coloured clothing is not a gap
    m = scene["mask"].astype(np.uint8)
    deep = cv2.distanceTransform(m, cv2.DIST_L2, 3) > 121  # beyond the 120 px reach
    assert np.abs(out - fg)[deep].max() < 1e-6  # the artist's interior is the phone's pixels, untouched
    edge = (cv2.distanceTransform(m, cv2.DIST_L2, 3) > 0.5) & (cv2.distanceTransform(m, cv2.DIST_L2, 3) < 2.5)
    blue_before = (fg[..., 0] - fg[..., 2])[edge & ~scene["shirt"]].mean()
    blue_after = (out[..., 0] - out[..., 2])[edge & ~scene["shirt"]].mean()
    assert blue_after < blue_before - 0.01  # the bluish fringe moved toward the target


def test_despill_scales_with_the_close_up_reframe(scene):
    """Through the compositor at 1.9x: the pocket is 1.9x further from the edge in output pixels and must still go
    (it once survived every medium and close shot because the reach was fixed in output pixels)."""
    z = 1.9
    cw, ch = W / z, H / z
    px, py = 640, 760  # pocket centre in source pixels
    crop = (px - cw / 2, py - ch / 2, cw, ch)
    comp = C.Compositor(args(grain=0.0))
    small = cv2.resize(scene["world"], (270, 480)).astype(np.float32) / 255
    outs, _ = comp.frame(scene["src"], scene["mask"], crop, scene["world"], scene["front"], C.lab_stats(small),
                         0, 0, ["receipt"])
    out = outs["receipt"]
    centre = out[H // 2 - 20:H // 2 + 20, W // 2 - 20:W // 2 + 20]
    assert sat(centre).mean() < 0.08, sat(centre).mean()  # grey world shows through, not the bluish room


def test_pockets_are_cut_only_when_asked_and_never_along_the_outline():
    """A bright patch inside the matte that matches the background: cut as a pocket only with pockets=True (the
    gap-filling test masks); a patch lying along the matte's outline (a head's rim highlight) is never a pocket."""
    import cv2
    import numpy as np
    from realone.common import H, W
    from realone.composite import despill_estimate
    fg = np.zeros((H, W, 3), np.float32)
    fg[:] = (0.9, 0.9, 0.9)                         # a white background
    hard = np.zeros((H, W), bool)
    hard[500:1500, 300:800] = True
    fg[hard] = (0.25, 0.18, 0.12)                   # the artist
    gap = np.zeros((H, W), bool)
    gap[900:960, 340:400] = True                    # an enclosed gap the mask filled, 40 px in from the edge
    fg[gap] = (0.9, 0.9, 0.9)
    rim = np.zeros((H, W), bool)
    rim[500:540, 450:650] = True                    # a highlight along the top outline, background-coloured
    fg[rim] = (0.9, 0.9, 0.9)
    on = despill_estimate(fg, hard, pockets=True)
    off = despill_estimate(fg, hard, pockets=False)
    assert on["cut"][gap].mean() > 0.5               # the enclosed gap is cut with pockets on
    assert off["cut"].max() == 0.0                   # nothing is cut with pockets off (SAM)
    assert on["cut"][rim].mean() < 0.1               # the rim is never treated as a pocket
