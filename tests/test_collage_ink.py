"""Collage legibility: when the world under the paper border is light, the border becomes a dark outline, so the
cut-out never melts into a white world; on a dark world it stays white paper. The decision reads the band the
border actually covers, not the frame."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from realone import composite as C
from realone.common import H, W


def args(**kw):
    d = dict(erode=2, feather=1.4, wrap=0.6, match=0.0, grain=0.0, despill=True)
    d.update(kw)
    return SimpleNamespace(**d)


@pytest.fixture(scope="module")
def figure():
    src = np.full((H, W, 3), (70, 56, 46), np.uint8)
    mask = np.zeros((H, W), np.uint8)
    cv2.rectangle(mask, (330, 700), (750, 1500), 1, -1)
    cv2.circle(mask, (540, 520), 150, 1, -1)
    src[mask > 0] = (40, 40, 160)
    return src, mask > 0


def ring(mask, comp):
    d = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    return (d >= 2) & (d <= 0.5 * comp.collage.border - 0.5)  # always border, whatever the boil


def collage(figure, world, ink):
    src, mask = figure
    comp = C.Compositor(args())
    small = cv2.resize(world, (270, 480)) / 255
    outs, info = comp.frame(src, mask, (0, 0, W, H), world, np.zeros_like(world), C.lab_stats(small), 0, 0,
                            ["collage"], ink=ink)
    return outs["collage"], comp


def lum(img):
    return (img[..., 0] * 0.0722 + img[..., 1] * 0.7152 + img[..., 2] * 0.2126) / 255


def test_light_world_switches_the_border_to_a_dark_outline(figure):
    src, mask = figure
    white = np.full((H, W, 3), 238.0, np.float32)
    ink, share = C.border_ink(white, mask)
    assert ink == "dark" and share > 0.9
    out, comp = collage(figure, white, C.INK if ink == "dark" else None)
    r = ring(mask, comp)
    assert lum(out[r].astype(np.float32)).mean() < 0.15                  # the outline is dark
    assert lum(white).mean() - lum(out[r].astype(np.float32)).mean() > 0.7  # and stands off the world


def test_dark_world_keeps_the_white_paper(figure):
    src, mask = figure
    dark = np.full((H, W, 3), 30.0, np.float32)
    ink, share = C.border_ink(dark, mask)
    assert ink == "white" and share == 0.0
    out, comp = collage(figure, dark, None)
    assert lum(out[ring(mask, comp)].astype(np.float32)).mean() > 0.85


def test_the_decision_reads_the_border_band_not_the_frame(figure):
    """A world that is white only where the border sits (the acid stage's rings) must switch; one that is white
    only far from the artist must not."""
    src, mask = figure
    d = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    near = np.where((d > 0) & (d < 40), 240.0, 25.0).astype(np.float32)
    far = np.where(d > 200, 240.0, 25.0).astype(np.float32)
    assert C.border_ink(np.repeat(near[..., None], 3, 2), mask)[0] == "dark"
    assert C.border_ink(np.repeat(far[..., None], 3, 2), mask)[0] == "white"
