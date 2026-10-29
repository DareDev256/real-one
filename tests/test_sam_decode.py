"""Offline test of the SAM 3.1 output path, no API call (ported from spike/test_sam_decode.py to the package).

Builds a fake `output_text` in the documented wire grammar using the parser library's own one_bit ENCODER, then
runs it through realone.masks parse -> decode -> place, and checks the pixels land where the box says. It can
fail: a mask placed 5 px off must not match."""
import numpy as np
from meta_sam_parser import _mask_codec as mc

from realone import masks as M
from realone.common import H, W


def record(obj, x1, y1, mask):
    h, w = mask.shape
    payload = mc._encode_raster(mask.reshape(-1).tolist(), w, h)
    return (f"{obj}<|box;x1={x1};y1={y1};x2={x1 + w - 1};y2={y1 + h - 1};w={W};h={H}|>"
            f"<|mask;x=0;y=0;data={h},{w},{payload}|>")


def blob(h, w, seed):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    m = (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) < 1
    m[r.integers(0, h, 20), r.integers(0, w, 20)] ^= True
    return m.astype(np.uint8)


def test_parse_decode_place_and_hold():
    a, b = blob(60, 40, 1), blob(30, 50, 2)
    text = (f"<0f>{record(0, 100, 200, a)},{record(1, 600, 900, b)}"
            f"\n<2f>{record(0, 110, 210, a)}\n")  # frame 1 deliberately absent
    assert len(M.parse_sam_text(text)) == 3
    packed, stats = M.sam_masks(text, 3)
    assert stats["missing_frames"] == 1
    f0 = np.unpackbits(packed[0])[:W * H].reshape(H, W) > 0
    assert (f0[200:260, 100:140] == (a > 0)).all()
    assert (f0[900:930, 600:650] == (b > 0)).all()
    assert f0.sum() == a.sum() + b.sum()
    assert (np.unpackbits(packed[1])[:W * H]).sum() > 0  # held from a neighbour
    shifted = np.zeros((H, W), bool)
    M.place_mask(shifted, a > 0, 105, 200, 145, 260)
    assert not (shifted[200:260, 100:140] == (a > 0)).all()
    assert len(M.parse_sam_text(text.replace("\n", ""))) == 3  # UNCONFIRMED newline handling, both ways
