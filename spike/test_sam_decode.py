#!/usr/bin/env python3
"""Offline test of the SAM 3.1 output path, no API call.

Builds a fake `output_text` in the documented wire grammar (docs/sam/reading-segmentation) using the parser
library's own one_bit ENCODER, then runs it through freeze_test's parse -> decode -> place code and checks the
pixels land where the box says. Also proves it can fail: a shifted box must be caught.

    .venv/bin/python spike/test_sam_decode.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze_test as ft  # noqa: E402
from meta_sam_parser import _mask_codec as mc  # noqa: E402

W, H = ft.W, ft.H


def record(obj, x1, y1, mask):
    h, w = mask.shape
    payload = mc._encode_raster(mask.reshape(-1).tolist(), w, h)
    return (f"{obj}<|box;x1={x1};y1={y1};x2={x1 + w - 1};y2={y1 + h - 1};w={W};h={H}|>"
            f"<|mask;x=0;y=0;data={h},{w},{payload}|>")


def blob(h, w, seed):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    m = (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) < 1
    m[r.integers(0, h, 20), r.integers(0, w, 20)] ^= True  # a few noisy pixels
    return m.astype(np.uint8)


def main():
    a, b = blob(60, 40, 1), blob(30, 50, 2)
    text = (f"<0f>{record(0, 100, 200, a)},{record(1, 600, 900, b)}"
            f"\n<2f>{record(0, 110, 210, a)}\n")  # frame 1 deliberately absent
    recs = ft.parse_sam_text(text)
    assert len(recs) == 3, recs
    packed, stats = ft.sam_masks(text, 3)
    assert stats["missing_frames"] == 1
    f0 = np.unpackbits(packed[0])[:W * H].reshape(H, W) > 0
    assert (f0[200:260, 100:140] == (a > 0)).all(), "object 0 not at its box"
    assert (f0[900:930, 600:650] == (b > 0)).all(), "object 1 not at its box"
    assert f0.sum() == a.sum() + b.sum(), "stray pixels outside the boxes"
    f1 = np.unpackbits(packed[1])[:W * H].reshape(H, W) > 0
    assert f1.sum() > 0, "missing frame 1 should hold a neighbouring mask"
    f2 = np.unpackbits(packed[2])[:W * H].reshape(H, W) > 0
    assert (f2[210:270, 110:150] == (a > 0)).all()
    # the check can fail: a mask placed 5 px off must not equal the expected region
    shifted = np.zeros((H, W), bool)
    ft.place_mask(shifted, a > 0, 105, 200, 145, 260)
    assert not (shifted[200:260, 100:140] == (a > 0)).all()
    # same stream with the newlines stripped (if the API's deltas carry none) must parse identically
    assert len(ft.parse_sam_text(text.replace("\n", ""))) == 3
    print("test_sam_decode: OK (parse, one_bit decode, box-local placement, missing-frame hold, can fail)")


if __name__ == "__main__":
    main()
