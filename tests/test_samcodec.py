"""The compiled SAM mask codec (realone/samcodec.py) against Meta's reference (meta_sam_parser._mask_codec):
the same payload decodes to the same raster, the same raster encodes to the same payload, the same errors."""
import numpy as np
import pytest

mc = pytest.importorskip("meta_sam_parser._mask_codec")
SC = pytest.importorskip("realone.samcodec")


def _blob(w, h, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.zeros((h, w), np.uint8)
    for _ in range(4):  # a few overlapping ellipses with noisy edges: a mask-like raster, not white noise
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        a, b = rng.uniform(2, max(3, w / 2)), rng.uniform(2, max(3, h / 2))
        r |= (((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 < 1 + rng.normal(0, 0.05, (h, w))).astype(np.uint8)
    return r.ravel()


@pytest.mark.parametrize("w,h,seed", [(37, 53, 1), (120, 200, 2), (8, 8, 3), (201, 64, 4)])
def test_bit_exact_with_the_reference(w, h, seed):
    bits = _blob(w, h, seed)
    ref_payload = mc._encode_raster(bits.tolist(), w, h)
    assert SC.fast_encode_raster(bits, w, h) == ref_payload
    assert SC.fast_decode_raster(ref_payload, w, h) == mc._decode_raster(ref_payload, w, h) == bits.tobytes()


def test_same_errors_as_the_reference():
    bits = _blob(40, 40, 5)
    payload = mc._encode_raster(bits.tolist(), 40, 40)
    short = "!" + mc._pack(mc._unpack(payload[1:])[:3])
    for bad in (short,):
        with pytest.raises(type(mc._invalid("x"))):
            mc._decode_raster(bad, 40, 40)
        with pytest.raises(type(mc._invalid("x"))):
            SC.fast_decode_raster(bad, 40, 40)
    with pytest.raises(type(mc._invalid("x"))):
        SC.fast_encode_raster(np.full(16, 2, np.uint8), 4, 4)


def test_fast_codec_is_installed_only_inside_the_block():
    before = mc._decode_raster
    with SC.fast_codec():
        assert mc._decode_raster is SC.fast_decode_raster
    assert mc._decode_raster is before


def _outcome(fn, payload, w, h):
    try:
        return ("ok", fn(payload, w, h))
    except Exception as e:  # the reference raises InvalidSegmentationMaskError; so must the port
        return ("err", type(e).__name__)


def test_corrupt_and_truncated_payloads_behave_like_the_reference():
    rng = np.random.default_rng(9)
    good = mc._encode_raster(_blob(24, 24, 6).tolist(), 24, 24)
    raw = mc._unpack(good[1:])
    cases = ["!" + mc._pack(raw[:k]) for k in (5, 6, 8, len(raw) // 2)]            # cut short: runs out
    cases += ["!" + mc._pack(bytes(rng.integers(0, 256, rng.integers(4, 40)).astype(np.uint8)))
              for _ in range(150)]                                                   # random bytes: clamps, garbage
    for p in cases:
        assert _outcome(SC.fast_decode_raster, p, 24, 24) == _outcome(mc._decode_raster, p, 24, 24), p
