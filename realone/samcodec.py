"""SAM 3.1 one-bit mask codec, compiled: a numba port of the range coder in Meta's meta-sam-parser
(`meta_sam_parser/_mask_codec.py`, SAM License, which permits modifications and derivative works).

Why: the parser validates every mask by decoding it and re-encoding it in pure Python (~456k Python calls per mask,
~1 s each), and the pipeline then decodes it again. On a 150-frame take that was 134 s of parsing plus 417 ms per
mask of decoding; a 3-minute song would take hours. The compiled versions here are the same algorithm, bit for bit
(tests/test_samcodec.py checks them against the reference on real SAM 3.1 output), installed into the parser for
the duration of a parse with `fast_codec()`. The payload text packing (`_pack` / `_unpack`) stays the library's."""
import contextlib

import numpy as np
from numba import njit

_TOP = 1 << 24
_BOTTOM = 1 << 16
_INCREMENT = 14
_COUNT_LIMIT = 4096
_CONTEXT_COUNT = 1 << 12
_PADDING = 2
_M32 = 0xFFFFFFFF
_M16 = 0xFFFF


@njit(cache=True)
def _context(p, pos, prior, earlier):
    return (p[pos - 1] | (p[pos - 2] << 1) | (p[prior - 2] << 2) | (p[prior - 1] << 3) | (p[prior] << 4)
            | (p[prior + 1] << 5) | (p[prior + 2] << 6) | (p[earlier - 2] << 7) | (p[earlier - 1] << 8)
            | (p[earlier] << 9) | (p[earlier + 1] << 10) | (p[earlier + 2] << 11))


@njit(cache=True)
def _update(counts, off, bit):
    counts[off + bit] = (counts[off + bit] + _INCREMENT) & _M16
    if counts[off] + counts[off + 1] >= _COUNT_LIMIT:
        a = counts[off] >> 1
        b = counts[off + 1] >> 1
        counts[off] = a if a else 1
        counts[off + 1] = b if b else 1


@njit(cache=True)
def decode_raster(data, width, height):
    """-> (uint8 raster width*height, error code). 0 ok, 1 payload too short, 2 ran out, 3 bad coding state."""
    out = np.zeros(width * height, np.uint8)
    n = data.shape[0]
    if n < 4:
        return out, 1
    pos = 0
    code = 0
    for _ in range(4):
        code = ((code << 8) | data[pos]) & _M32
        pos += 1
    low = 0
    rng = _M32
    counts = np.ones(_CONTEXT_COUNT * 2, np.int64)
    pw = width + 2 * _PADDING
    padded = np.zeros(pw * (height + _PADDING), np.int64)
    for y in range(height):
        row = (y + _PADDING) * pw + _PADDING
        for x in range(width):
            p = row + x
            off = _context(padded, p, p - pw, p - 2 * pw) << 1
            zero = counts[off]
            one = counts[off + 1]
            total = zero + one
            scaled = rng // total
            if scaled == 0:
                return out, 3
            value = ((code - low) & _M32) // scaled
            # (the reference clamps value to total - 1 here; value is only ever compared with zero <= total - 1,
            # so the clamp cannot change a bit, and it is left out)
            if value < zero:
                bit = 0
                cum, freq = 0, zero
            else:
                bit = 1
                cum, freq = zero, one
            low = (low + scaled * cum) & _M32
            rng = scaled * freq
            while True:
                if ((low ^ ((low + rng) & _M32)) & _M32) < _TOP:
                    pass
                elif rng < _BOTTOM:
                    rng = ((-low) & _M32) & (_BOTTOM - 1)
                else:
                    break
                if pos >= n:
                    return out, 2
                code = ((code << 8) | data[pos]) & _M32
                pos += 1
                low = (low << 8) & _M32
                rng = (rng << 8) & _M32
            padded[p] = bit
            out[y * width + x] = bit
            _update(counts, off, bit)
    return out, 0


@njit(cache=True)
def encode_raster(bits, width, height):
    """-> (encoded bytes as uint8 array, error code). 0 ok, 1 not binary."""
    buf = np.zeros(max(64, width * height // 2 + 64), np.uint8)
    nout = 0
    low = 0
    rng = _M32
    counts = np.ones(_CONTEXT_COUNT * 2, np.int64)
    pw = width + 2 * _PADDING
    padded = np.zeros(pw * (height + _PADDING), np.int64)
    for y in range(height):
        row = (y + _PADDING) * pw + _PADDING
        for x in range(width):
            p = row + x
            off = _context(padded, p, p - pw, p - 2 * pw) << 1
            zero = counts[off]
            one = counts[off + 1]
            bit = bits[y * width + x]
            if bit > 1:
                return buf[:0], 1
            total = zero + one
            scaled = rng // total
            cum = 0 if bit == 0 else zero
            freq = zero if bit == 0 else one
            low = (low + scaled * cum) & _M32
            rng = scaled * freq
            while True:
                if ((low ^ ((low + rng) & _M32)) & _M32) < _TOP:
                    pass
                elif rng < _BOTTOM:
                    rng = ((-low) & _M32) & (_BOTTOM - 1)
                else:
                    break
                if nout >= buf.shape[0]:
                    nb = np.zeros(buf.shape[0] * 2, np.uint8)
                    nb[:nout] = buf[:nout]
                    buf = nb
                buf[nout] = (low >> 24) & 0xFF
                nout += 1
                low = (low << 8) & _M32
                rng = (rng << 8) & _M32
            padded[p] = bit
            _update(counts, off, bit)
    for _ in range(4):
        if nout >= buf.shape[0]:
            nb = np.zeros(buf.shape[0] * 2, np.uint8)
            nb[:nout] = buf[:nout]
            buf = nb
        buf[nout] = (low >> 24) & 0xFF
        nout += 1
        low = (low << 8) & _M32
    return buf[:nout], 0


def _mc():
    from meta_sam_parser import _mask_codec as mc
    return mc


def fast_decode_raster(payload, width, height):
    """Drop-in for meta_sam_parser._mask_codec._decode_raster (same errors, same bytes)."""
    mc = _mc()
    data = np.frombuffer(mc._unpack(payload[1:]), np.uint8)
    out, err = decode_raster(data, int(width), int(height))
    if err == 1:
        raise mc._invalid("Mask payload is missing its finalization.")
    if err == 2:
        raise mc._invalid("Mask payload ended before decoding completed.")
    if err == 3:
        raise mc._invalid("Mask payload has invalid coding state.")
    return out.tobytes()


def fast_encode_raster(input_, width, height):
    """Drop-in for meta_sam_parser._mask_codec._encode_raster."""
    mc = _mc()
    bits = np.frombuffer(bytes(input_), np.uint8) if not isinstance(input_, np.ndarray) else input_.astype(np.uint8)
    enc, err = encode_raster(bits, int(width), int(height))
    if err:
        raise mc._invalid("Decoded mask is not binary.")
    return f"!{mc._pack(enc.tobytes())}"


@contextlib.contextmanager
def fast_codec():
    """Within this block the parser's own validation and decoding use the compiled codec."""
    mc = _mc()
    saved = mc._decode_raster, mc._encode_raster
    mc._decode_raster, mc._encode_raster = fast_decode_raster, fast_encode_raster
    try:
        yield
    finally:
        mc._decode_raster, mc._encode_raster = saved
