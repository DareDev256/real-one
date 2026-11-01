"""Stage clean plate: the real floor with the artist taken out, rebuilt from neighbouring frames, used to take
real floor pixels OUT of the artist's matte at his feet.

Because the floor is tracked (realone/stage/camera.py), any other frame's floor can be warped onto this frame's
floor exactly. Six neighbours, each with the artist masked out, give a per-pixel median: the floor as it would look
with nobody standing there. A matte pixel near the matte's edge that matches that clean floor is floor, not artist:
a chip of brick under a heel, the floor seen between motion-blurred feet. Those pixels leave the matte. Nothing is
ever added to the matte, and nothing in the artist's interior is touched (a grey shoe on grey stone stays)."""
import collections

import cv2
import numpy as np

from realone.common import H, W

OFFSETS = (-24, -16, -8, 8, 16, 24)
MIN_SAMPLES = 3
EDGE_PX = 10          # only matte pixels within this distance of the matte's edge may be removed
DIFF_FLOOR = 18.0     # 0..255 colour distance under which a matte pixel is the clean floor
GUARD_PX = 25         # neighbours' artist masks are dilated by this before they vote


def foot_roi(mask, contact_y, pad=150):
    """(x0, y0, x1, y1): the lower third of the figure's height around his feet, padded, inside the frame."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    hgt = ys.max() - ys.min()
    y0 = int(max(0, min(contact_y, ys.max()) - 0.33 * hgt))
    sel = ys >= y0
    x0, x1 = int(max(0, xs[sel].min() - pad)), int(min(W, xs[sel].max() + pad))
    return x0, y0, x1, H


class CleanPlate:
    """Streaming: feed frames in order (frame index, BGR, artist mask); ask for frame i once i + max(OFFSETS) has
    been fed (or the take ended). Holds about 50 frames."""

    def __init__(self, H_to0):
        self.Hs = H_to0
        self.buf = collections.OrderedDict()

    def feed(self, i, frame, mask):
        self.buf[i] = (frame, mask)
        while len(self.buf) > 2 * max(OFFSETS) + 4:
            self.buf.popitem(last=False)

    def floor_at(self, i, roi):
        """-> (clean BGR float32 over roi, valid bool over roi) or None."""
        x0, y0, x1, y1 = roi
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
        stack, ok = [], []
        Hi_inv = np.linalg.inv(self.Hs[i])
        for o in OFFSETS:
            j = i + o
            if j not in self.buf:
                continue
            fr, m = self.buf[j]
            G = T @ Hi_inv @ self.Hs[j]  # frame j floor -> frame i floor, then into the roi
            size = (x1 - x0, y1 - y0)
            wf = cv2.warpPerspective(fr, G, size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)).astype(np.float32)
            guard = cv2.dilate(m.astype(np.uint8), np.ones((2 * GUARD_PX + 1,) * 2, np.uint8))
            inside = cv2.warpPerspective(np.ones((H, W), np.uint8), G, size, flags=cv2.INTER_NEAREST) > 0
            wm = cv2.warpPerspective(guard, G, size, flags=cv2.INTER_NEAREST) > 0
            stack.append(wf)
            ok.append(inside & ~wm)
        if len(stack) < MIN_SAMPLES:
            return None
        S = np.stack(stack)
        V = np.stack(ok)
        S[~V] = np.nan
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                clean = np.nanmedian(S, axis=0)
        valid = V.sum(0) >= MIN_SAMPLES
        return np.nan_to_num(clean), valid

    def refine(self, i, frame, mask, contact_y):
        """-> (mask with real-floor pixels removed at its edge, pixels removed)."""
        if not mask.any():
            return mask, 0
        roi = foot_roi(mask, contact_y)
        if roi is None:
            return mask, 0
        got = self.floor_at(i, roi)
        if got is None:
            return mask, 0
        clean, valid = got
        x0, y0, x1, y1 = roi
        m = mask[y0:y1, x0:x1]
        a = cv2.GaussianBlur(frame[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.0)
        b = cv2.GaussianBlur(clean, (0, 0), 1.0)
        diff = np.abs(a - b).max(axis=2)
        dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        edge = dist <= EDGE_PX
        drop = m & edge & valid & (diff < DIFF_FLOOR)
        # only whole little regions of floor go; a speckle of matching pixels inside a shoe stays
        drop = cv2.morphologyEx(drop.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        # and only regions that TOUCH the matte's edge: floor seen through the matte enters from outside. A
        # floor-coloured patch inside the artist (a brick-orange stripe on a shoe) is his, and stays (gauntlet r3:
        # the 28 px band without this ate the stripe on 112 of 150 frames)
        k, lab = cv2.connectedComponents(drop.astype(np.uint8), connectivity=8)
        if k > 1:
            touching = np.unique(lab[drop & (dist <= 1.5)])
            drop = np.isin(lab, touching[touching > 0])
        out = mask.copy()
        out[y0:y1, x0:x1] = m & ~drop
        return out, int(drop.sum())


def refine_masks(proxy, n, masks, H_to0, contact, out_path, log=print):
    """One pass over the take: every frame's matte with the real floor taken out at its feet, written as a packed
    mask file in the same format as the input (np.packbits rows). -> (path, removed px per frame)."""
    from pathlib import Path

    from realone.media import read_frames
    out_path = Path(out_path)
    tmp = out_path.with_suffix(".tmp.npy")
    packed = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, W * H // 8))
    cp = CleanPlate(H_to0)
    lag = max(OFFSETS)
    pending = collections.deque()
    removed = np.zeros(n, np.int64)

    def unpack(i):
        return np.unpackbits(masks[i])[: W * H].reshape(H, W).astype(bool)

    def flush(upto):
        while pending and pending[0][0] <= upto:
            j, fr, m = pending.popleft()
            cy = contact[j][1] if np.isfinite(contact[j][1]) else H - 1
            r, k = cp.refine(j, fr, m, cy)
            packed[j] = np.packbits(r)
            removed[j] = k

    for i, fr in enumerate(read_frames(proxy)):
        if i >= n:
            break
        m = unpack(i)
        cp.feed(i, fr, m)
        pending.append((i, fr, m))
        flush(i - lag)
    flush(n)
    packed.flush()
    del packed
    tmp.rename(out_path)
    log(f"stage: clean-plate feet: {int((removed > 0).sum())}/{n} frames had real floor in the matte, "
        f"{int(removed.sum())} px removed (median {int(np.median(removed[removed > 0])) if removed.any() else 0}"
        f" px on those frames)")
    return out_path, removed
