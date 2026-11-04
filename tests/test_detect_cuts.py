"""The render's cut detector (realone/encode.detect_cuts), Nov 4: frames held dark on purpose (the entrance) must not
set the typical frame-to-frame change, or ordinary motion after the lights come up reads as a cut."""
import cv2
import numpy as np

from realone import encode as EC

DARK, RAMP, WALK = 180, 20, 40


def _frames(monkeypatch, frames):
    monkeypatch.setattr(EC, "read_frames", lambda path, size=None: iter(frames))


def _clip():
    rng = np.random.default_rng(1)
    scene = cv2.GaussianBlur(rng.integers(0, 255, (240, 400, 3)).astype(np.uint8), (0, 0), 6)
    scene = cv2.normalize(scene, None, 30, 220, cv2.NORM_MINMAX)
    other = np.full_like(scene, 245)                              # a different shot entirely

    def view(img, x):
        return np.ascontiguousarray(img[:, x:x + 135])
    frames = [(view(scene, 0) * 0.06).astype(np.uint8)] * DARK     # the world waiting in the dark
    for k in range(RAMP):                                          # the lights come up
        frames.append((view(scene, 0) * (0.06 + 0.94 * (k + 1) / RAMP)).astype(np.uint8))
    for k in range(WALK):                                          # then ordinary motion, a little every frame
        frames.append(view(scene, 3 * k))
    frames += [view(other, 0)] * 5                                 # a real cut
    return frames


def test_dark_frames_do_not_set_the_baseline(monkeypatch):
    frames = _clip()
    quiet = range(0, DARK + RAMP)
    _frames(monkeypatch, frames)
    loose, _ = EC.detect_cuts("x.mp4")
    assert any(c >= DARK + RAMP + 1 and c < DARK + RAMP + WALK for c in loose)  # the walk read as cuts
    _frames(monkeypatch, frames)
    cuts, _ = EC.detect_cuts("x.mp4", quiet=quiet)
    assert [c for c in cuts if c >= DARK + RAMP] == [DARK + RAMP + WALK]        # only the real cut
