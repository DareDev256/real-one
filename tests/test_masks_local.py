"""The local (TEST ONLY) Apple Vision masks: written as a memmap by several processes, identical to one process;
the drawn test figure takes the foreground-instance fallback (person segmentation returns empty on it)."""
import subprocess
import sys

import numpy as np
import pytest

import make_test_clip as mtc
from realone.common import FPS, H, W

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision is macOS only")


@pytest.fixture(scope="module")
def figure_clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("vision") / "fig.mp4"
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
                            "-pix_fmt", "yuv420p", str(p)], stdin=subprocess.PIPE)
    for i in range(16):
        t = i / FPS
        f = mtc.figure(t)
        a = f[..., 3:4].astype(np.float32) / 255
        enc.stdin.write(np.clip(f[..., :3] * a + mtc.background(t) * (1 - a), 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    assert enc.wait() == 0
    return p


def test_parallel_vision_masks_equal_one_process(figure_clip, tmp_path):
    from realone.masks import vision_masks
    p1, name1 = vision_masks(figure_clip, 16, tmp_path / "local_one.npy", workers=1)
    p2, name2 = vision_masks(figure_clip, 16, tmp_path / "local_two.npy", workers=2)
    assert name1 == name2 == "VNGenerateForegroundInstanceMaskRequest"
    a, b = np.load(p1), np.load(p2)
    assert a.shape == (16, W * H // 8) and np.array_equal(a, b)
    areas = [np.unpackbits(r).mean() for r in a]
    assert min(areas) > 0.05  # the figure was found on every frame
