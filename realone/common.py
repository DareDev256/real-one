"""Shared constants and the refusal type every stage raises instead of guessing."""
import hashlib
import json
import subprocess
from pathlib import Path

W, H, FPS = 1080, 1920, 30
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "out"
LOCAL_PREFIX = "TEST_LOCAL_"


class Refuse(Exception):
    """A stage cannot produce a trustworthy result. The CLI prints the message and exits non-zero."""
    exit_code = 2


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _np(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=_np))
    return path


def unpack_mask(packed_row):
    import numpy as np
    return np.unpackbits(packed_row)[:W * H].reshape(H, W) > 0
