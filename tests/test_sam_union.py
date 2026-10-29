"""--sam-prompt union (keyed path, tested offline): each prompt's cached output is decoded on its own and the masks
are OR-ed per frame, so a held prop stays in the cut-out; an empty prompt is left out; the quote counts only the
prompts not yet cached; nothing is called without a key."""
import numpy as np
import pytest

from realone import masks as M
from realone.common import H, W, Refuse
from test_sam_decode import blob, record


def test_union_keeps_the_person_and_the_prop(tmp_path):
    person, mic = blob(60, 40, 1), blob(20, 14, 3)
    tag = "clip_3f"
    (tmp_path / f"{tag}__the-person.txt").write_text(
        "".join(f"<{f}f>{record(0, 100 + f, 200, person)}\n" for f in range(3)))
    (tmp_path / f"{tag}__the-microphone.txt").write_text(
        "".join(f"<{f}f>{record(0, 600, 900 + f, mic)}\n" for f in range(3)))
    (tmp_path / f"{tag}__a-hat.txt").write_text("")  # SAM matched nothing
    path, engine, billed = M.sam_masks_union(None, None, tag, 3, ["the person", "the microphone", "a hat"],
                                             mdir=tmp_path)
    assert billed == 0.0 and "union" in engine and "a hat" not in engine
    u = np.load(path)
    for f in range(3):
        m = np.unpackbits(u[f])[:W * H].reshape(H, W) > 0
        assert (m[200:260, 100 + f:140 + f] == (person > 0)).all()
        assert (m[900 + f:920 + f, 600:614] == (mic > 0)).all()
        assert m.sum() == person.sum() + mic.sum()
    assert M.sam_uncached(tag, ["the person", "the guitar"], mdir=tmp_path) == ["the guitar"]
    with pytest.raises(Refuse, match="no MODEL_API_KEY"):
        M.sam_masks_union(None, None, tag, 3, ["the guitar"], mdir=tmp_path)
