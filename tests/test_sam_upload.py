"""SAM upload guards (keyless, fake client): one upload per take however many prompts, an expiry on the upload, and
refusals before any byte or call when the take is over the Files API size cap or SAM's frame cap."""
import types

import pytest

from realone import masks as M
from realone.common import Refuse


class FakeStream:
    def __init__(self, text):
        self.events = [types.SimpleNamespace(type="response.output_text.delta", delta=text)]

    def __enter__(self):
        return iter(self.events)

    def __exit__(self, *a):
        return False


class FakeClient:
    def __init__(self):
        self.uploads, self.calls = [], []
        self.files = types.SimpleNamespace(create=self._create)
        self.responses = types.SimpleNamespace(stream=self._stream)

    def _create(self, purpose, file, expires_after=None):
        self.uploads.append({"purpose": purpose, "expires_after": expires_after, "bytes": len(file.read())})
        return types.SimpleNamespace(id=f"file-{len(self.uploads)}")

    def _stream(self, model, input, metadata):
        self.calls.append(input[0]["content"][1]["file_id"])
        return FakeStream("")


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(M, "_UPLOADED", {})


def take(tmp_path, n=1000):
    p = tmp_path / "norm.mp4"
    p.write_bytes(b"\0" * n)
    return p


def test_a_prompt_union_uploads_the_take_once_with_an_expiry(tmp_path):
    c, norm = FakeClient(), take(tmp_path)
    for prompt in ("the person", "the microphone"):
        M.sam_text_one(c, norm, "t", 300, prompt, mdir=tmp_path / "m")
    assert len(c.uploads) == 1 and c.calls == ["file-1", "file-1"]
    assert c.uploads[0]["expires_after"] == {"anchor": "created_at", "seconds": M.UPLOAD_TTL_S}


def test_a_take_over_the_files_cap_refuses_before_any_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "FILES_MAX_BYTES", 999)
    c = FakeClient()
    with pytest.raises(Refuse, match="1 GiB"):
        M.sam_text_one(c, take(tmp_path, 1000), "t", 300, "the person", mdir=tmp_path / "m")
    assert c.uploads == [] and c.calls == []


def test_a_take_over_the_frame_cap_refuses_before_any_upload(tmp_path):
    c = FakeClient()
    with pytest.raises(Refuse, match="per request"):
        M.sam_text_one(c, take(tmp_path), "t", M.SAM_MAX_FRAMES + 1, "the person", mdir=tmp_path / "m")
    assert c.uploads == [] and c.calls == []
