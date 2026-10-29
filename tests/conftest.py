"""Shared fixtures. Everything is generated into pytest's tmp dirs: no footage, no out/, no network, no key."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "spike"))

import make_test_song as mts  # noqa: E402


@pytest.fixture(scope="session")
def song(tmp_path_factory):
    """The procedural test song (98 BPM, intro 4 / verse 8 / hook 8 / verse 8 / hook 8) and its ground truth."""
    import soundfile as sf
    y, truth = mts.make_song()
    path = tmp_path_factory.mktemp("song") / "test_song.wav"
    sf.write(str(path), np.stack([y, y], 1), mts.SR, subtype="PCM_16")
    return path, y, truth


@pytest.fixture(scope="session")
def timeline(song):
    from realone.audio import analyze
    return analyze(song[0])


@pytest.fixture(scope="session")
def master22(song):
    from realone.media import load_audio
    from realone.sync import SR
    return load_audio(song[0], SR)


@pytest.fixture(scope="session")
def live_vocal():
    return mts.make_live_vocal()


def to22(x):
    import librosa
    return librosa.resample(np.asarray(x, np.float32), orig_sr=mts.SR, target_sr=22050)
