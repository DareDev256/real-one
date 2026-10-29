"""Beat grid, downbeats and sections against the test song's ground truth."""
import numpy as np
import pytest

import make_test_song as mts
from realone import audio


def test_bpm_within_1_of_truth(timeline, song):
    truth = song[2]
    assert abs(timeline["bpm"] - truth["bpm"]) < 1.0, timeline["bpm"]


def test_downbeats_on_the_true_bar_lines(timeline, song):
    """Phase and timing: every detected downbeat sits within 1 frame (33 ms) of a true bar line, and the first
    one is the song's real first downbeat (not beat 2 or 3)."""
    truth = song[2]
    true_db = truth["first_downbeat_s"] + np.arange(truth["n_bars"] + 1) * truth["bar_s"]
    db = np.array(timeline["downbeats"])
    assert timeline["beats_per_bar"] == 4
    assert abs(db[0] - truth["first_downbeat_s"]) < 1 / 30
    err = np.array([np.abs(true_db - d).min() for d in db])
    assert err.max() < 1 / 30, err.max()


def test_hook_boundaries_within_1_bar(timeline, song):
    truth = song[2]
    bar = truth["bar_s"]
    hooks = [s for s in timeline["sections"] if s["role"] == "hook"]
    assert len(hooks) == len(truth["hook"]) == 2, timeline["sections"]
    for got, want in zip(hooks, truth["hook"]):
        assert abs(got["start_s"] - want["start_s"]) <= bar, (got, want)
        assert abs(got["end_s"] - want["end_s"]) <= bar, (got, want)
    hook = timeline["sections"][timeline["hook_index"]]
    assert hook["is_hook"] and hook["repeats"] == 2


def test_hook_is_the_loudest_repeated_section(timeline):
    hook_db = max(s["energy_db"] for s in timeline["sections"] if s["is_hook"])
    other = [s["energy_db"] for s in timeline["sections"] if not s["is_hook"]]
    assert hook_db > max(other)


def test_onset_lag_correction_is_what_makes_it_line_up(song):
    """The 21 ms lag constant is measured, not decorative: without it the raw tracker sits ~21 ms late."""
    from realone.media import load_audio
    y = load_audio(song[0], audio.SR)
    beats, *_ = audio.beat_grid(y)
    truth = song[2]
    true_beats = truth["first_downbeat_s"] + np.arange(truth["n_bars"] * 4 + 1) * 60 / truth["bpm"]
    err = np.array([b - true_beats[np.abs(true_beats - b).argmin()] for b in beats])
    assert abs(err.mean()) < 0.008, err.mean()
    assert abs(err.mean() + audio.ONSET_LAG_S) > 0.012  # i.e. the correction is doing real work


@pytest.mark.parametrize("bpm,structure", [
    (87.0, [("intro", 2), ("verse", 8), ("hook", 8), ("bridge", 4), ("hook", 8)]),
])
def test_second_structure_is_not_overfit(tmp_path, bpm, structure):
    """A different tempo and layout, so the analysis is not tuned to one song."""
    import soundfile as sf
    y, truth = mts.make_song(bpm=bpm, seed=5, structure=structure, lead_in=0.73)
    p = tmp_path / "v.wav"
    sf.write(str(p), y, mts.SR)
    tl = audio.analyze(p)
    assert abs(tl["bpm"] - bpm) < 1.0
    hooks = [s for s in tl["sections"] if s["role"] == "hook"]
    assert len(hooks) == 2
    for got, want in zip(hooks, truth["hook"]):
        assert abs(got["start_s"] - want["start_s"]) <= truth["bar_s"]
        assert abs(got["end_s"] - want["end_s"]) <= truth["bar_s"]


def test_pick_section_refuses_what_is_not_there(timeline):
    from realone.common import Refuse
    assert audio.pick_section(timeline, "hook:2")["start_bar"] > audio.pick_section(timeline, "hook")["start_bar"]
    with pytest.raises(Refuse):
        audio.pick_section(timeline, "hook:3")
    with pytest.raises(Refuse):
        audio.pick_section(timeline, "bars:30-99")


def test_short_fragments_fold_into_the_neighbour_they_sound_like():
    """A 2-bar fill between two 8-bar sections joins the one it resembles; nothing under 4 bars survives."""
    F = np.r_[np.zeros((8, 3)), np.full((2, 3), 0.9), np.ones((8, 3))]
    segs = audio._absorb_short(F, [[0, 8, 0], [8, 10, 1], [10, 18, 2]])
    assert segs == [[0, 8, 0], [8, 18, 2]]
    assert all(e - a >= audio.MIN_SECTION_BARS for a, e, _ in segs)
    # a song shorter than the minimum keeps its one section
    tiny = audio._absorb_short(F[:3], [[0, 1, 0], [1, 3, 1]])
    assert len(tiny) == 1 and tiny[0][:2] == [0, 3]
