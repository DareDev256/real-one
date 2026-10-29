"""Sync: recover a known offset within 1 frame; refuse garbage and genuinely ambiguous takes."""
import subprocess

import numpy as np
import pytest
import soundfile as sf

import make_test_song as mts
from conftest import to22
from realone import sync as S

FRAME = 1 / 30


def test_offset_recovered_within_one_frame(master22, song, live_vocal):
    """A 26 s phone take (playback through a room, the artist singing a separate take live at 0 dB, noise at
    20 dB) that starts 26.317 s into the song."""
    mic = to22(mts.phone_mic(song[1], 26.317, 26.0, seed=1, live_vocal=live_vocal))
    r = S.measure_offset(master22, mic)
    assert abs(r["offset_s"] - 26.317) < FRAME, r


def test_offset_with_the_artist_louder_than_the_playback(master22, song, live_vocal):
    mic = to22(mts.phone_mic(song[1], 26.317, 26.0, seed=2, live_vocal=live_vocal, vocal_db=6, snr_db=10))
    r = S.measure_offset(master22, mic)
    assert abs(r["offset_s"] - 26.317) < FRAME, r


def test_take_that_starts_before_the_song(master22, song, live_vocal):
    mic = to22(mts.phone_mic(song[1], -2.0, 18.0, seed=3, live_vocal=live_vocal))
    r = S.measure_offset(master22, mic)
    assert abs(r["offset_s"] + 2.0) < FRAME, r


def test_mp4_end_to_end_with_audio_starting_late(tmp_path, song, live_vocal):
    """Through a real container: AAC encode/decode, and an audio track that starts 0.25 s after the video.
    song_time = video_time + offset, so the offset must come out 0.25 s smaller than the mic's own.
    (26 s, like the main take: a 20 s take ending inside hook 1 is ambiguous with hook 2 and rightly refuses.)"""
    wav = tmp_path / "mic.wav"
    sf.write(str(wav), mts.phone_mic(song[1], 26.317, 26.0, seed=4, live_vocal=live_vocal), mts.SR)
    mp4 = tmp_path / "take.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=30:d=26.5",
                    "-itsoffset", "0.25", "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                    "-c:a", "aac", "-t", "26.5", str(mp4)], check=True)
    r = S.sync_clip(song[0], mp4)
    assert r["audio_start_s"] - r["video_start_s"] > 0.2  # the skew is really in the file
    assert abs(r["offset_s"] - (26.317 - 0.25)) < FRAME, r


@pytest.mark.parametrize("kind", ["white noise", "other song"])
def test_garbage_refuses_as_no_match(master22, kind, live_vocal):
    if kind == "white noise":
        clip = np.random.default_rng(1).standard_normal(26 * S.SR).astype(np.float32) * 0.1
    else:
        other, _ = mts.make_song(bpm=87.0, seed=5)
        clip = to22(mts.phone_mic(other, 20, 26, seed=3, live_vocal=live_vocal))
    with pytest.raises(S.SyncRefused, match="does not match"):
        S.measure_offset(master22, clip)


def test_silence_refuses(master22):
    with pytest.raises(S.SyncRefused, match="silent"):
        S.measure_offset(master22, np.zeros(20 * S.SR, np.float32))


def test_take_inside_an_identical_loop_refuses_instead_of_guessing():
    """Nine copies of the same 4 bars: any 12 s window fits several places equally. It must refuse."""
    loop, _ = mts.make_song(structure=[("verse", 4)] * 9, seed=4)
    clip = to22(mts.phone_mic(loop, 20.0, 12.0, seed=2))
    with pytest.raises(S.SyncRefused, match="ambiguous"):
        S.measure_offset(to22(loop), clip)


def test_sync_near_resolves_a_short_ambiguous_take(master22, song, live_vocal):
    """8 s that sit mostly inside a repeated hook: refused blind, recovered with a hint."""
    mic = to22(mts.phone_mic(song[1], 27.0, 8.0, seed=5, live_vocal=live_vocal))
    with pytest.raises(S.SyncRefused, match="ambiguous"):
        S.measure_offset(master22, mic)
    r = S.measure_offset(master22, mic, near_s=26.0)
    assert abs(r["offset_s"] - 27.0) < FRAME, r


def test_a_whole_song_take_syncs(master22, song, live_vocal):
    """A take of the whole song from its first second: ~59 windows, and sections that loop every 4 bars fit a
    4-bar shift almost as well, so 2 to 9 windows lose to it by more than 0.04 even at the true lag. 0.1.0 refused
    every such take as ambiguous; 0.2.0 judges a long take on all of its windows (sync.WIN_LONG_N). Seed 2 is the
    hardest measured (9 windows lost)."""
    for seed in (1, 2, 5):
        mic = to22(mts.phone_mic(song[1], 0.0, 90.0, seed=seed, live_vocal=live_vocal))
        r = S.measure_offset(master22, mic)
        assert abs(r["offset_s"] - 0.0) < FRAME, r


def test_the_long_take_rule_still_refuses_what_it_should():
    """Each condition of the long-take rule, on its own (window advantages, best lag minus competitor)."""
    rng = np.random.default_rng(0)
    good = np.r_[np.full(20, 0.25), rng.normal(0.03, 0.03, 30), np.full(6, -0.06)]  # 6/56 lost, clear wins
    assert S._beaten(good)
    assert not S._beaten(np.r_[good[:-6], np.full(6, -0.13)])           # the competitor clearly wins somewhere
    assert not S._beaten(np.r_[np.full(20, 0.25), np.full(24, -0.06)])  # loses 55% of its windows
    flat = np.r_[0.13, rng.normal(0.0, 0.05, 40)]
    flat[flat < -0.04] = -0.041
    assert not S._beaten(flat)                                         # no consistent advantage: t too low
    assert not S._beaten(np.r_[np.full(8, 0.25), np.full(2, -0.05)])    # a short take keeps the 0.1.0 rule
    assert not S._beaten(np.full(40, 0.11))                             # never a clear win
