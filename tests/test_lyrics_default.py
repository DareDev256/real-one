"""Lyrics in the world are on by default when the treatment carries lyric lines (James, Nov 2: "when the text is
there, it's a lot more interesting"), off with --no-lyrics or with no lyric, and off by default on a lyric carrying
words on the violence list (forced on only by --lyrics-in-world)."""
from types import SimpleNamespace

from realone.render import lyrics_mode


def _a(**kw):
    base = {"lyrics_in_world": False, "no_lyrics": False, "_treatment_lyrics": None}
    return SimpleNamespace(**{**base, **kw})


def test_on_by_default_with_lyrics():
    assert lyrics_mode(_a(_treatment_lyrics={2: ["I'm good, I'm good"]})) == "auto"


def test_off_without_lyrics_or_when_asked():
    assert lyrics_mode(_a()) == "off"
    assert lyrics_mode(_a(_treatment_lyrics={0: [], 1: []})) == "off"
    assert lyrics_mode(_a(_treatment_lyrics={2: ["I'm good"]}, no_lyrics=True)) == "off"


def test_violent_lyric_stays_off_unless_forced():
    a = _a(_treatment_lyrics={1: ["calm line"], 3: ["Glock in the whip"]})
    assert lyrics_mode(a) == "off" and "glock" in a._lyrics_off_why
    assert lyrics_mode(_a(_treatment_lyrics={3: ["Glock in the whip"]}, lyrics_in_world=True)) == "forced"


def test_forced_wins_over_no_lyrics_absent():
    assert lyrics_mode(_a(lyrics_in_world=True)) == "forced"  # forced with no lyric: the render refuses later
