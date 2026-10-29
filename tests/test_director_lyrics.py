"""Lyrics per section: transcript turns mapped onto sections, the artist's sheet aligned in order and outranking
the transcript, chunk offsets. No network: transcription goes through the FIXTURE backend."""
import pytest

from realone.common import Refuse
from realone.director import lyrics as L
from realone.director.backend import FixtureBackend


def secs():
    rows = [(0, "intro", False, 0.4, 10.2), (1, "verse", False, 10.2, 29.79), (2, "hook", True, 29.79, 49.38),
            (3, "bridge", False, 49.38, 68.97), (4, "hook", True, 68.97, 88.57)]
    return [{"index": i, "role": r, "is_hook": h, "start_s": a, "end_s": b} for i, r, h, a, b in rows]


def turn(a, b, text, tid=1):
    return {"turnId": tid, "startMs": int(a * 1000), "endMs": int(b * 1000), "transcript": text}


def test_turns_land_in_the_section_they_overlap_most():
    m = L.map_turns([turn(11.0, 15.0, "inside the verse"),
                     turn(27.0, 36.0, "straddles verse and hook, mostly hook"),   # 2.79 s verse, 6.21 s hook
                     turn(26.0, 31.0, "straddles, mostly verse"),                 # 3.79 s verse, 1.21 s hook
                     turn(89.0, 90.5, "after the last section"),                  # in no section: nearest
                     turn(40.0, 41.0, "")], secs())                               # empty turn: dropped
    assert [t["text"] for t in m[1]] == ["inside the verse", "straddles, mostly verse"]
    assert [t["text"] for t in m[2]] == ["straddles verse and hook, mostly hook"]
    assert [t["text"] for t in m[4]] == ["after the last section"]
    assert m[0] == [] and m[3] == []


def test_chunk_offsets_are_added(song, monkeypatch, tmp_path):
    """A long song is sent in chunks; each chunk's turns come back in chunk time and must land in song time."""
    monkeypatch.setattr(L, "CHUNK_S", 30)
    same = {"transcript": "x", "turns": [turn(1.0, 2.0, "line")], "audioDurationMs": 30000}
    fb = FixtureBackend({"transcribe": [same, same, same, same]})
    out = L.transcribe_song(fb, song[0], tmp_path)
    assert [t["startMs"] for t in out["turns"]] == [1000, 31000, 61000, 91000]
    assert out["source"].startswith("FIXTURE")
    assert sorted(p.name for p in tmp_path.glob("asr_*.wav")) == ["asr_00.wav", "asr_01.wav", "asr_02.wav",
                                                                   "asr_03.wav"]


SHEET = """# a comment line is skipped
[Verse 1]
Told my mama I'm good

[Chorus]
I'm good, don't call it

[Verse 2]
Bus pass, two jobs

[Chorus]
I'm good, don't call it"""


def test_sheet_aligns_in_order_and_drops_the_instrumental_intro():
    out, warns = L.align_blocks(L.parse_lyrics_text(SHEET), secs())
    assert out == {1: ["Told my mama I'm good"], 2: ["I'm good, don't call it"], 3: ["Bus pass, two jobs"],
                   4: ["I'm good, don't call it"]}
    assert warns == []


def test_index_tags_bind_and_the_rest_fill_in_order():
    out, _ = L.align_blocks(L.parse_lyrics_text("[#4]\nlast hook\n\nfirst\n\nsecond\n\nthird"), secs())
    assert out == {4: ["last hook"], 1: ["first"], 2: ["second"], 3: ["third"]}


def test_a_hook_label_on_a_non_hook_section_warns():
    _, warns = L.align_blocks(L.parse_lyrics_text("[Chorus]\na\n\nb\n\nc\n\nd"), secs())
    assert warns and "#1" in warns[0]


@pytest.mark.parametrize("text", ["a\n\nb", "a\n\nb\n\nc\n\nd\n\ne\n\nf", "[#9]\nx"])
def test_ambiguous_sheets_refuse(text):
    with pytest.raises(Refuse):
        L.align_blocks(L.parse_lyrics_text(text), secs())


def test_artist_text_outranks_the_transcript():
    transcript = {"turns": [turn(12, 20, "told my mama i'm could"), turn(50, 60, "bus pass to jobs")],
                  "source": "FIXTURE"}
    artist = {1: ["Told my mama I'm good"]}
    rows = {r["index"]: r for r in L.build(secs(), artist, transcript)["sections"]}
    assert rows[1]["lines"] == ["Told my mama I'm good"] and rows[1]["source"] == "artist"
    assert rows[1]["heard"][0]["text"] == "told my mama i'm could"          # kept, not used
    assert rows[3]["lines"] == ["bus pass to jobs"] and rows[3]["source"] == "transcript"
    assert rows[0]["source"] == "none" and rows[0]["lines"] == []


def test_keywords_come_from_the_sheet():
    kw = L.keywords_from({1: ["Told my mama I'm good"], 3: ["Bus pass, overtime"]})
    assert "mama" in kw and "overtime" in kw and "good" in kw and "i'm" not in kw
