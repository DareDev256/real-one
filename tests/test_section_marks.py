"""The artist's section marks (--sections): song B's hook, verse and hook share one beat, so the analyzer heard one
48 s "hook" (Nov 4). Marks replace the heard sections; each snaps to a downbeat, and anything it cannot place
exactly is refused."""
from pathlib import Path

import numpy as np
import pytest

from realone.audio import analyze, parse_marks, sections_from_marks
from realone.common import Refuse
from realone.director.tools import Renderer

REPO = Path(__file__).resolve().parents[1]
DB = [0.04 + 1.718 * i for i in range(32)]  # song B's grid: 139.7 BPM, 32 bars
END = DB[-1] + 1.718
ENERGY = np.full(32, 0.1)
SONG_B = "intro@0:00,hook@6.86,verse@20.44,hook@41.2"


def test_song_b_marks_give_intro_hook_verse_hook():
    secs = sections_from_marks(parse_marks(SONG_B), DB, END, ENERGY)
    assert [(s["role"], s["start_bar"], s["end_bar"]) for s in secs] == [
        ("intro", 0, 4), ("hook", 4, 12), ("verse", 12, 24), ("hook", 24, 32)]
    assert [s["label"] for s in secs] == ["A", "B", "C", "B"]  # the two hooks are one label
    assert [s["occurrence"] for s in secs] == [1, 1, 1, 2]
    assert [s["is_hook"] for s in secs] == [False, True, False, True]
    assert secs[1]["repeats"] == 2 and secs[2]["repeats"] == 1
    assert secs[-1]["end_s"] == round(END, 4)
    assert all(abs(s["mark"]["snap_s"]) <= 1.718 / 4 for s in secs)


def test_marks_read_minutes_seconds_and_a_file(tmp_path):
    assert parse_marks("intro@0:00, hook@1:02.5") == [("intro", 0.0), ("hook", 62.5)]
    f = tmp_path / "marks.txt"
    f.write_text("# song B\nintro@0:00\nhook@6.86, verse@20.44\n")
    assert parse_marks(str(f)) == [("intro", 0.0), ("hook", 6.86), ("verse", 20.44)]


@pytest.mark.parametrize("spec, says", [
    ("chorus@0:07", "ROLE@TIME"),
    ("hook 0:07", "ROLE@TIME"),
    ("hook@soon", "seconds or m:ss"),
    ("", "no section marks"),
])
def test_unreadable_marks_are_refused(spec, says):
    with pytest.raises(Refuse, match=says):
        parse_marks(spec)


@pytest.mark.parametrize("marks, says", [
    ([("intro", 0.0), ("hook", 6.0), ("verse", 20.44)], "between bars"),           # 0.81 s past bar 3
    ([("intro", 0.0), ("hook", 6.86), ("verse", 20.44), ("hook", 42.0)], "between bars"),  # the rough 0:42 label
    ([("hook", 6.86), ("verse", 20.44)], "first bar"),
    ([("intro", 0.0), ("hook", 6.86), ("verse", 7.2)], "at least one bar"),          # both snap to bar 4
    ([("intro", 0.0), ("verse", 20.44), ("hook", 6.86)], "at least one bar"),        # out of order
    ([("intro", 0.0), ("verse", 6.86)], "no hook"),
])
def test_marks_that_cannot_be_placed_exactly_are_refused(marks, says):
    with pytest.raises(Refuse, match=says):
        sections_from_marks(marks, DB, END, ENERGY)


def test_analyze_uses_the_marks_and_says_so():
    song = REPO / "footage" / "test_song.wav"
    heard = analyze(song)
    assert heard["sections_from"] == "heard"
    db = heard["downbeats"]
    spec = f"intro@0:00,verse@{db[4]:.3f},hook@{db[12]:.3f}"
    tl = analyze(song, marks=spec)
    assert tl["sections_from"] == "the artist's marks"
    assert [(s["role"], s["start_bar"]) for s in tl["sections"]] == [("intro", 0), ("verse", 4), ("hook", 12)]
    assert tl["hook_index"] == 2
    assert tl["downbeats"] == heard["downbeats"]  # the grid is still measured, only the sections change


def test_the_director_loop_renders_with_the_same_marks():
    """`direct` renders `--section #N` in a subprocess that re-analyzes the song: without the marks, #2 would be a
    different section (or none)."""
    r = Renderer("take.mov", "song.wav", dry_run=True, help_text="", sections=SONG_B)
    argv, _ = r.argv(2, "clean", False, "a world", "t", 0.5)
    assert argv[argv.index("--sections") + 1] == SONG_B
    plain, _ = Renderer("take.mov", "song.wav", dry_run=True, help_text="").argv(2, "clean", False, "w", "t", 0.5)
    assert "--sections" not in plain
