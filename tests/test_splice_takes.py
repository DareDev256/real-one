"""Splicing renders of different takes (scripts/splice_takes.py, Nov 4): each section comes from the render of the
take the director picked, cut frame-exact on the one song clock."""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("splice_takes", REPO / "scripts" / "splice_takes.py")
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

SECS = [{"index": 0, "start_s": 0.037, "end_s": 6.8985}, {"index": 1, "start_s": 6.8985, "end_s": 20.6331},
        {"index": 2, "start_s": 20.6331, "end_s": 41.1712}, {"index": 3, "start_s": 41.1712, "end_s": 54.917}]
# section renders start on their own section; the full render starts on the song's first downbeat
PARTS = {0: ("t3_s0.mp4", 0.037, 206), 1: ("t1_s1.mp4", 6.8985, 412), 2: ("t3_s2.mp4", 20.6331, 616),
         3: ("t2_all.mp4", 0.037, 1646)}


def test_each_section_is_cut_from_its_part_on_the_song_clock():
    segs = sp.plan(SECS, PARTS, 0.037)
    assert [v for v, _, _ in segs] == ["t3_s0.mp4", "t1_s1.mp4", "t3_s2.mp4", "t2_all.mp4"]
    assert segs[0][1:] == (0, 206) and segs[1][1:] == (0, 412) and segs[2][1:] == (0, 616)
    assert segs[3][1:] == (1234, 1646)          # the last hook from the full render, on its own frame grid
    assert sum(b - a for _, a, b in segs) == 1646  # the joined video is the full song, frame for frame


def test_a_part_that_does_not_cover_its_section_is_refused():
    short = dict(PARTS)
    short[2] = ("t3_s2.mp4", 20.6331, 500)
    with pytest.raises(SystemExit, match="needs 0-616"):
        sp.plan(SECS, short, 0.037)
    late = dict(PARTS)
    late[1] = ("t1_s1.mp4", 7.5, 412)              # starts after its section does
    with pytest.raises(SystemExit, match="REFUSED"):
        sp.plan(SECS, late, 0.037)
    with pytest.raises(SystemExit, match="no part for section 2"):
        sp.plan(SECS, {k: v for k, v in PARTS.items() if k != 2}, 0.037)



def test_a_later_start_trims_the_intro_on_the_song_clock():
    """--start-bar 3 (5.18 s): the intro part is cut from its frame 154, and the joined video is shorter by that."""
    s0 = 0.037 + round((5.1834 - 0.037) * 30) / 30   # the downbeat, snapped onto the parts' grid (as main() does)
    segs = sp.plan(SECS, PARTS, s0)
    assert segs[0][0] == "t3_s0.mp4" and segs[0][1] == round((s0 - 0.037) * 30) and segs[0][2] == 206
    assert sum(b - a for _, a, b in segs) == 1646 - round((s0 - 0.037) * 30)
    late = sp.plan(SECS, PARTS, 0.037 + round((8.0 - 0.037) * 30) / 30)  # inside the first hook: no intro
    assert late[0][0] == "t1_s1.mp4" and len(late) == 3
