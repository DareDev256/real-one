"""Lyric moments on a full-song render (Nov 4): a pick the director made for a section lands only in that section,
and the cap counts per section. The first full render of James's take spent all four hits in its first 14 s, one of
them on the intro's "real one" that nobody picked, and the verse and the last hook got none."""
import numpy as np

from realone import lyrictext as LT
from realone import moments as MO

FPS = 30
BEAT = 0.43
BARS = 40
TL = {"beats": [i * BEAT for i in range(BARS * 4)], "downbeats": [i * 4 * BEAT for i in range(BARS)],
      "sections": [{"index": 0, "start_s": 0.0, "end_s": 4 * 4 * BEAT},
                   {"index": 1, "start_s": 4 * 4 * BEAT, "end_s": 16 * 4 * BEAT},
                   {"index": 2, "start_s": 16 * 4 * BEAT, "end_s": 28 * 4 * BEAT},
                   {"index": 3, "start_s": 28 * 4 * BEAT, "end_s": BARS * 4 * BEAT}]}
HOOK = ["I'm the real one", "fire on stage", "the real one again", "real one real one", "fire in the frame",
        "real one forever"]
LINES = {0: ["One take, I'm the real one"], 1: HOOK, 2: ["tokens in the lights ignite", "the real deal"], 3: HOOK}
PICKS = [{"word": "real", "effect": "slam", "section": 1}, {"word": "fire", "effect": "ignite", "section": 1},
         {"word": "ignite", "effect": "ignite", "section": 2},
         {"word": "real", "effect": "slam", "section": 3}, {"word": "fire", "effect": "ignite", "section": 3}]
N = int(BARS * 4 * BEAT * FPS)


def _render_like():
    secs = []
    lines, line_id, shown, _ = LT.schedule(LINES, TL, 0.0, N, FPS, None, secs)
    return lines, line_id, shown, secs


def test_each_phrase_knows_its_section():
    lines, _, _, secs = _render_like()
    assert len(secs) == len(lines)
    assert secs[0] == 0 and "real" in [MO._norm(w) for w in lines[0] + lines[1]]  # the intro sings "real"
    assert sorted(set(secs)) == [0, 1, 2, 3]
    _, _, rep = LT.for_render(LINES, TL, 0.0, N, FPS)
    assert rep["line_section"] == secs


def test_a_pick_lands_only_in_its_own_section():
    lines, line_id, shown, secs = _render_like()
    picked = MO.pick(lines, PICKS, line_section=secs)
    assert picked and all(secs[g] != 0 for g, _, _, _ in picked)  # the intro's "real one" was never picked
    reals = {secs[g] for g, _, e, w in picked if MO._norm(w) == "real"}
    assert reals == {1, 3}  # the hook pick, in both hooks; not the verse's "real deal" either
    assert {secs[g] for g, _, e, w in picked if MO._norm(w) == "ignite"} == {2}


def test_the_cap_counts_per_section_so_the_last_hook_still_hits():
    lines, line_id, shown, secs = _render_like()
    picked = MO.pick(lines, PICKS, line_section=secs)
    _, _, _, rep = MO.profile(picked, line_id, shown, N, line_section=secs)
    per = {}
    for m in rep:
        per[m["section"]] = per.get(m["section"], 0) + 1
    assert per.get(3, 0) >= 1 and per.get(2, 0) == 1  # the verse and the last hook both get theirs
    assert all(v <= MO.MAX_HITS for v in per.values())
    frames = [m["frame"] for m in rep]
    assert all(b - a >= MO.MIN_GAP for a, b in zip(frames, frames[1:]))


def test_without_sections_it_is_the_old_per_render_behaviour():
    lines, line_id, shown, _ = _render_like()
    picked = MO.pick(lines, [{k: v for k, v in p.items() if k != "section"} for p in PICKS])
    _, _, _, rep = MO.profile(picked, line_id, shown, N)
    assert len(rep) <= MO.MAX_HITS
