"""Lyrics on the vocal (--word-times): each word lands when it is sung, from scripts/align_lyrics.py (Demucs vocal
stem + MMS forced alignment). On the beat grid, song B's verse (about 110 words over 48 beats) ran out at bar 20 of
24 while the vocal carried on (Nov 4)."""
import json

import numpy as np
import pytest

from realone import lyrictext as LT
from realone import moments as MO

FPS = 30
BEAT = 0.43
TL = {"beats": [i * BEAT for i in range(64)], "downbeats": [i * 4 * BEAT for i in range(16)],
      "sections": [{"index": 0, "start_s": 0.0, "end_s": 16 * BEAT}, {"index": 1, "start_s": 16 * BEAT,
                                                                       "end_s": 64 * BEAT}]}
LINES = {0: ["Fire on stage, let the show begin"], 1: ["One take, phone up, that's the start"]}
# sung times that are deliberately NOT on beats, and fast (a word every 0.18 s in section 1)
TIMES = {0: [[(1.00, 1.20), (1.25, 1.40), (1.42, 1.80), (2.10, 2.30), (2.31, 2.50), (2.52, 2.70), (2.72, 3.30)]],
         1: [[(7.10, 7.25), (7.28, 7.46), (7.46, 7.64), (7.64, 7.82), (7.82, 8.00), (8.00, 8.18), (8.18, 8.60)]]}


def first_frame_shown(line_id, shown, gid, k):
    hit = np.nonzero((line_id == gid) & (shown >= k + 1))[0]
    return int(hit[0]) if hit.size else None


def test_each_word_lands_when_it_is_sung():
    lines, line_id, shown, _ = LT.schedule(LINES, TL, 0.0, int(10 * FPS), FPS, TIMES)
    assert lines[0] == ["Fire", "on", "stage,", "let"]  # phrases of at most 4 words, as on the beat grid
    starts = [s for sec in (0, 1) for s, _ in TIMES[sec][0]]
    got = []
    for gid, ws in enumerate(lines):
        for k in range(len(ws)):
            got.append(first_frame_shown(line_id, shown, gid, k) / FPS)
    err = np.abs(np.asarray(got) - (np.asarray(starts) - LT.LEAD_S))
    assert err.max() <= 1 / FPS + 1e-6  # every word within a frame of its sung start (less the lead)


def test_a_phrase_holds_until_the_next_one_starts_but_not_forever():
    lines, line_id, shown, _ = LT.schedule(LINES, TL, 0.0, int(12 * FPS), FPS, TIMES)
    # phrase 0 ("Fire on stage, let") is up until phrase 1 ("the show begin") starts at 2.31 s
    f_next = first_frame_shown(line_id, shown, 1, 0)
    assert np.all(line_id[int(1.2 * FPS):f_next] == 0)
    # the section's last phrase ends HOLD_S after its last word, not at the end of the section
    last = [g for g, ws in enumerate(lines) if ws[-1] == "begin"][0]
    up = np.nonzero(line_id == last)[0]
    assert up[-1] / FPS <= 3.30 + LT.HOLD_S + 1 / FPS
    assert not np.any(line_id[int(4.6 * FPS):int(7.0 * FPS)] >= 0)  # the instrumental gap is empty


def test_without_word_times_it_is_still_the_beat_grid():
    a = LT.schedule(LINES, TL, 0.0, int(10 * FPS), FPS)
    b = LT.schedule(LINES, TL, 0.0, int(10 * FPS), FPS, None)
    assert all(np.array_equal(x, y) for x, y in zip(a[1:], b[1:]))
    _, _, rep = LT.for_render(LINES, TL, 0.0, int(10 * FPS), FPS)
    assert rep["timing"] == LT.BEAT_TIMING
    _, arr, rep = LT.for_render(LINES, TL, 0.0, int(10 * FPS), FPS, TIMES, "MMS")
    assert "sung" in rep["timing"] and "MMS" in rep["timing"]
    sung = LT.schedule(LINES, TL, 0.0, int(10 * FPS), FPS, TIMES)
    assert np.array_equal(arr["lyr_line"], sung[1]) and np.array_equal(arr["lyr_shown"], sung[2])
    assert not np.array_equal(arr["lyr_shown"], a[2])  # and they differ from the beat grid


def test_a_moment_fires_on_the_sung_word():
    lines, line_id, shown, _ = LT.schedule(LINES, TL, 0.0, int(10 * FPS), FPS, TIMES)
    picked = MO.pick(lines, [{"word": "Fire", "effect": "ignite"}])
    _, _, rep = MO.profile(picked, line_id, shown, int(10 * FPS))[1:]
    assert rep and abs(rep[0]["frame"] / FPS - (1.00 - LT.LEAD_S)) <= 1 / FPS + 1e-6


def _file(tmp_path, lines=LINES, times=TIMES, drop=None):
    rec = {"source": "test", "lines": [
        {"section": s, "line": j, "text": t,
         "words": [{"w": w, "start": a, "end": b} for w, (a, b) in zip(LT.words_of(t), times[s][j])]}
        for s, ls in lines.items() for j, t in enumerate(ls)]}
    if drop:
        rec["lines"][0]["words"].pop()
    p = tmp_path / "words.json"
    p.write_text(json.dumps(rec))
    return p


def test_word_times_load_only_for_the_same_lyric(tmp_path):
    assert LT.load_word_times(_file(tmp_path), LINES) == TIMES
    other = {0: ["Fire on stage, let the show start"], 1: LINES[1]}
    with pytest.raises(LT.LyricError, match="not the treatment's line"):
        LT.load_word_times(_file(tmp_path), other)
    with pytest.raises(LT.LyricError, match="times for 7 words"):
        LT.load_word_times(_file(tmp_path, drop=True), LINES)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(LT.LyricError, match="cannot read"):
        LT.load_word_times(bad, LINES)


def test_the_render_loads_word_times_through_the_treatment(tmp_path):
    from argparse import Namespace
    from realone.common import Refuse
    from realone.render import apply_treatment
    tr = tmp_path / "t.json"
    tr.write_text(json.dumps({"sections": [{"index": s, "world_prompt": "an empty plaza", "lyric": ls}
                                           for s, ls in LINES.items()]}))
    a = Namespace(from_treatment=str(tr), word_times=str(_file(tmp_path)), section_world=None, moves_from=None)
    apply_treatment(a)
    assert a._word_times == TIMES and a._word_source == "test"
    a = Namespace(from_treatment=str(tr), word_times=str(_file(tmp_path, drop=True)), section_world=None,
                  moves_from=None)
    with pytest.raises(Refuse, match="times for 7 words"):
        apply_treatment(a)
    with pytest.raises(Refuse, match="needs --from-treatment"):
        apply_treatment(Namespace(from_treatment=None, word_times=str(_file(tmp_path))))
