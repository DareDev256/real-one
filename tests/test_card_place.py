"""The editor's text placement: edits.json "cards" moves a lyric line to a sign in the world, a bottom card or a top
card (James, Nov 4: "move the text while keeping it still tracked in 3D space"). fx.load_cards validates;
lyrictext.apply_card_places maps the treatment's line index onto the render's phrases."""
import json

import numpy as np
import pytest

from realone import fx as FX
from realone import lyrictext as LT
from realone.common import Refuse

SECTION_LINES = {0: ["one two three"], 1: ["four five six", "seven eight"]}
# the render split the second section's first line into two phrases
LINES = [["one", "two", "three"], ["four", "five"], ["six"], ["seven", "eight"]]
LINE_SECTION = [0, 1, 1, 1]
LINE_ID = np.array([0, 0, 1, 1, 2, 2, 3, 3, -1], np.int16)


def test_place_override_applies_to_every_phrase_of_the_line():
    card = np.zeros(9, np.int8)
    out, rep = LT.apply_card_places(card, LINE_ID, LINES, LINE_SECTION, SECTION_LINES, [{"line": 1, "place": "top"}])
    assert out.tolist() == [0, 0, 2, 2, 2, 2, 0, 0, 0]
    assert [r["phrase"] for r in rep] == [1, 2]
    out, _ = LT.apply_card_places(np.ones(9, np.int8), LINE_ID, LINES, LINE_SECTION, SECTION_LINES,
                                  [{"line": 2, "place": "world"}])
    assert out.tolist() == [1, 1, 1, 1, 1, 1, 0, 0, 1]   # only "seven eight" goes back into the world


def test_line_out_of_range_refuses():
    with pytest.raises(Refuse):
        LT.apply_card_places(np.zeros(9, np.int8), LINE_ID, LINES, LINE_SECTION, SECTION_LINES, [{"line": 3, "place": "top"}])


@pytest.mark.parametrize("cards", [[{"line": 0, "place": "middle"}], [{"line": -1, "place": "top"}],
                                   [{"line": "0", "place": "top"}], [{"line": 0, "place": "top", "size": 3}], {"line": 0}])
def test_bad_cards_refuse(tmp_path, cards):
    p = tmp_path / "edits.json"
    p.write_text(json.dumps({"cards": cards}))
    with pytest.raises(Refuse):
        FX.load_edits(p)


def test_a_cards_only_file_loads_with_no_effects(tmp_path):
    p = tmp_path / "edits.json"
    p.write_text(json.dumps({"cards": [{"line": 4, "place": "bottom", "text": "Every pixel real"}], "said": "editor"}))
    assert FX.load_edits(p) == []
    assert FX.load_cards(p) == [{"line": 4, "place": "bottom"}]
    p.write_text(json.dumps({"fx": [{"track": "his hands", "at_s": 32.817, "effect": "fire", "seconds": 1.4}],
                             "cards": [{"line": 0, "place": "world"}]}))
    assert len(FX.load_edits(p)) == 1 and FX.load_cards(p)[0]["place"] == "world"
