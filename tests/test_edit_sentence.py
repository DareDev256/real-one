"""Edit by sentence (python -m realone.director edit, Nov 4): Wonder names the part, the word and the effect; plain
code places it on the sung word and drops what it cannot place."""
from realone.director import edit as ED

WORDS = {"lines": [
    {"section": 2, "line": 9, "text": "World built 'round me, feel the heat",
     "words": [{"w": "World", "start": 32.17}, {"w": "built", "start": 32.4}, {"w": "'round", "start": 32.5},
               {"w": "me,", "start": 32.6}, {"w": "feel", "start": 32.7}, {"w": "the", "start": 32.78},
               {"w": "heat", "start": 32.867}]},
    {"section": 3, "line": 3, "text": "Fire on stage, let the show begin",
     "words": [{"w": "Fire", "start": 46.269}]}]}


def reply(**kw):
    e = {"track": "his hands", "word": "heat", "section": 2, "effect": "fire", "seconds": 1.4, "why": "the heat line"}
    e.update(kw)
    return {"fx": [e], "why": ""}


def test_the_effect_lands_on_the_sung_word_not_on_the_models_say_so():
    kept, dropped = ED.place(reply(), WORDS, ["fire"])
    assert not dropped and kept[0]["at_s"] == round(32.867 - ED.LEAD_S, 3) and kept[0]["track"] == "his hands"


def test_what_cannot_be_placed_is_dropped_with_the_reason():
    for kw, says in ({"word": "heat", "section": 3}, "not sung in section 3"), ({"effect": "laser"}, "unknown effect"), \
            ({"track": "the whole crowd behind him on stage"}, "not a short phrase"), ({"track": "his gun"}, "violence"):
        kept, dropped = ED.place(reply(**kw), WORDS, ["fire"])
        assert not kept and says in dropped[0]["why"]
    kept, _ = ED.place(reply(seconds=30), WORDS, ["fire"])
    assert kept[0]["seconds"] == 4.0
