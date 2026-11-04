"""Edit by sentence: the artist says what to add ("fire from his hands on heat"), Muse Wonder turns it into tracked
effects, and plain code places each one on the sung word, so the render can draw it (`render --edits`).

James, Nov 4: "imagine it has an app ... once you're done, you're able to edit ... place tracked explosions and visual
effects down ... like a real video editor, all with Meta's abilities". The editor's spine is the edit file; this is
its first front door. Wonder names the part to track (a short phrase SAM 3.1 can segment), the word, its section and
the effect. Nothing it says is trusted for timing: the word must be sung in that section (the word times from
scripts/align_lyrics.py), and its time comes from there. Anything that fails is dropped and reported."""
import json
import re
from pathlib import Path

from realone.common import Refuse
from realone.director.backend import text_part, user
from realone.director.schema import parse_json, validate

LEAD_S = 0.05
MAX_TRACK_WORDS = 5

INSTRUCTIONS = """You are the editor of a finished music video. The artist tells you in one sentence what to add.
Turn it into tracked effects. For each effect give:
- track: the part of the artist to follow, as a short phrase a segmentation model can find in the footage ("his
  hands", "his right hand", "his shoes", "his hood"). Never a whole scene, never another person.
- word: the sung word the effect lands on, exactly as written in that section's lyric.
- section: the index of that section.
- effect: one of the effects listed.
- seconds: how long it lasts (0.3 to 4).
Add only what the sentence asks for. If it names a word that is sung more than once, pick the section it most
plausibly means and say why. Return only the JSON."""

EDIT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["fx", "why"],
    "properties": {
        "fx": {"type": "array", "maxItems": 6, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["track", "word", "section", "effect", "seconds", "why"],
            "properties": {"track": {"type": "string", "minLength": 2, "maxLength": 60},
                           "word": {"type": "string", "minLength": 1, "maxLength": 40},
                           "section": {"type": "integer", "minimum": 0},
                           "effect": {"type": "string"},
                           "seconds": {"type": "number"},
                           "why": {"type": "string", "maxLength": 240}}}},
        "why": {"type": "string", "maxLength": 400}}}


def _norm(w):
    return re.sub(r"[^a-z']", "", str(w).lower().replace("’", "'")).strip("'")


def place(obj, word_times, effects):
    """Wonder's reply -> (edit fx entries on the song clock, dropped with reasons). word_times: the aligner's JSON
    (scripts/align_lyrics.py) as a dict."""
    from realone.moments import VIOLENCE
    lines = word_times.get("lines", [])
    kept, dropped = [], []
    for e in obj.get("fx", []):
        why = None
        track, word, sec = str(e.get("track", "")).strip(), _norm(e.get("word", "")), e.get("section")
        hits = [w for ln in lines if ln.get("section") == sec for w in ln.get("words", []) if _norm(w["w"]) == word]
        if e.get("effect") not in effects:
            why = f"unknown effect {e.get('effect')!r}"
        elif not track or len(track.split()) > MAX_TRACK_WORDS:
            why = f"track {track!r} is not a short phrase"
        elif any(_norm(t) in VIOLENCE for t in track.split()) or word in VIOLENCE:
            why = "a word on the violence list"
        elif not hits:
            why = f"'{e.get('word')}' is not sung in section {sec}"
        if why:
            dropped.append({"fx": e, "why": why})
            continue
        secs = min(4.0, max(0.3, float(e.get("seconds", 1.4))))
        kept.append({"track": track, "at_s": round(hits[0]["start"] - LEAD_S, 3), "effect": e["effect"],
                     "seconds": secs, "word": hits[0]["w"], "section": sec, "why": e.get("why", "")})
    return kept, dropped


def edit(backend, rec, word_times, sentence, effects, effort="medium"):
    """One Wonder call -> the edit record ({"fx": [...], ...})."""
    brief = {"effects": effects, "sentence": sentence,
             "sections": [{"index": s["index"], "role": s.get("role"), "lyric": s.get("lyric", [])}
                          for s in rec["sections"]]}
    b = json.dumps(brief, indent=1)
    turn = backend.respond(purpose="edit", instructions=INSTRUCTIONS,
                           input=[user(text_part("The video's sections, the effects, and what the artist said (JSON):\n"
                                                 + b))],
                           schema=EDIT_SCHEMA, schema_name="tracked_edits", effort=effort,
                           est_input_tokens=len(b) // 4 + 600)
    obj, errs = parse_json(turn.text)
    errs = errs or validate(obj, EDIT_SCHEMA)
    if errs:
        raise Refuse("edit: the reply did not validate: " + "; ".join(errs[:6]))
    kept, dropped = place(obj, word_times, effects)
    if not kept:
        raise Refuse("edit: nothing placeable in the reply: " + "; ".join(d["why"] for d in dropped))
    return {"fx": kept, "dropped": dropped, "said": sentence, "why": obj.get("why", ""), "source": turn.source,
            "response_id": turn.id, "usd": round(turn.usd, 5)}


def load_word_times(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise Refuse(f"--word-times {path}: cannot read it ({e})") from None
