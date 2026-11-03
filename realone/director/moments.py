"""The director picks the lyric moments: one Muse Wonder call reads a finished treatment (each section's lyric and
the subtext it already wrote) and names the few words that deserve a visual hit on the frame they land.

Effects (rendered by realone/moments.py): ignite (an ember burst behind the artist and a warm flash) and slam (the
word slams in huge in front of him). Restraint is the brief, from James's bar "what would Kendrick Lamar do?": a few
conceptual moments, not effects everywhere.

Checked locally before anything is kept: every word must be sung in the section it names, at most MAX_PER_SECTION
a section and MAX_TOTAL in the song, and never a word from the local VIOLENCE list (Meta employees judge the
hackathon; a slammed gun line is a judging risk, Nov 2). Anything that fails is dropped and reported, never
silently fixed."""
import json

from realone.common import Refuse
from realone.director.backend import text_part, user
from realone.director.schema import parse_json, validate
from realone.moments import EFFECTS, VIOLENCE, _norm
from realone.lyrictext import words_of

MAX_PER_SECTION = 2
MAX_TOTAL = 8

INSTRUCTIONS = """You are the director of a music video. The treatment (one world per section) is done. Now pick the
lyric moments: the few words that get a visual hit on the exact frame they are sung.

Two effects:
- ignite: an ember burst rises behind the artist and the world takes a warm flash. For words of fire, heat or
  burning, or a word where something catches, breaks open or turns.
- slam: the word itself slams in huge in front of the artist on its beat, then gets out of the way. For the one
  word a hook is about, a punchline, the word the whole section turns on.

Restraint is the point. Think of what Kendrick Lamar's videos do: a few conceptual moments that land, never effects
on every line. Most verse sections get 0 or 1. At most 2 in any section and 8 in the whole song. A hook that repeats
can carry the same moment each time it comes back.
Never pick a slur, a profanity, or a word about guns or violence.
Pick each word exactly as it is written in that section's lyric. Give one short line of why for each.
Return only the JSON."""

MOMENTS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["picks", "why"],
    "properties": {  # "picks", not "moments": a treatment's `moments` is LOCAL_ONLY and wire_schema strips that name
        "picks": {"type": "array", "maxItems": MAX_TOTAL, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["section", "word", "effect", "why"],
            "properties": {
                "section": {"type": "integer", "minimum": 0},
                "word": {"type": "string", "minLength": 1, "maxLength": 40,
                         "description": "one word, exactly as written in that section's lyric"},
                "effect": {"type": "string", "enum": list(EFFECTS)},
                "why": {"type": "string", "minLength": 4, "maxLength": 240},
            }}},
        "why": {"type": "string", "maxLength": 400, "description": "the overall reasoning, one or two lines"},
    },
}


def brief(rec):
    """What Wonder reads: per section its role, lyric and the subtext the treatment already wrote."""
    return {"song_subtext": rec.get("song_subtext", ""),
            "sections": [{"index": r["index"], "role": r.get("role"), "is_hook": r.get("is_hook", False),
                          "lyric": r.get("lyric", []), "subtext": r.get("subtext", "")}
                         for r in rec.get("sections", [])]}


def check(obj, rec):
    """-> (kept: {section index: [{"word", "effect", "why"}]}, dropped: [{"moment", "why"}])."""
    lyric = {r["index"]: {_norm(w) for line in r.get("lyric", []) for w in words_of(line)}
             for r in rec.get("sections", [])}
    kept, dropped, total = {}, [], 0
    for m in obj.get("picks", []):
        sec, word = m.get("section"), _norm(m.get("word", ""))
        why = None
        if sec not in lyric:
            why = f"section {sec} is not in the treatment"
        elif not word or " " in str(m.get("word", "")).strip():
            why = "not a single word"
        elif word not in lyric[sec]:
            why = f"'{m.get('word')}' is not sung in section {sec}"
        elif word in VIOLENCE:
            why = f"'{m.get('word')}' is on the local violence list"
        elif m.get("effect") not in EFFECTS:
            why = f"unknown effect {m.get('effect')}"
        elif len(kept.get(sec, [])) >= MAX_PER_SECTION:
            why = f"section {sec} already has {MAX_PER_SECTION} moments"
        elif total >= MAX_TOTAL:
            why = f"the song already has {MAX_TOTAL} moments"
        if why:
            dropped.append({"moment": m, "why": why})
            continue
        kept.setdefault(sec, []).append({"word": m["word"], "effect": m["effect"], "why": m.get("why", "")})
        total += 1
    return kept, dropped


def pick(backend, rec, effort="medium"):
    """One Wonder call -> the treatment record with `moments` on each section, plus a report. Refuses on a reply that
    is not valid JSON for the schema (one call, no repair turn: nothing is guessed)."""
    b = json.dumps(brief(rec), indent=1)
    turn = backend.respond(purpose="moments", instructions=INSTRUCTIONS,
                           input=[user(text_part("Treatment sections with their lyric (JSON):\n" + b))],
                           schema=MOMENTS_SCHEMA, schema_name="lyric_moments", effort=effort,
                           est_input_tokens=len(b) // 4 + 600)
    obj, errs = parse_json(turn.text)
    errs = errs or validate(obj, MOMENTS_SCHEMA)
    if errs:
        raise Refuse("the moments reply did not validate: " + "; ".join(errs[:6]))
    kept, dropped = check(obj, rec)
    out = {**rec, "sections": [{**r, "moments": kept.get(r["index"], [])} for r in rec["sections"]],
           "moments_pick": {"source": turn.source, "response_id": turn.id, "usd": round(turn.usd, 5),
                            "why": obj.get("why", ""), "dropped": dropped,
                            "kept": sum(len(v) for v in kept.values())}}
    return out
