"""The treatment: Muse Wonder reads the timeline, the lyrics and the song, and directs from what the song MEANS.

Flow (each step at most once):
1. one Wonder call, `text.format` json_schema (strict), song audio as `input_audio` by file id when given;
2. local validation (JSON Schema + semantic rules); on failure ONE repair turn that names every error
   (`previous_response_id`, so the song is not resent); a second failure refuses;
3. the literal-lyrics guard on every world_prompt; flagged sections get ONE re-ask naming the offending words.
   Only flagged sections may change. Whatever is still literal after that is reported, never hidden.
"""
import copy
import json

from realone.common import Refuse
from realone.director import guard
from realone.director.backend import AudioRejected, text_part, user
from realone.director.budget import INSTRUCTION_TOKENS, MAX_OUT, text_tokens
from realone.director.lyrics import lines_by_index
from realone.director.schema import (REWRITE_SCHEMA, TREATMENT_SCHEMA, check_treatment, parse_json, people_in,
                                     validate)

INSTRUCTIONS = """You direct a music video made from one real phone performance.
The artist is filmed and stays real: SAM 3.1 cuts them out of their own footage, pixel for pixel. You design the
world behind them, one world per section of the song, which Muse Image will build. Cuts already land on the beat.

Your rule is the Subtext Cut: direct from what the song MEANS, not from what its words SAY. AI music videos get
mocked for illustrating lyrics literally (a line like "watch me move" becoming a wristwatch). Read the situation
under the words: who is pretending, what is being avoided, what changes between the first hook and the last.

For every section of the timeline (same index, same order) return:
- subtext: what this section means underneath its words, in your own words.
- world_prompt: a PLACE for Muse Image, vertical 9:16, with no people in it (the filmed artist is the only person
  in frame and is added later). Carry the subtext through architecture, light, weather, time of day, the state of
  a room. Never draw what the lyric names, and do not reuse the lyric's words: a code guard rejects prompts that
  share the section's concrete nouns (or stand-ins for them) or more than a fifth of its words.
- palette: 3 to 5 hex colours.
- camera_energy: low, mid or high (how hard the reframes push).
- look: clean (a photographic composite), collage (a paper cut-out with a 12 fps hand-made boil; playful), or
  receipt (the generated world drains to grey and only the filmed artist keeps colour; use it where the song is
  about what is real).
- freeze: true on exactly ONE hook section, false everywhere else. On that hook the whole world stops and only the
  artist keeps moving. Choose the hook that earns it.
- move: the camera move, timed to the bars. none on most sections (the camera simply follows the phone).
  dolly at most ONCE in the song: a dolly zoom, the artist keeps their size while the world stretches behind them;
  the moment the ground shifts (a realisation, a drop, a turn). crash: a snap zoom to the artist's face on every
  downbeat; emphasis, confrontation, a punchline section. swap: a new place on every bar (the song's other
  worlds); restlessness, no home, a montage of everywhere. Never swap on the frozen hook. A move is earned by what
  the section means, the same as its world.
- why: one line, the directing reason.
Also song_subtext: the through-line of the whole song.
Sections with no lyric are instrumental: direct them from the music. If audio is attached, listen for energy and
turns; the timeline's numbers are exact. Return only the JSON."""


def brief(timeline, lyrics):
    """The text half of the treatment request: exact timing from our analysis, lyrics with their source."""
    by = {r["index"]: r for r in lyrics["sections"]}
    secs = []
    for s in timeline["sections"]:
        r = by.get(s["index"], {})
        secs.append({"index": s["index"], "role": s["role"], "label": s.get("label"), "is_hook": s.get("is_hook"),
                     "start_s": round(s["start_s"], 2), "end_s": round(s["end_s"], 2), "bars": s.get("bars"),
                     "energy_db": s.get("energy_db"), "lyric": r.get("lines", []),
                     "lyric_source": r.get("source", "none")})
    return {"song": {"duration_s": round(timeline["duration_s"], 2), "bpm": round(timeline["bpm"], 2),
                     "beats_per_bar": timeline["beats_per_bar"], "sections": len(secs)},
            "sections": secs}


def repair_message(errors):
    return ("Your treatment JSON did not pass validation. Fix exactly these errors and return the full corrected "
            "treatment (every section, same schema):\n" + "\n".join(f"- {e}" for e in errors))


def make(backend, timeline, lyrics, audio_file_id=None, effort="high", log=None):
    """-> {"treatment", "guard", "attempts", "audio_in", "last_response_id"}. Refuses after one failed repair."""
    log = log if log is not None else []
    sections = timeline["sections"]
    b = brief(timeline, lyrics)
    btxt = json.dumps(b, indent=1)
    est = INSTRUCTION_TOKENS + text_tokens(len(btxt))
    parts = [text_part("Timeline and lyrics (JSON):\n" + btxt)]
    audio_in = "not sent"
    if audio_file_id:
        parts.append({"type": "input_audio", "file_id": audio_file_id})
        audio_in = "sent as input_audio (file id)"
    try:
        turn = backend.respond(purpose="treatment", instructions=INSTRUCTIONS, input=[user(*parts)],
                               schema=TREATMENT_SCHEMA, schema_name="real_one_treatment", effort=effort,
                               est_input_tokens=est)
    except AudioRejected as e:
        audio_in = f"REJECTED by the API ({e}); this treatment was made from the timeline and lyrics only"
        print(f"director: the API refused the song as audio input; continuing WITHOUT it (logged): {e}")
        turn = backend.respond(purpose="treatment", instructions=INSTRUCTIONS, input=[user(parts[0])],
                               schema=TREATMENT_SCHEMA, schema_name="real_one_treatment", effort=effort,
                               est_input_tokens=est)
    tr, errs = check_treatment(turn.text, sections)
    attempts = [{"purpose": "treatment", "source": turn.source, "response_id": turn.id, "usd": round(turn.usd, 5),
                 "errors": errs}]
    log.append({"step": "treatment", "source": turn.source, "errors": errs})
    if errs:
        print(f"director: treatment failed validation ({len(errs)} errors); one repair turn")
        turn = backend.respond(purpose="treatment_repair", instructions=INSTRUCTIONS,
                               input=[user(text_part(repair_message(errs)))], schema=TREATMENT_SCHEMA,
                               schema_name="real_one_treatment", previous_id=turn.id, effort="medium",
                               est_input_tokens=est + 1000)
        tr, errs = check_treatment(turn.text, sections)
        attempts.append({"purpose": "treatment_repair", "source": turn.source, "response_id": turn.id,
                         "usd": round(turn.usd, 5), "errors": errs})
        log.append({"step": "treatment_repair", "source": turn.source, "errors": errs})
        if errs:
            raise Refuse("the treatment failed validation twice (after one repair turn): " + "; ".join(errs[:8]))

    lyr = lines_by_index(lyrics)
    reports = guard.check_treatment(tr, lyr)
    first = copy.deepcopy(reports)
    first_prompts = {s["index"]: s["world_prompt"] for s in tr["sections"]}
    flagged = {i: r for i, r in reports.items() if r["verdict"] == "reject"}
    reask = None
    if flagged:
        print(f"director: literal-lyrics guard rejected sections {sorted(flagged)}; one re-ask naming the words")
        msg = guard.reask_message(flagged, tr, lyr)
        turn = backend.respond(purpose="guard_reask", instructions=INSTRUCTIONS, input=[user(text_part(msg))],
                               schema=REWRITE_SCHEMA, schema_name="world_rewrites", previous_id=turn.id,
                               effort="medium", est_input_tokens=est + MAX_OUT["treatment"] + 800)
        obj, perrs = parse_json(turn.text)
        perrs = perrs or validate(obj, REWRITE_SCHEMA)
        applied, ignored = [], []
        if not perrs:
            by = {s["index"]: s for s in tr["sections"]}
            for rw in obj["rewrites"]:
                i = rw["index"]
                if i not in flagged:
                    ignored.append({"index": i, "why": "not a flagged section; only flagged sections may change"})
                elif people_in(rw["world_prompt"]):
                    ignored.append({"index": i, "why": f"describes people ({', '.join(people_in(rw['world_prompt']))})"})
                else:
                    by[i]["world_prompt"] = rw["world_prompt"]
                    by[i]["why"] = rw["why"]
                    applied.append(i)
        reports = guard.check_treatment(tr, lyr)
        reask = {"source": turn.source, "response_id": turn.id, "usd": round(turn.usd, 5), "asked": sorted(flagged),
                 "message": msg, "errors": perrs, "applied": applied, "ignored": ignored}
        log.append({"step": "guard_reask", "source": turn.source, "asked": sorted(flagged), "applied": applied,
                    "errors": perrs})
    still = sorted(i for i, r in reports.items() if r["verdict"] == "reject")
    return {"treatment": tr, "attempts": attempts, "audio_in": audio_in, "last_response_id": turn.id,
            "guard": {"threshold": guard.THRESHOLD, "first_pass": first, "first_prompts": first_prompts,
                      "final": reports, "reask": reask, "still_literal": still}}


def record(result, timeline, lyrics, meta):
    """The treatment file: one row per section with timing, lyric, treatment and guard side by side."""
    lyr = {r["index"]: r for r in lyrics["sections"]}
    tr = {s["index"]: s for s in result["treatment"]["sections"]}
    g = result["guard"]
    rows = []
    for s in timeline["sections"]:
        i = s["index"]
        rows.append({"index": i, "role": s["role"], "is_hook": s.get("is_hook", False), "start_s": s["start_s"],
                     "end_s": s["end_s"], "bars": s.get("bars"), "lyric": lyr.get(i, {}).get("lines", []),
                     "lyric_source": lyr.get(i, {}).get("source", "none"), **{k: v for k, v in tr[i].items()
                                                                              if k != "index"},
                     "guard": g["final"][i],
                     "guard_first_pass": None if g["first_pass"][i] == g["final"][i] else
                     {**g["first_pass"][i], "prompt": g["first_prompts"][i]}})
    return {**meta, "song_subtext": result["treatment"]["song_subtext"], "audio_in": result["audio_in"],
            "attempts": result["attempts"], "guard_reask": result["guard"]["reask"],
            "still_literal": result["guard"]["still_literal"], "guard_threshold": result["guard"]["threshold"],
            "sections": rows}


def from_record(rec, timeline, lyrics):
    """Re-check a treatment file (hand-edited or from an earlier run): schema, semantics, guard. No model call."""
    tr = {"song_subtext": rec.get("song_subtext", ""),
          "sections": [{k: r[k] for k in ("index", "subtext", "world_prompt", "palette", "camera_energy", "look",
                                           "freeze", "move", "why") if k in r} for r in rec.get("sections", [])]}
    tr, errs = check_treatment(tr, timeline["sections"])
    if errs:
        raise Refuse("the treatment file does not validate against this song's timeline: " + "; ".join(errs[:8]))
    reports = guard.check_treatment(tr, lines_by_index(lyrics))
    still = sorted(i for i, r in reports.items() if r["verdict"] == "reject")
    return {"treatment": tr, "attempts": [{"purpose": "treatment_file", "source": "file", "errors": []}],
            "audio_in": rec.get("audio_in", "n/a (from file)"), "last_response_id": None,
            "guard": {"threshold": guard.THRESHOLD, "first_pass": reports, "first_prompts": {},
                      "final": reports, "reask": None, "still_literal": still}}
