"""Schemas for everything the director accepts from a model, and a small strict validator (no dependency).

Two layers, on purpose:
- the JSON Schema below. A trimmed copy (`wire_schema`) goes to Muse Wonder as `text.format` json_schema; the docs
  say decoding is constrained to it (dev.meta.ai/docs/structured-output). We still validate locally, because the
  first keyed run has not proven that guarantee for Wonder, and because some rules are not JSON Schema at all;
- `semantic_errors`: rules that need the timeline (indices, the freeze on exactly one hook) or the house rules
  (no generated people: the artist is the only person in frame).
Every error is a short string with a JSON path, so a repair turn can name it back to the model.
"""
import json
import re

LOOKS = ("clean", "collage", "receipt")  # realone/composite.py LOOKS (not imported: cv2 is not needed here)
ENERGIES = ("low", "mid", "high")
MOVES = ("none", "dolly", "crash", "swap")  # realone/stage/moves.py KINDS, plus none

# Section fields only the revision pass writes (never asked of the model in the treatment): allowed when a saved
# treatment is read back, left out of the strict schema sent to Wonder (strict mode needs every property required).
LOCAL_ONLY = {"particles", "moments"}

SECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["index", "subtext", "world_prompt", "palette", "camera_energy", "look", "freeze", "move", "why"],
    "properties": {
        "index": {"type": "integer", "minimum": 0,
                  "description": "the section index from the timeline, unchanged"},
        "subtext": {"type": "string", "minLength": 12, "maxLength": 400,
                    "description": "what this section means underneath its words"},
        "world_prompt": {"type": "string", "minLength": 20, "maxLength": 600,
                         "description": "a place for Muse Image, 9:16, empty of people, that carries the subtext; "
                                        "never an illustration of the lyric"},
        "palette": {"type": "array", "minItems": 3, "maxItems": 5,
                    "items": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
                    "description": "3 to 5 hex colours"},
        "camera_energy": {"type": "string", "enum": list(ENERGIES)},
        "look": {"type": "string", "enum": list(LOOKS)},
        "freeze": {"type": "boolean", "description": "true on exactly one hook section, false everywhere else"},
        "move": {"type": "string", "enum": list(MOVES),
                 "description": "the camera move for this section: none (most sections), dolly (at most once), "
                                "crash or swap"},
        "particles": {"type": "string", "enum": ["dust", "rain", "embers", "none"],
                      "description": "the particle overlay, set only by a revision"},
        "moments": {"type": "array", "maxItems": 2, "description": "lyric moments, set only by the moments pick "
                                                                  "(realone/director/moments.py)",
                    "items": {"type": "object", "additionalProperties": False, "required": ["word", "effect"],
                              "properties": {"word": {"type": "string", "minLength": 1, "maxLength": 40},
                                             "effect": {"type": "string", "enum": ["ignite", "slam"]},
                                             "why": {"type": "string", "maxLength": 240}}}},
        "why": {"type": "string", "minLength": 8, "maxLength": 240, "description": "one line"},
    },
}

TREATMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["song_subtext", "sections"],
    "properties": {
        "song_subtext": {"type": "string", "minLength": 12, "maxLength": 400,
                         "description": "the through-line of the whole song, one or two sentences"},
        "sections": {"type": "array", "minItems": 1, "items": SECTION_SCHEMA},
    },
}

REWRITE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rewrites"],
    "properties": {
        "rewrites": {"type": "array", "minItems": 1, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["index", "world_prompt", "why"],
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "world_prompt": SECTION_SCHEMA["properties"]["world_prompt"],
                "why": SECTION_SCHEMA["properties"]["why"],
            }}},
    },
}

NOTES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "notes", "suggestion"],
    "properties": {
        "verdict": {"type": "string", "enum": ["keep", "revise"]},
        "notes": {"type": "array", "minItems": 1, "maxItems": 8, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["t_s", "against", "severity", "note"],
            "properties": {
                "t_s": {"type": "number", "minimum": 0, "description": "seconds from the start of this render"},
                "against": {"type": "string", "enum": ["subtext", "world", "look", "freeze", "camera", "edit"]},
                "severity": {"type": "string", "enum": ["low", "mid", "high"]},
                "note": {"type": "string", "minLength": 8, "maxLength": 500},
            }}},
        "suggestion": {
            "type": "object", "additionalProperties": False,
            "required": ["field", "value", "why"],
            "properties": {
                "field": {"type": "string", "enum": ["none", "world_prompt", "look", "particles", "freeze"]},
                "value": {"type": "string", "maxLength": 600},
                "why": {"type": "string", "maxLength": 500},
            }},
    },
}

DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["revisions", "why"],
    "properties": {
        "revisions": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["section", "field", "value", "because"],
            "properties": {
                "section": {"type": "integer", "minimum": 0},
                "field": {"type": "string", "enum": ["world_prompt", "look", "particles", "freeze"]},
                "value": {"type": "string", "maxLength": 600},
                "because": {"type": "string", "minLength": 8, "maxLength": 300},
            }}},
        "why": {"type": "string", "maxLength": 400},
    },
}

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool}

# Keys the wire copy keeps. Length, range and pattern limits are checked locally instead: the strict subset on
# dev.meta.ai/docs/structured-output names root, required and additionalProperties rules but does not say which
# validation keywords it accepts, and an unknown keyword under strict mode would be an HTTP 400 on the first call.
_WIRE_KEYS = {"type", "properties", "required", "additionalProperties", "items", "enum", "description"}


def wire_schema(schema):
    """The copy sent as `text.format.schema` / tool `parameters`: structure only (see _WIRE_KEYS)."""
    out = {}
    for k, v in schema.items():
        if k not in _WIRE_KEYS:
            continue
        if k == "properties":
            out[k] = {name: wire_schema(sub) for name, sub in v.items() if name not in LOCAL_ONLY}
        elif k == "items":
            out[k] = wire_schema(v)
        elif k == "required":  # a key stripped from properties must not stay required (strict mode: HTTP 400)
            out[k] = [r for r in v if r not in LOCAL_ONLY]
        else:
            out[k] = v
    return out


def _is_type(value, t):
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    py = _TYPES[t]
    return isinstance(value, py) and not (py is not bool and isinstance(value, bool))


def validate(value, schema, path="$"):
    """-> list of error strings (empty = valid). Supports the subset the schemas above use."""
    errs = []
    t = schema.get("type")
    if t and not _is_type(value, t):
        return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} is not one of {', '.join(map(str, schema['enum']))}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errs.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{path}: {value!r} does not match {schema['pattern']}")
    if _is_type(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} is below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} is above {schema['maximum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errs.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: at most {schema['maxItems']} items")
        for i, v in enumerate(value):
            errs += validate(v, schema.get("items", {}), f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}.{k}: missing")
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    errs.append(f"{path}.{k}: not allowed")
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
    return errs


def parse_json(text):
    """-> (obj, errors). Tolerates a ```json fence around the object (a model that ignored json_schema)."""
    s = (text or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    if m:
        s = m.group(1)
    try:
        return json.loads(s), []
    except json.JSONDecodeError as e:
        return None, [f"$: not valid JSON ({e.msg} at line {e.lineno} column {e.colno})"]


# People words: Real One's rule is that the artist, filmed, is the only person in frame. A generated face is the
# backlash the brief is built against (r/eurovision, Mar 2026), and a second body confuses the cut-out.
_PEOPLE = re.compile(r"\b(man|men|woman|women|person|people|crowd|girl|girls|boy|boys|guy|guys|lady|ladies|kid|kids|"
                     r"child|children|dancer|dancers|figure|figures|someone|somebody|stranger|strangers|"
                     r"couple|lover|lovers|singer|rapper|artist|audience|family|friends?)\b", re.I)


def people_in(prompt):
    return sorted({m.group(0).lower() for m in _PEOPLE.finditer(prompt or "")})


def semantic_errors(treatment, sections):
    """Rules that need the timeline. `sections` = the timeline's section dicts (index, role, is_hook, ...)."""
    errs = []
    got = treatment.get("sections", [])
    want = [s["index"] for s in sections]
    idx = [s.get("index") for s in got]
    if idx != want:
        errs.append(f"$.sections: indices must be exactly {want} in order (one entry per timeline section), "
                    f"got {idx}")
    hook = {s["index"] for s in sections if s.get("is_hook")}
    frozen = [s.get("index") for s in got if s.get("freeze") is True]
    for i, s in enumerate(got):
        if s.get("freeze") is True and s.get("index") not in hook:
            errs.append(f"$.sections[{i}].freeze: true on section {s.get('index')}, which is not a hook "
                        f"(hooks: {sorted(hook) or 'none'})")
        who = people_in(s.get("world_prompt", ""))
        if who:
            errs.append(f"$.sections[{i}].world_prompt: describes people ({', '.join(who)}); the filmed artist is the "
                        f"only person in frame, so the world must be empty of people")
    dollies = [s.get("index") for s in got if s.get("move") == "dolly"]
    if len(dollies) > 1:
        errs.append(f"$.sections: move dolly at most once in a song (it is the one moment the ground shifts); "
                    f"dolly now on sections {dollies}")
    for i, s in enumerate(got):
        if s.get("move") == "swap" and s.get("freeze") is True:
            errs.append(f"$.sections[{i}].move: swap on the frozen hook; the freeze holds one world, so a swap "
                        f"there cannot happen (pick none, dolly or crash)")
    on_hooks = [f for f in frozen if f in hook]
    if hook and len(on_hooks) != 1:
        errs.append(f"$.sections: freeze must be true on exactly one hook section (hooks: {sorted(hook)}); "
                    f"frozen hooks now: {on_hooks or 'none'}")
    return errs


def check_treatment(text_or_obj, sections):
    """-> (treatment or None, errors)."""
    obj, errs = (text_or_obj, []) if isinstance(text_or_obj, dict) else parse_json(text_or_obj)
    if errs:
        return None, errs
    errs = validate(obj, TREATMENT_SCHEMA)
    if isinstance(obj, dict) and isinstance(obj.get("sections"), list) and \
            all(isinstance(s, dict) for s in obj["sections"]):
        errs += semantic_errors(obj, sections)  # all at once, so one repair turn can fix everything
    return (obj if not errs else None), errs
