"""The director's tools: what Muse Wonder (or the code driver) can do, and the rules code enforces on it.

    generate_world(section, prompt)          Muse Image, after the literal-lyrics guard
    render_section(section, look, freeze)    `python -m realone render ...` by subprocess
    watch(render_path)                       Files API upload + Wonder with video in -> timestamped notes
    revise(section, field, value, because)   the ONE revision pass: world_prompt, look, or move the freeze
    finish(summary)

Rules (enforced here, not asked for; `max_tool_calls` does not cap client tools, docs/tool-calling):
- the treatment is the single source of truth: generate_world and render_section must use the section's current
  world_prompt, look and freeze; the only way to change them is `revise`, so the one-pass limit cannot be dodged;
- one revision pass, at most MAX_REVISIONS changes, all made before the first re-render; then it is closed;
- a section renders at most twice (first cut + its revision), only if the take covers it;
- watch only accepts a render this run produced (no arbitrary file goes to the Files API);
- a tool error is returned to the model as {"ok": false, "error": ...}; a budget or key refusal ends the run.
"""
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from realone.common import FPS, LOCAL_PREFIX, OUT, REPO, Refuse
from realone.director import guard
from realone.director.backend import text_part, user
from realone.director.budget import MAX_REVISIONS, VIDEO_TOKENS_PER_S
from realone.director.schema import (LOOKS, NOTES_SCHEMA, SECTION_SCHEMA, parse_json, people_in, validate)

PARTICLES = ("dust", "rain", "embers", "none")  # the renderer's --particles overlay
DEFAULT_PARTICLES = "dust"

MAX_RENDERS_PER_SECTION = 2


class ToolError(Exception):
    """A refusal the model can act on (returned as a tool result, not raised to the CLI)."""


def _fn(name, description, props, required):
    return {"type": "function", "name": name, "description": description, "strict": True,
            "parameters": {"type": "object", "additionalProperties": False, "required": required,
                           "properties": props}}


TOOL_SPECS = [
    _fn("generate_world",
        "Build the world for one section with Muse Image. `prompt` must be the treatment's current world_prompt for "
        "that section (change it with revise). The literal-lyrics guard checks it against the section's lyric first "
        "and refuses a literal prompt.",
        {"section": {"type": "integer", "description": "section index from the timeline"},
         "prompt": {"type": "string", "description": "the section's world_prompt"}},
        ["section", "prompt"]),
    _fn("render_section",
        "Render one section with the real renderer: the artist cut out of the phone take (real pixels) over the "
        "section's world, cuts on downbeats, master audio. Only sections the take covers can render. Look and freeze "
        "must match the treatment. Takes minutes.",
        {"section": {"type": "integer"}, "look": {"type": "string", "enum": list(LOOKS)},
         "freeze": {"type": "boolean"}},
        ["section", "look", "freeze"]),
    _fn("watch",
        "Watch a finished render against the treatment (the MP4 is uploaded and you see it). Returns timestamped "
        "notes, a keep/revise verdict, one suggestion, and the render's own measured checks.",
        {"render_path": {"type": "string", "description": "the `video` path render_section returned"}},
        ["render_path"]),
    _fn("revise",
        f"One change in the single revision pass: swap a section's world_prompt, change its look, change its particle "
        f"overlay ({'/'.join(PARTICLES)}; the overlay is ours, drawn over the world, never in the world image), or move "
        f"the freeze to another hook (value 'true' on the hook that should freeze). Up to {MAX_REVISIONS} changes, all "
        f"before the first re-render; the pass closes when a re-render starts.",
        {"section": {"type": "integer"},
         "field": {"type": "string", "enum": ["world_prompt", "look", "particles", "freeze"]},
         "value": {"type": "string"}, "because": {"type": "string", "description": "the note this answers"}},
        ["section", "field", "value", "because"]),
    _fn("finish", "End the run with a short summary of what was rendered and what was changed.",
        {"summary": {"type": "string"}}, ["summary"]),
]
SPECS = {t["name"]: t for t in TOOL_SPECS}

WATCH_INSTRUCTIONS = """You are the director watching your own cut. The video attached is one rendered section of the
music video: a filmed artist (real pixels, cut out by SAM 3.1) over a generated world. Judge it against the treatment
below, not against taste in general. Give timestamped notes (seconds from the start of THIS clip): where the world,
the look, the camera or the freeze fights the section's subtext, and where it lands. Then a verdict (keep or revise)
and at most one suggestion: a new world_prompt (a place, no people, none of the lyric's words), a different look,
a different particle overlay, or moving the freeze. The particles (dust, rain, embers) are an overlay the renderer
draws over the world; they are not in the world image, so a world_prompt cannot remove them: change them with the
particles field. render.move says which camera move ran, when, and what it does on screen; judge whether it reads at
that time, not whether a different move would. The measured checks are facts from the renderer; do not contradict
them."""


SCREEN = {"dolly": "dolly zoom: the virtual camera pulls back while the lens zooms in, so the artist keeps his size and "
                   "the world behind him grows (about {mag:.1f}x for a far wall) and its perspective flattens",
          "crash": "crash zoom: a fast snap in toward his face on each downbeat, with a zoom blur",
          "swap": "world swap: a new world on every bar, his feet on the same floor"}


def move_facts(run_json):
    """-> what the section's camera move did on screen, in this clip's seconds, from the renderer's own shot plan
    (the `_shots.json` beside the run report), or None when no move ran. The dolly's magnification is computed from
    the solved subject distance and the planned pull-back: (z + dz) / z."""
    sp = Path(str(run_json).replace("_run.json", "_shots.json"))
    if not sp.exists():
        return None
    plan = json.loads(sp.read_text())
    fps = float(plan.get("fps") or FPS)
    out = []
    for m in plan.get("moves") or []:
        f = {"kind": m["kind"], "start_s": round(m["start_frame"] / fps, 2), "end_s": round(m["end_frame"] / fps, 2)}
        mag = 1.0
        if m["kind"] == "dolly":
            d = (plan.get("stage") or {}).get("dolly") or {}
            z = sum(d.get("subject_m") or [0]) / max(1, len(d.get("subject_m") or []))
            if z > 0 and d.get("max_m"):
                mag = (z + d["max_m"]) / z
                f["background_magnification"] = round(mag, 2)
            f["eases_until_s"] = f["end_s"]
        f["on_screen"] = SCREEN.get(m["kind"], m["kind"]).format(mag=mag)
        out.append(f)
    return out or None


# --------------------------------------------------------------------------------------------- renderer adapter

def detect_freeze(help_text):
    """-> (flag or None, note). The director passes the flag bare, so it uses a plain on/off flag or one whose value
    is optional (`--freeze [SECTION]`, bracketed in argparse help: bare = the rendered section). A flag that
    REQUIRES a value is not guessed at."""
    opts = re.findall(r"(--[\w-]*freeze[\w-]*)(?:[ =]([A-Z][A-Z_]*))?", help_text or "")
    if not opts:
        return None, "the renderer has no freeze option (realone 0.1.0); freeze is planned but not rendered"
    plain = [o for o, meta in opts if not meta]
    if "--freeze" in plain:
        return "--freeze", "renderer flag --freeze"
    if plain:
        return plain[0], f"renderer flag {plain[0]}"
    return None, (f"the renderer's {opts[0][0]} takes a value ({opts[0][1]}); the director does not guess its "
                  f"meaning, so freeze is not applied")


class Renderer:
    """Builds and runs `python -m realone render` for one section. Uses only flags that exist in realone 0.1.0,
    plus a freeze flag found in `--help` at runtime."""

    def __init__(self, clip, song, *, dry_run, max_minutes=20.0, sync_near=None, runner=None, help_text=None,
                 stage=False, sections=None):
        self.clip, self.song = str(clip), str(song)
        self.sections = sections  # the artist's section marks: a render re-analyzes the song, so `#N` must mean the same N
        self.dry_run = dry_run
        self.stage = stage  # keyed renders through the tracked stage (needed for a dolly)
        self.max_minutes = max_minutes
        self.sync_near = sync_near
        self.runner = runner or self._run
        self._help = help_text

    def help_text(self):
        if self._help is None:
            p = subprocess.run([sys.executable, "-m", "realone", "render", "--help"], cwd=REPO,
                               capture_output=True, text=True, timeout=60)
            self._help = p.stdout
        return self._help

    def argv(self, section, look, freeze, world_prompt, tag, max_usd, move="none", particles=DEFAULT_PARTICLES):
        a = [sys.executable, "-m", "realone", "render", "--clip", self.clip, "--song", self.song,
             "--section", f"#{section}", "--look", look, "--tag", tag, "--max-minutes", f"{self.max_minutes:g}"]
        if self.sync_near is not None:
            a += ["--sync-near", f"{self.sync_near:g}"]
        if self.sections:
            a += ["--sections", str(self.sections)]
        if self.dry_run:
            a += ["--mask", "local", "--local-plate"]
        else:
            a += ["--mask", "sam", "--world", world_prompt, "--max-usd", f"{max_usd:.4f}"]
            if self.stage:
                a += ["--stage", "tracked"]
        if particles != DEFAULT_PARTICLES and "--particles" in self.help_text():
            a += ["--particles", particles]
        flag, note = detect_freeze(self.help_text())
        if freeze and flag:
            a.append(flag)
        self.move_note = "no move"
        if move and move != "none":
            if "--move" not in self.help_text():
                self.move_note = f"{move}: this renderer has no --move"
            elif move == "dolly" and (self.dry_run or not self.stage):
                self.move_note = "dolly skipped: it needs the tracked stage (a keyed render with --stage)"
            else:
                a += ["--move", f"{move}@#{section}"]
                self.move_note = move
        return a, (note if freeze else "no freeze on this section")

    def _run(self, argv):
        p = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=self.max_minutes * 60 + 300)
        return p.returncode, (p.stdout[-3000:] + "\n" + p.stderr[-2000:])

    def run_json(self, tag, section):
        base = (LOCAL_PREFIX if self.dry_run else "") + tag
        return OUT / f"{base}_s{section}_run.json"


# --------------------------------------------------------------------------------------------- state + tools

@dataclass
class State:
    timeline: dict
    lyrics: dict              # {index: [lines]}
    treatment: dict           # {"song_subtext", "sections": [...]} (mutated by revise)
    covered: list             # section indices the take covers
    tag: str
    dry_run: bool
    worlds: dict = field(default_factory=dict)
    renders: dict = field(default_factory=dict)
    watches: list = field(default_factory=list)
    revision: dict = field(default_factory=lambda: {"state": "unused", "changes": [], "changed": [],
                                                    "rerendered": []})
    finished: str = None
    log: list = field(default_factory=list)

    def section(self, i):
        s = next((s for s in self.timeline["sections"] if s["index"] == i), None)
        if s is None:
            raise ToolError(f"there is no section {i} (sections 0-{len(self.timeline['sections']) - 1})")
        return s

    def plan(self, i):
        return next(s for s in self.treatment["sections"] if s["index"] == i)


def _norm(s):
    return " ".join((s or "").split())


class Tools:
    def __init__(self, backend, state, renderer, meter=None):
        self.backend = backend
        self.state = state
        self.renderer = renderer
        self.meter = meter

    # ---- dispatch
    def dispatch(self, name, arguments):
        spec = SPECS.get(name)
        if spec is None:
            out = {"ok": False, "error": f"unknown tool '{name}'; tools: {', '.join(SPECS)}"}
        else:
            args, errs = (arguments, []) if isinstance(arguments, dict) else parse_json(arguments)
            errs = errs or validate(args, spec["parameters"])
            if errs:
                out = {"ok": False, "error": "bad arguments: " + "; ".join(errs)}
            else:
                try:
                    out = getattr(self, f"t_{name}")(**args)
                except ToolError as e:
                    out = {"ok": False, "error": str(e)}
        self.state.log.append({"tool": name, "args": arguments if isinstance(arguments, dict) else
                               (parse_json(arguments)[0] or arguments), "result": out})
        return out

    # ---- generate_world
    def t_generate_world(self, section, prompt):
        st = self.state
        st.section(section)
        plan = st.plan(section)
        if _norm(prompt) != _norm(plan["world_prompt"]):
            raise ToolError(f"that prompt is not section {section}'s world_prompt in the treatment; change it with "
                            f"revise (field world_prompt) first")
        rep = guard.check(prompt, "\n".join(st.lyrics.get(section, [])))
        if rep["verdict"] == "reject":
            raise ToolError(f"the literal-lyrics guard rejects this prompt for section {section} (score "
                            f"{rep['score']:.2f}, limit {rep['threshold']:.2f}; words: "
                            f"{', '.join(guard.offending_words(rep))})")
        path, note = self.backend.image(prompt)
        st.worlds[section] = {"prompt": _norm(prompt), "path": str(path) if path else None, "note": note}
        return {"ok": True, "section": section, "world": str(path) if path else None, "note": note,
                "guard": {"verdict": rep["verdict"], "score": rep["score"]}}

    # ---- render_section
    def t_render_section(self, section, look, freeze):
        st = self.state
        sec = st.section(section)
        plan = st.plan(section)
        if section not in st.covered:
            raise ToolError(f"the phone take does not cover section {section}; it covers {st.covered or 'none'}")
        if look != plan["look"] or freeze != plan["freeze"]:
            raise ToolError(f"section {section}'s treatment says look={plan['look']} freeze={plan['freeze']}; "
                            f"change it with revise first")
        done = st.renders.get(section, [])
        rev = st.revision
        if done:
            if section not in rev["changed"] or section in rev["rerendered"]:
                raise ToolError(f"section {section} is already rendered and nothing changed since; renders are "
                                f"capped at {MAX_RENDERS_PER_SECTION} per section (first cut + its revision)")
        if not st.dry_run:
            w = st.worlds.get(section)
            if not w or w["prompt"] != _norm(plan["world_prompt"]):
                raise ToolError(f"generate_world for section {section}'s current world_prompt first")
        if rev["state"] == "open":
            rev["state"] = "closed"  # the re-render starts: no more changes in this pass
        tag = f"{st.tag}_r{len(done)}"
        if self.meter and not st.dry_run:
            self.meter.before(f"render section {section} (SAM 3.1; the renderer quotes and caps its own spend)", 0.0)
        max_usd = self.meter.remaining() if self.meter else 0.0
        argv, freeze_note = self.renderer.argv(section, look, freeze, plan["world_prompt"], tag, max_usd,
                                               move=plan.get("move", "none"),
                                               particles=plan.get("particles", DEFAULT_PARTICLES))
        print(f"director: rendering section {section} ({sec['role']}, {look}{', freeze' if freeze else ''}) ...")
        rc, tail = self.renderer.runner(argv)
        if rc != 0:
            raise ToolError(f"the renderer refused or failed (exit {rc}): {tail.strip()[-600:]}")
        rj = self.renderer.run_json(tag, section)
        if not rj.exists():
            raise ToolError(f"the renderer exited 0 but wrote no run report at {rj}")
        run = json.loads(rj.read_text())
        rs = run.get("section", {})
        if abs(rs.get("start_s", -1) - sec["start_s"]) > 0.002 or abs(rs.get("end_s", -1) - sec["end_s"]) > 0.002:
            raise ToolError(f"the renderer's section #{section} ({rs.get('start_s')}-{rs.get('end_s')} s) is not the "
                            f"director's ({sec['start_s']}-{sec['end_s']} s); the analysis changed between them")
        lk = run["looks"][look]
        if self.meter and not st.dry_run:
            self.meter.charge("render:sam+image", float(run.get("billed_usd", 0.0)), {"section": section})
        rec = {"section": section, "video": lk["video"], "look": look, "freeze_requested": freeze,
               "freeze": freeze_note if freeze else "off", "move": getattr(self.renderer, "move_note", "no move"),
               "move_on_screen": move_facts(rj), "particles": plan.get("particles", DEFAULT_PARTICLES),
               "tag": tag, "run_json": str(rj),
               "contact_sheet": lk.get("contact_sheet"), "duration_s": round(sec["end_s"] - sec["start_s"], 3),
               "measured": {"spec_ok": lk.get("spec_ok"), "cuts_on_plan": lk.get("cuts_on_plan"),
                            "cuts_off_plan": lk.get("cuts_off_plan"),
                            "audio_vs_master_error_ms": lk.get("audio_vs_master_error_ms"),
                            "cuts_s": [round(c / FPS, 3) for c in lk.get("cuts_detected", [])],
                            "masks": run.get("engine"), "billed_usd": run.get("billed_usd")}}
        st.renders.setdefault(section, []).append(rec)
        if done:
            rev["rerendered"].append(section)
        return {"ok": True, **rec}

    # ---- watch
    def _find_render(self, render_path):
        st = self.state
        m = re.fullmatch(r"@render:(\d+)", render_path or "")
        if m and self.backend.fixture:  # FIXTURE scripts cannot know paths in advance: latest render of section
            recs = st.renders.get(int(m.group(1)))
            if recs:
                return recs[-1]
        want = Path(render_path).resolve() if render_path else None
        for recs in st.renders.values():
            for r in recs:
                if want and Path(r["video"]).resolve() == want:
                    return r
        raise ToolError("watch only takes a video that render_section returned in this run")

    def t_watch(self, render_path):
        st = self.state
        rec = self._find_render(render_path)
        i = rec["section"]
        plan = st.plan(i)
        ctx = {"section": {k: st.section(i)[k] for k in ("index", "role", "is_hook", "start_s", "end_s")},
               "lyric": st.lyrics.get(i, []), "song_subtext": st.treatment.get("song_subtext"),
               "treatment": {k: plan[k] for k in ("subtext", "world_prompt", "palette", "camera_energy", "look",
                                                  "freeze", "why") if k in plan},
               "render": {"look": rec["look"], "freeze": rec["freeze"], "duration_s": rec["duration_s"],
                          "move": rec.get("move_on_screen") or {"kind": rec.get("move", "no move")},
                          "particles": rec.get("particles", DEFAULT_PARTICLES) + " (our overlay, not in the world)",
                          "measured": rec["measured"]}}
        if st.dry_run:
            ctx["render"]["note"] = ("TEST render: the world is a procedural test plate (--local-plate) and the "
                                     "figure is drawn; judge timing and look, not the world")
        fid = self.backend.upload(rec["video"], "render")
        turn = self.backend.respond(
            purpose="watch", instructions=WATCH_INSTRUCTIONS,
            input=[user(text_part("Treatment and measured facts (JSON):\n" + json.dumps(ctx, indent=1)),
                        {"type": "input_video", "file_id": fid})],
            schema=NOTES_SCHEMA, schema_name="render_notes", effort="medium",
            est_input_tokens=2000 + int(rec["duration_s"] * VIDEO_TOKENS_PER_S))
        notes, errs = parse_json(turn.text)
        errs = errs or validate(notes, NOTES_SCHEMA)
        if errs:
            raise ToolError("the watch notes did not validate: " + "; ".join(errs[:6]))
        w = {"section": i, "video": rec["video"], "source": turn.source, **notes, "measured": rec["measured"]}
        st.watches.append(w)
        return {"ok": True, **w}

    # ---- revise
    def t_revise(self, section, field, value, because):
        st = self.state
        sec = st.section(section)
        rev = st.revision
        if rev["state"] == "closed":
            raise ToolError("the one revision pass is spent (a re-render has started); no more changes")
        if len(rev["changes"]) >= MAX_REVISIONS:
            raise ToolError(f"the revision pass already holds {MAX_REVISIONS} changes")
        plan = st.plan(section)
        touched = [section]
        if field == "world_prompt":
            v = _norm(value)
            errs = validate(v, SECTION_SCHEMA["properties"]["world_prompt"], "world_prompt")
            if people_in(v):
                errs.append(f"describes people ({', '.join(people_in(v))}); the world is empty of people")
            rep = guard.check(v, "\n".join(st.lyrics.get(section, [])))
            if rep["verdict"] == "reject":
                errs.append(f"the literal-lyrics guard rejects it (score {rep['score']:.2f}; words: "
                            f"{', '.join(guard.offending_words(rep))})")
            if errs:
                raise ToolError("; ".join(errs))
            old, plan["world_prompt"] = plan["world_prompt"], v
            st.worlds.pop(section, None)
        elif field == "particles":
            v = value.strip().lower()
            if v not in PARTICLES:
                raise ToolError(f"particles must be one of {', '.join(PARTICLES)}")
            if v == plan.get("particles", DEFAULT_PARTICLES):
                raise ToolError(f"section {section} already uses {v}")
            old, plan["particles"] = plan.get("particles", DEFAULT_PARTICLES), v
        elif field == "look":
            if value not in LOOKS:
                raise ToolError(f"look must be one of {', '.join(LOOKS)}")
            if value == plan["look"]:
                raise ToolError(f"section {section} already uses {value}")
            old, plan["look"] = plan["look"], value
        else:
            if value.strip().lower() != "true":
                raise ToolError("move the freeze by setting value 'true' on the hook that should freeze")
            if not sec.get("is_hook"):
                raise ToolError(f"section {section} is not a hook; the freeze lives on a hook")
            if plan["freeze"]:
                raise ToolError(f"section {section} already holds the freeze")
            frm = [s["index"] for s in st.treatment["sections"] if s["freeze"]]
            for s in st.treatment["sections"]:
                s["freeze"] = s["index"] == section
            old = f"freeze on {frm}"
            touched += frm
        rev["state"] = "open"
        change = {"section": section, "field": field, "old": old, "new": plan[field] if field != "freeze" else True,
                  "because": because}
        rev["changes"].append(change)
        for t in touched:
            if t not in rev["changed"]:
                rev["changed"].append(t)
        rerender = [t for t in rev["changed"] if t in st.covered and t in st.renders]
        return {"ok": True, "applied": change, "rerender": rerender,
                "changes_left": MAX_REVISIONS - len(rev["changes"])}

    # ---- finish
    def t_finish(self, summary):
        self.state.finished = summary
        return {"ok": True}
