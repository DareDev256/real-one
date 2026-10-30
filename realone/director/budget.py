"""Spend: quote before any keyed call, meter during, refuse rather than overspend.

Prices are the documented ones (dev.meta.ai/docs/pricing-rate-limits, read 2026-10-29). The quote has two numbers:
- ceiling: every planned call at its `max_output_tokens` (which counts reasoning tokens, so it is a real bound on
  output), every conditional call (repair, guard re-ask) included, input uncached. The cap is checked against this.
- expected: the calls that normally happen, at typical output sizes, half of the loop's history from cache.
Input tokens for audio and video are ASSUMPTIONS (the docs do not publish a per-second rate, docs/token-counting);
they are marked as such in every quote and the meter reads real `usage` from each response instead.

The meter is the hard stop: before each paid call it checks calls-so-far against `max_calls` and
spent-so-far + this call's ceiling against the cap, and refuses (nothing called) if either would be passed.
"""
import math
import os

from realone.common import Refuse
from realone.meta_api import IMAGE_USD, SAM_MIN_FRAMES, SAM_USD_PER_1K_FRAMES, USD_TO_CAD

WONDER_MODEL = "muse-wonder-1.0"
VOICE_MODEL = "muse-voice-transcribe-1.0"

WONDER_IN_USD = 4.00 / 1e6
WONDER_CACHED_USD = 0.20 / 1e6
WONDER_OUT_USD = 20.00 / 1e6
VOICE_USD_PER_S = 0.18 / 3600

# Muse Spark Open on Baseten (baseten.co/library/muse-spark-open, read 2026-10-30). The hackathon's open-weight bonus
# names `meta-models/muse-spark-open-1026`; Baseten's own sample uses `meta-models/muse-spark-open`. The first keyed
# call settles which one answers; --spark-model switches without a code change.
SPARK_MODEL = "meta-models/muse-spark-open-1026"
SPARK_BASE_URL = "https://inference.baseten.co/v1"
SPARK_IN_USD = 1.25 / 1e6
SPARK_CACHED_USD = 0.15 / 1e6
SPARK_OUT_USD = 4.25 / 1e6

# ASSUMPTIONS (UNCONFIRMED until the first keyed run reads usage; see docs/meta-api-director.md checklist 5)
AUDIO_TOKENS_PER_S = 50     # song audio as input_audio
VIDEO_TOKENS_PER_S = 300    # a render as input_video: sampled frames plus its embedded audio
CHARS_PER_TOKEN = 3.3       # English prose and JSON
INSTRUCTION_TOKENS = 1500   # the director brief sent as `instructions`
FX_NOTE = f"US$1 = CA${USD_TO_CAD:.2f} (rough, fixed; realone/meta_api.py)"

# Output ceilings per call kind (max_output_tokens sent with the call) and typical output sizes.
MAX_OUT = {"treatment": 12000, "treatment_repair": 12000, "guard_reask": 4000, "loop": 3000, "watch": 3000,
           "decide": 3000}
TYPICAL_OUT = {"treatment": 6000, "treatment_repair": 6000, "guard_reask": 2000, "loop": 1200, "watch": 1500,
               "decide": 1500}
LOOP_BASE_TOKENS = 5000     # the loop's opening context (treatment, coverage, rules, tool specs)
LOOP_GROWTH_TOKENS = 1500   # history added per turn (tool outputs, model text)

DEFAULT_CAP_USD = 2.00
DEFAULT_MAX_CALLS = 40
MAX_REVISIONS = 3           # changes allowed in the one revision pass


def text_tokens(chars):
    return int(math.ceil(chars / CHARS_PER_TOKEN))


def wonder_usd(input_tokens, output_tokens, cached_tokens=0):
    fresh = max(0, input_tokens - cached_tokens)
    return fresh * WONDER_IN_USD + cached_tokens * WONDER_CACHED_USD + output_tokens * WONDER_OUT_USD


def wonder_ceiling(kind, input_tokens):
    return wonder_usd(input_tokens, MAX_OUT[kind])


def spark_usd(input_tokens, output_tokens, cached_tokens=0):
    fresh = max(0, input_tokens - cached_tokens)
    return fresh * SPARK_IN_USD + cached_tokens * SPARK_CACHED_USD + output_tokens * SPARK_OUT_USD


BRAIN_USD = {"wonder": wonder_usd, "spark-open": spark_usd}
BRAIN_NAME = {"wonder": "Wonder", "spark-open": "Spark Open"}


def sam_usd(frames):
    return max(frames, SAM_MIN_FRAMES) * SAM_USD_PER_1K_FRAMES / 1000


def quote(*, song_s, n_sections, brief_chars, audio_in=True, transcribe=True, direct=False, covered_s=(),
          max_turns=8, driver="model", sam_s=None, brain="wonder"):
    """-> {"lines": [...], "ceiling_usd", "expected_usd", ...}. covered_s: seconds of each section the take covers
    (what `direct` will render, mask and watch). sam_s: seconds that still need SAM (default: all of covered_s;
    masks already cached by the renderer cost nothing). brain: which model writes the treatment (plan only)."""
    if brain != "wonder" and direct:
        raise Refuse(f"--brain {brain} drafts the treatment only (plan); direct stays on Muse Wonder")
    brain_usd, name = BRAIN_USD[brain], BRAIN_NAME[brain]
    lines = []

    def add(what, ceiling, expected, n=1):
        lines.append({"what": what, "calls": n, "ceiling_usd": round(ceiling, 4), "expected_usd": round(expected, 4)})

    if transcribe:
        chunks = max(1, math.ceil(song_s / 540))
        usd = math.floor(song_s) * VOICE_USD_PER_S
        add(f"Voice Transcribe, {song_s:.0f} s of audio", usd, usd, chunks)
    t_in = INSTRUCTION_TOKENS + text_tokens(brief_chars) + (int(song_s * AUDIO_TOKENS_PER_S) if audio_in else 0)
    add(f"{name} treatment" + (" (+ song audio, ASSUMED tokens/s)" if audio_in else ""),
        brain_usd(t_in, MAX_OUT["treatment"]), brain_usd(t_in, TYPICAL_OUT["treatment"]))
    add(f"{name} repair turn (only if the JSON fails validation)", brain_usd(t_in + 1000, MAX_OUT["treatment_repair"]), 0.0)
    add(f"{name} guard re-ask (only if a world_prompt is literal)",
        brain_usd(t_in + MAX_OUT["treatment"] + 800, MAX_OUT["guard_reask"]), 0.0)
    if direct:
        n_cov = len(covered_s)
        if driver == "model":
            ceil_in = sum(LOOP_BASE_TOKENS + k * LOOP_GROWTH_TOKENS for k in range(max_turns))
            exp_turns = min(max_turns, 4 + 2 * min(n_cov, 2))
            exp_in = sum(LOOP_BASE_TOKENS + k * LOOP_GROWTH_TOKENS for k in range(exp_turns))
            add(f"Wonder director loop, up to {max_turns} turns",
                ceil_in * WONDER_IN_USD + max_turns * MAX_OUT["loop"] * WONDER_OUT_USD,
                wonder_usd(exp_in, exp_turns * TYPICAL_OUT["loop"], cached_tokens=exp_in // 2), max_turns)
        else:
            d_in = LOOP_BASE_TOKENS + n_cov * 800
            add("Wonder revision decision (code driver)", wonder_ceiling("decide", d_in),
                wonder_usd(d_in, TYPICAL_OUT["decide"]))
        renders = list(covered_s)
        revised = sorted(covered_s, reverse=True)[:min(MAX_REVISIONS, n_cov)]
        w_ceil = sum(wonder_ceiling("watch", 2000 + int(s * VIDEO_TOKENS_PER_S)) for s in renders + revised)
        w_exp = sum(wonder_usd(2000 + int(s * VIDEO_TOKENS_PER_S), TYPICAL_OUT["watch"]) for s in renders) + \
            (wonder_usd(2000 + int(revised[0] * VIDEO_TOKENS_PER_S), TYPICAL_OUT["watch"]) if revised else 0.0)
        add("Wonder watches each render (+ video, ASSUMED tokens/s)", w_ceil, w_exp, len(renders) + len(revised))
        img_n = n_sections + MAX_REVISIONS
        add(f"Muse Image worlds ({n_sections} + up to {MAX_REVISIONS} revised)", img_n * IMAGE_USD,
            n_sections * IMAGE_USD + IMAGE_USD, img_n)
        fresh_s = sum(covered_s) if sam_s is None else sam_s
        sam = sum(sam_usd(int(round(s * 30))) for s in covered_s) if sam_s is None else sam_usd(int(round(sam_s * 30)))
        add(f"SAM 3.1 on {fresh_s:.0f} s of the take (cached after the first render)", sam if fresh_s else 0.0,
            sam if fresh_s else 0.0, n_cov if fresh_s else 0)
    ceiling = sum(x["ceiling_usd"] for x in lines)
    expected = sum(x["expected_usd"] for x in lines)
    return {"lines": lines, "ceiling_usd": round(ceiling, 4), "expected_usd": round(expected, 4),
            "ceiling_cad": round(ceiling * USD_TO_CAD, 2), "expected_cad": round(expected * USD_TO_CAD, 2),
            "assumptions": [f"audio input {AUDIO_TOKENS_PER_S} tokens/s (ASSUMED)",
                            f"video input {VIDEO_TOKENS_PER_S} tokens/s (ASSUMED)",
                            f"{CHARS_PER_TOKEN} characters per text token (ASSUMED)", FX_NOTE,
                            "ceiling = every call at max_output_tokens, input uncached; "
                            "expected = typical outputs, half the loop history cached (ASSUMED)"]}


def format_quote(q, header):
    out = [header]
    for x in q["lines"]:
        out.append(f"  {x['what']:<66s} x{x['calls']:<3d} ceiling US${x['ceiling_usd']:.3f}  "
                   f"expected US${x['expected_usd']:.3f}")
    out.append(f"  TOTAL ceiling US${q['ceiling_usd']:.2f} (about CA${q['ceiling_cad']:.2f}), expected "
               f"US${q['expected_usd']:.2f} (about CA${q['expected_cad']:.2f})")
    out.append("  " + "; ".join(q["assumptions"]))
    return "\n".join(out)


def gate(q, cap_usd, yes_spend):
    """Refuse a keyed run whose ceiling is over the cap, unless the person said yes to the spend."""
    if q["ceiling_usd"] > cap_usd and not yes_spend:
        raise Refuse(f"the spend ceiling US${q['ceiling_usd']:.2f} (about CA${q['ceiling_cad']:.2f}) is over the "
                     f"cap US${cap_usd:.2f}; nothing was called. Re-run with --yes-spend to accept it, or raise "
                     f"--cap-usd, or narrow the run (--sections, --max-turns).")
    return max(cap_usd, q["ceiling_usd"]) if yes_spend else cap_usd


def require_key():
    """The director's key gate. Same rule as the renderer: refuse before any work, never print the key."""
    key = os.environ.get("MODEL_API_KEY") or None
    if not key:
        raise Refuse("MODEL_API_KEY is not set, and this run needs the Meta Model API (Muse Wonder, Voice "
                     "Transcribe, Muse Image, SAM 3.1). Nothing was called. Offline run: --dry-run (FIXTURE "
                     "responses, never real model output).")
    return key


def require_baseten_key():
    """Spark Open's key gate: BASETEN_API_KEY, checked before any work, never printed."""
    key = os.environ.get("BASETEN_API_KEY") or None
    if not key:
        raise Refuse("BASETEN_API_KEY is not set, and --brain spark-open runs Muse Spark Open on Baseten. Nothing "
                     "was called. The key comes with the Baseten hackathon credits; it lives outside the repo.")
    return key


class Meter:
    """Counts paid calls and dollars. `before()` refuses BEFORE a call that would pass a limit."""

    def __init__(self, cap_usd=DEFAULT_CAP_USD, max_calls=DEFAULT_MAX_CALLS):
        self.cap_usd = cap_usd
        self.max_calls = max_calls
        self.calls = 0
        self.spent = 0.0
        self.ledger = []

    def remaining(self):
        return max(0.0, self.cap_usd - self.spent)

    def before(self, kind, ceiling_usd):
        if self.calls >= self.max_calls:
            raise Refuse(f"call cap reached ({self.max_calls} paid calls) before {kind}; stopped, nothing more "
                         f"called. Spent US${self.spent:.4f}. Raise --max-calls to allow more.")
        if self.spent + ceiling_usd > self.cap_usd + 1e-9:
            raise Refuse(f"{kind} could cost up to US${ceiling_usd:.4f}; with US${self.spent:.4f} spent that passes "
                         f"the cap US${self.cap_usd:.2f}. Stopped before calling.")
        self.calls += 1

    def charge(self, kind, usd, detail=None):
        self.spent += usd
        self.ledger.append({"kind": kind, "usd": round(usd, 6), **(detail or {})})

    def summary(self):
        return {"calls": self.calls, "spent_usd": round(self.spent, 4),
                "spent_cad": round(self.spent * USD_TO_CAD, 2), "cap_usd": self.cap_usd,
                "max_calls": self.max_calls, "ledger": self.ledger}
