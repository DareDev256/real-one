"""Time every lyric word to the vocal, with two Meta open models: Demucs separates the vocal from the beat, then MMS
forced alignment (torchaudio MMS_FA) places the artist's own words on it.

Why (Nov 4, James's first real take): the render placed lyric words on the beat grid, one per beat. Song B's verse
sings about 110 words over 48 beats, so the words on screen ran out at bar 20 while he was still rapping to bar 24.
Voice Transcribe gives one time per turn, not per word. Forced alignment does not guess the words (they are the
artist's, from the treatment); it only finds when each one is sung.

Runs in its own environment (torch is not a render dependency):
    uv venv .venv-align --python 3.12 && VIRTUAL_ENV=.venv-align uv pip install torch torchaudio demucs soundfile
    .venv-align/bin/python scripts/align_lyrics.py --song SONG --treatment TREATMENT --out WORDS.json \
        [--check whisper_words.json]

The render reads the JSON (`render --word-times WORDS.json`). A star token sits between sections and at both ends,
so ad-libs that are not on the lyric sheet are absorbed instead of stretching a neighbouring word, and a word never
starts in a silent stretch of the vocal stem. Refuses (exit 2) when words land outside their own section or out of
order, instead of writing times nobody should trust."""
import argparse
import difflib
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

VOICED_DB = -30.0         # vocal stem level (dB re its own peak) above which a word can be sung
PICKUP_S = 0.5            # a line may start this early, before its section's first downbeat (a sung pickup)
MAX_BAD_WORDS = 0.05      # more than this share of words outside their section, or out of order = refuse
# Not a gate: the aligner's token score. On song B, lines scoring 0.001 to 0.03 (layered hook vocals) still sat
# 40 to 100 ms from whisper's times, so the score says nothing about timing (calibrated Nov 4). It is reported only.


def norm(w):
    return re.sub(r"[^a-z']", "", w.lower().replace("’", "'")).strip("'")


def words_of(line):  # the render's own split (realone/lyrictext.words_of)
    return [w for w in str(line).replace("’", "'").split() if w.strip()]


def vocal_stem(song, work):
    """Demucs htdemucs, two stems -> path of the vocal stem (wav)."""
    subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems", "vocals", "-o", str(work),
                    str(song)], check=True, capture_output=True)
    hits = list(Path(work).glob("htdemucs/*/vocals.wav"))
    if not hits:
        raise SystemExit("align_lyrics: REFUSED: demucs wrote no vocal stem")
    return hits[0]


def transcript_of(lines):
    """-> (tokens for the aligner, owner per token: (line index, word index, word) or None for a star)."""
    transcript, owner, prev = ["*"], [None], None
    for li, ln in enumerate(lines):
        if prev is not None and ln["section"] != prev:
            transcript.append("*"), owner.append(None)
        prev = ln["section"]
        for wi, w in enumerate(words_of(ln["text"])):
            t = norm(w)
            if t:
                transcript.append(t), owner.append((li, wi, w))
    transcript.append("*"), owner.append(None)
    return transcript, owner


def fill_unaligned(out):
    """A word with no letters (a lone dash) gets its neighbour's time, so word indices match the render's split."""
    for ln in out:
        ws = words_of(ln["text"])
        if len(ln["words"]) != len(ws):
            filled, k = [], 0
            for w in ws:
                if norm(w) and k < len(ln["words"]):
                    filled.append(ln["words"][k])
                    k += 1
                else:
                    ref = filled[-1] if filled else (ln["words"][0] if ln["words"] else {"start": 0.0, "end": 0.0})
                    filled.append({"w": w, "start": ref["start"], "end": ref["end"], "score": 0.0})
            ln["words"] = filled
        ln["score"] = round(sum(x["score"] for x in ln["words"]) / max(1, len(ln["words"])), 3)


def voiced_snap(out, mono, sr, hop_s=0.01, voiced_db=VOICED_DB):
    """CTC alignment lets a word's first token sit on silence (blank frames are free): song B's first hook line
    started at 3.82 s, in the gap after the intro, though it is sung at 6.86 s. A word starts on the first voiced
    frame of the vocal stem at or after its aligned start, never past its end. Mutates `out`; -> words moved."""
    import numpy as np
    hop = max(1, int(sr * hop_s))
    n = len(mono) // hop
    if n == 0:
        return 0
    rms = np.sqrt(np.mean(np.asarray(mono[: n * hop], float).reshape(n, hop) ** 2, axis=1)) + 1e-12
    voiced = 20 * np.log10(rms / rms.max()) > voiced_db
    moved = 0
    for ln in out:
        for x in ln["words"]:
            i0 = min(int(x["start"] / hop_s), n - 1)
            i1 = min(n, max(i0 + 1, int(x["end"] / hop_s)))
            if voiced[i0]:
                continue
            on = np.nonzero(voiced[i0:i1])[0]
            if on.size:
                x["snapped_from"] = x["start"]
                x["start"] = round((i0 + int(on[0])) * hop_s, 3)
                moved += 1
    return moved


def structure_errors(out, sections, pickup_s=PICKUP_S):
    """Words outside their own section (allowing a sung pickup) or out of order -> list of problems."""
    bounds = {s["index"]: (float(s["start_s"]), float(s["end_s"])) for s in sections}
    bad, last = [], -1.0
    for ln in out:
        a, b = bounds[ln["section"]]
        for x in ln["words"]:
            if not (a - pickup_s <= x["start"] <= b):
                bad.append(f"'{x['w']}' (section {ln['section']}, line {ln['line']}) at {x['start']:.2f}s is outside "
                           f"its section {a:.2f}-{b:.2f}s")
            elif x["start"] + 1e-3 < last:
                bad.append(f"'{x['w']}' (section {ln['section']}, line {ln['line']}) at {x['start']:.2f}s starts "
                           f"before the word sung ahead of it ({last:.2f}s)")
            last = max(last, x["start"])
    return bad


def align(vocal_wav, lines):
    """lines: [{"section", "line", "text"}] in sung order -> (the same with "words": [{w, start, end, score}],
    words moved out of silence)."""
    import torch
    import torchaudio
    import soundfile as sf
    bundle = torchaudio.pipelines.MMS_FA
    y, sr = sf.read(str(vocal_wav), dtype="float32", always_2d=True)
    mono = y.mean(1)
    wav = torchaudio.functional.resample(torch.from_numpy(mono).unsqueeze(0), sr, bundle.sample_rate)
    wav = wav / (wav.abs().max() + 1e-9)
    model, tokenizer, aligner = bundle.get_model(), bundle.get_tokenizer(), bundle.get_aligner()
    with torch.inference_mode():
        emission, _ = model(wav)
    sec_per_frame = wav.size(1) / emission.size(1) / bundle.sample_rate
    transcript, owner = transcript_of(lines)
    spans = aligner(emission[0], tokenizer(transcript))
    out = [dict(ln, words=[]) for ln in lines]
    for sp, who in zip(spans, owner):
        if who is None:
            continue
        li, _, w = who
        frames = sum(s.end - s.start for s in sp)
        score = float(sum(s.score * (s.end - s.start) for s in sp) / max(1, frames))
        out[li]["words"].append({"w": w, "start": round(sp[0].start * sec_per_frame, 3),
                                 "end": round(sp[-1].end * sec_per_frame, 3), "score": round(score, 3)})
    moved = voiced_snap(out, mono, sr)
    fill_unaligned(out)
    return out, moved


def check_against(out, ref_path):
    """Independent instrument: match the aligned words to another transcriber's word times by sequence alignment
    -> {matched, median_ms, p90_ms}."""
    ref = json.loads(Path(ref_path).read_text())["words"]
    a = [(norm(x["w"]), x["start"]) for ln in out for x in ln["words"]]
    b = [(norm(x["w"]), x["start"]) for x in ref]
    sm = difflib.SequenceMatcher(a=[w for w, _ in a], b=[w for w, _ in b], autojunk=False)
    d = sorted(abs(a[i + k][1] - b[j + k][1]) * 1000 for i, j, n in sm.get_matching_blocks() for k in range(n))
    if not d:
        return {"against": str(ref_path), "matched": 0}
    return {"against": str(ref_path), "matched": len(d), "of": len(a),
            "median_ms": round(d[len(d) // 2], 1), "p90_ms": round(d[int(len(d) * 0.9)], 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--song", required=True)
    ap.add_argument("--treatment", required=True, help="the director's treatment: its sections' lyric lines")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", help="another transcriber's word times JSON ({words: [{w, start}]}) to compare")
    ap.add_argument("--report-only", action="store_true", help="write the file even when it would refuse")
    a = ap.parse_args(argv)
    t = json.loads(Path(a.treatment).read_text())
    lines = [{"section": s["index"], "line": j, "text": x}
             for s in sorted(t["sections"], key=lambda s: s["index"]) for j, x in enumerate(s.get("lyric") or [])]
    if not lines:
        raise SystemExit("align_lyrics: REFUSED: the treatment carries no lyric lines")
    with tempfile.TemporaryDirectory() as work:
        out, moved = align(vocal_stem(a.song, work), lines)
    bad = structure_errors(out, t["sections"])
    n_words = sum(len(ln["words"]) for ln in out)
    rec = {"source": "Demucs htdemucs vocal stem + torchaudio MMS_FA forced alignment (Meta, open weights)",
           "song": a.song, "song_sha256": hashlib.sha256(Path(a.song).read_bytes()).hexdigest(),
           "treatment": a.treatment, "lines": out, "snapped_out_of_silence": moved, "structure_errors": bad,
           "thresholds": {"voiced_db": VOICED_DB, "pickup_s": PICKUP_S, "max_bad_words": MAX_BAD_WORDS}}
    if a.check:
        rec["check"] = check_against(out, a.check)
    print(f"aligned {n_words} words in {len(out)} lines; {moved} moved out of silence; {len(bad)} structure "
          f"error(s); check: {rec.get('check')}")
    for e in bad[:5]:
        print("  " + e)
    if len(bad) > MAX_BAD_WORDS * n_words and not a.report_only:
        raise SystemExit(f"align_lyrics: REFUSED: {len(bad)} of {n_words} words land outside their section or out "
                         f"of order (is the lyric sheet the sung words, and are the section marks right?)")
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
