"""Lyrics per section: Muse Voice Transcribe on the master, and/or the artist's own lyric sheet.

Voice Transcribe (dev.meta.ai/docs/speech-to-text) has turn-level timestamps only, and a turn's startMs/endMs are
"audio processed when the model detected" the edge, not acoustic boundaries. So a turn is placed by OVERLAP with the
section windows (largest share wins, nearest section if it falls in a gap), never by its exact edges.
Sung vocals over a full mix may transcribe badly (the docs only describe speech). The artist's own text therefore
outranks the transcript: `--lyrics <file.txt>`, one block per section, blank line between blocks, aligned to the
timeline's sections in order. A block may start with `[#3]` to bind it to section 3; other bracket lines such as
`[Verse 1]` are labels.
"""
import math
import re
import subprocess
from pathlib import Path

from realone.common import Refuse

ASR_RATE = 24000          # the model's native rate (docs: speech-to-text)
CHUNK_S = 540             # under the 10-minute cap with room; 540 s of 24 kHz mono s16 is 25.9 MB, under 32 MB
EDGE_ROLES = ("intro", "outro")
_TAG = re.compile(r"^\[(.+)\]$")
_HOOK_TAGS = ("hook", "chorus", "refrain")


# ----------------------------------------------------------------------------------------------- artist's sheet

def parse_lyrics_text(text):
    """-> [{"tag": str|None, "index": int|None, "lines": [str]}], one per blank-line-separated block.
    A first line wholly in brackets is a tag: `[#3]` binds the block to section 3; anything else (`[Verse 1]`,
    `[Pre-Chorus]`, Genius style) is kept as a label only."""
    blocks, cur = [], []
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if line.startswith("# "):
            continue  # a comment line (a lyric never starts with "# ")
        if not line:
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append(cur)
    out = []
    for b in blocks:
        tag, index = None, None
        m = _TAG.match(b[0])
        if m:
            tag = m.group(1).strip().lower()
            mi = re.fullmatch(r"#\s*(\d+)", tag)
            index = int(mi.group(1)) if mi else None
            b = b[1:]
        if b:
            out.append({"tag": tag, "index": index, "lines": b})
    return out


def align_blocks(blocks, sections):
    """-> ({section index: [lines]}, [warnings]). Blocks tagged [#N] go to section N. The rest fill the remaining
    sections in order (the brief's rule: the sheet is in song order). When there are fewer of them than sections,
    leading intros and trailing outros are dropped first (usually instrumental). Any other mismatch refuses: the
    fix is a [#N] tag, never a guess. Label tags are checked against the analyzer's role, as a warning only
    (the analyzer's non-hook labels are cosmetic, docs/implementation-notes.md)."""
    by_index = {s["index"]: s for s in sections}
    out, warnings = {}, []
    for b in blocks:
        if b["index"] is not None:
            if b["index"] not in by_index or b["index"] in out:
                raise Refuse(f"lyric block tagged [#{b['index']}] has no free section #{b['index']} "
                             f"(sections: {', '.join('#%d %s' % (s['index'], s['role']) for s in sections)})")
            out[b["index"]] = b["lines"]
    rest = [b for b in blocks if b["index"] is None]
    cands = [s for s in sections if s["index"] not in out]
    while len(cands) > len(rest) and cands and cands[0]["role"] in EDGE_ROLES:
        cands.pop(0)
    while len(cands) > len(rest) and cands and cands[-1]["role"] in EDGE_ROLES:
        cands.pop()
    if len(cands) != len(rest):
        listing = ", ".join("#%d %s" % (s["index"], s["role"]) for s in sections)
        raise Refuse(f"the lyric sheet has {len(rest)} untagged blocks for {len(cands)} open sections "
                     f"({listing}); start each block with a [#N] line to place it without guessing")
    for s, b in zip(cands, rest):
        out[s["index"]] = b["lines"]
        if b["tag"] and any(h in b["tag"] for h in _HOOK_TAGS) != bool(s.get("is_hook")):
            warnings.append(f"block [{b['tag']}] landed on section #{s['index']} ({s['role']}); check the order "
                            f"or tag it [#N]")
    return out, warnings


def read_lyrics_file(path, sections):
    p = Path(path)
    if not p.exists():
        raise Refuse(f"lyrics file not found: {path}")
    blocks = parse_lyrics_text(p.read_text(encoding="utf-8"))
    if not blocks:
        raise Refuse(f"lyrics file {path} has no text")
    return align_blocks(blocks, sections)  # (lines by index, warnings)


# ----------------------------------------------------------------------------------------------- transcript

def map_turns(turns, sections, offset_ms=0):
    """Voice Transcribe turns -> {section index: [{"start_s", "end_s", "text"}]}, by overlap (see module doc)."""
    out = {s["index"]: [] for s in sections}
    for t in turns:
        text = (t.get("transcript") or "").strip()
        if not text:
            continue  # a turn the clip ended inside carries an empty transcript (docs: TranscribeResponse)
        a = (t["startMs"] + offset_ms) / 1000.0
        b = (t["endMs"] + offset_ms) / 1000.0
        if b < a:
            a, b = b, a
        best, best_ov = None, 0.0
        for s in sections:
            ov = min(b, s["end_s"]) - max(a, s["start_s"])
            if ov > best_ov + 1e-9:
                best, best_ov = s, ov
        if best is None:
            mid = (a + b) / 2
            best = min(sections, key=lambda s: 0.0 if s["start_s"] <= mid <= s["end_s"]
                       else min(abs(mid - s["start_s"]), abs(mid - s["end_s"])))
        out[best["index"]].append({"start_s": round(a, 3), "end_s": round(b, 3), "text": text,
                                   "overlap_s": round(max(best_ov, 0.0), 3)})
    return out


def asr_chunks(song, work_dir):
    """The master as 24 kHz mono 16-bit WAV, cut into <=540 s chunks -> [(path, offset_s)]."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                "default=nw=1:nk=1", str(song)], check=True, capture_output=True,
                               text=True).stdout.strip())
    out = []
    for k in range(max(1, math.ceil(dur / CHUNK_S))):
        p = work_dir / f"asr_{k:02d}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(k * CHUNK_S), "-t", str(CHUNK_S), "-i", str(song),
                        "-ac", "1", "-ar", str(ASR_RATE), "-c:a", "pcm_s16le", "-map_metadata", "-1", str(p)],
                       check=True)
        out.append((p, k * CHUNK_S))
    return out, dur


def transcribe_song(backend, song, work_dir, keywords=None):
    """-> {"transcript", "turns" (song-time ms), "audio_s", "source"}. One Voice Transcribe call per chunk."""
    chunks, dur = asr_chunks(song, work_dir)
    turns, texts, source = [], [], None
    for path, off in chunks:
        r = backend.transcribe(path, keywords=keywords)
        source = r.get("_source", source)
        texts.append(r.get("transcript", ""))
        for t in r.get("turns", []):
            turns.append({**t, "startMs": t["startMs"] + int(off * 1000), "endMs": t["endMs"] + int(off * 1000)})
    return {"transcript": " ".join(x for x in texts if x), "turns": turns, "audio_s": round(dur, 3),
            "source": source or "muse-voice-transcribe-1.0"}


def keywords_from(lines_by_index, limit=40):
    """Distinctive words from the artist's sheet, to bias the transcriber (docs: `keywords`)."""
    from realone.director.guard import STOPWORDS, tokens
    seen = []
    for lines in lines_by_index.values():
        for w in tokens(" ".join(lines)):
            if len(w) > 3 and w not in STOPWORDS and w not in seen:
                seen.append(w)
    return seen[:limit]


# ----------------------------------------------------------------------------------------------- merge

def build(sections, artist=None, transcript=None):
    """The lyrics record the treatment reads. The artist's text outranks the transcript, section by section."""
    heard = map_turns(transcript["turns"], sections) if transcript else {}
    rows = []
    for s in sections:
        i = s["index"]
        a = (artist or {}).get(i)
        h = heard.get(i, [])
        if a:
            lines, src = list(a), "artist"
        elif h:
            lines, src = [t["text"] for t in h], "transcript"
        else:
            lines, src = [], "none"
        rows.append({"index": i, "role": s["role"], "start_s": s["start_s"], "end_s": s["end_s"],
                     "lines": lines, "source": src, "heard": h})
    return {"sections": rows,
            "sources": {"artist_sheet": artist is not None,
                        "transcript": (transcript or {}).get("source") if transcript else None},
            "rule": "the artist's own lyric text outranks the transcript, per section"}


def lines_by_index(lyrics):
    return {r["index"]: r["lines"] for r in lyrics["sections"]}
