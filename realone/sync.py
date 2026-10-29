"""Align the phone clip's own mic audio to the master by onset-envelope cross-correlation.

The render always uses the master audio. The phone mic is only the clock: it tells us which song time each video
frame was filmed at. Offset convention: song_time = clip_time + offset (positive = the phone started recording
after the song started).

Six onset envelopes are correlated jointly: the percussive part (low < 200 Hz, mid 200-2000 Hz, high > 2 kHz, so a
half-bar shift that lines the kick up with the snare does not score) and the harmonic part (150-600, 600-1500,
1500-4000 Hz: note onsets of the vocal, the lead, the chords). Loop-based beats repeat the drums exactly; what
tells verse 1 from verse 2 is the notes, so they get their own bands instead of drowning under the kit. The result REFUSES (SyncRefused, exit 3) instead of guessing
when the best match is weak, or when a different lag matches almost as well and nothing in the clip tells them
apart (a clip that sits wholly inside a copy-pasted loop is genuinely ambiguous).

Repeats: a chorus that comes back sample-identical scores the same at both places over the shared part. The
tie-break looks at 3 s windows: the best lag is accepted only if, against EVERY competitor within COMPETE_R of
it, it wins some window clearly (the part around the repeat that differs) and loses none. `near_s` (CLI
--sync-near) restricts the search to +-NEAR_WINDOW_S around a known start.
"""
import numpy as np

from realone.common import Refuse
from realone.media import av_skew, load_audio, probe, stream

SR = 22050
HOP = 128                 # 5.8 ms per envelope frame; parabolic refinement goes finer
BANDS = ((0, 200), (200, 2000), (2000, 8000))          # percussive
HARM_BANDS = ((150, 600), (600, 1500), (1500, 4000))   # harmonic
R_MIN = 0.35              # Pearson r at the best lag below this = no real match
MARGIN_MIN = 0.08         # best r minus the best r at any other lag (beyond EXCLUDE_S) below this = ambiguous
EXCLUDE_S = 0.15
MIN_OVERLAP = 0.6         # a candidate lag must keep >= 60% of the clip inside the song
MIN_CLIP_S = 4.0
COMPETE_R = 0.12          # peaks within this of the best are competitors that must be beaten window by window
WIN_S = 3.0
WIN_CLEAR = 0.12          # the best lag must beat each competitor by this in at least one window
WIN_LOSS = 0.04           # ... and never lose a window by more than this
# 0.2.0, long takes (a whole song): sections that loop every 4 bars fit a 4-bar-shifted lag almost as well, so a
# 90 s take loses a handful of its ~59 windows to that shift by more than WIN_LOSS even at the true lag (measured:
# 2 to 9 windows, worst -0.092, while winning 20+ windows by 0.28 to 0.64). A take with WIN_LONG_N windows or more
# is judged on all of its evidence instead: it still needs a clear win, may lose at most WIN_LONG_LOST of its
# windows, the competitor may never clearly win one, and the mean advantage must stand WIN_LONG_T standard errors
# clear (windows overlap by half, so n/2 of them count). Shorter takes keep the 0.1.0 rule unchanged.
WIN_LONG_N = 24
WIN_LONG_LOST = 0.20
WIN_LONG_T = 2.5
NEAR_WINDOW_S = 6.0
SILENT_DBFS = -55.0


class SyncRefused(Refuse):
    exit_code = 3


def band_envelopes(y):
    import librosa
    mel = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_fft=1024, n_mels=64, fmax=8000)
    freqs = librosa.mel_frequencies(n_mels=64, fmax=8000)
    harm, perc = librosa.decompose.hpss(mel, kernel_size=(17, 31))
    envs = []
    for part, bands in ((perc, BANDS), (harm, HARM_BANDS)):
        db = librosa.power_to_db(part, ref=np.max)
        for lo, hi in bands:
            sel = (freqs >= lo) & (freqs < hi)
            envs.append(librosa.onset.onset_strength(S=db[sel], sr=SR, hop_length=HOP))
    return np.array(envs, np.float64)


def _ncc(m, c):
    """Pearson r between the clip envelopes `c` (B, L) and every window of the master `m` (B, N), with the master
    zero-padded by L on both sides so the clip may hang off either end. Returns (r per lag, overlap fraction)."""
    from scipy.signal import fftconvolve
    B, L = c.shape
    N = m.shape[1]
    mu, sd = m.mean(1, keepdims=True), m.std(1, keepdims=True) + 1e-9
    mz = np.pad((m - mu) / sd, ((0, 0), (L, L)))
    cz = (c - c.mean(1, keepdims=True)) / (c.std(1, keepdims=True) + 1e-9)
    num = sum(fftconvolve(mz[b], cz[b][::-1], mode="valid") for b in range(B))   # length N + L + 1
    cs1 = np.r_[0, np.cumsum(mz.sum(0))]
    cs2 = np.r_[0, np.cumsum((mz ** 2).sum(0))]
    k = np.arange(len(num))
    s1 = cs1[k + L] - cs1[k]
    s2 = cs2[k + L] - cs2[k]
    n = B * L
    var = np.maximum(s2 / n - (s1 / n) ** 2, 1e-12)
    r = num / (n * np.sqrt(var))
    start = k - L                                  # master frame the clip's first frame lands on
    overlap = (np.minimum(start + L, N) - np.maximum(start, 0)) / L
    return r, np.clip(overlap, 0, 1)


def _peaks(r, ex, floor):
    """Local maxima of r above `floor`, at least `ex` frames apart, best first."""
    order = np.argsort(-r)
    out = []
    for k in order:
        if r[k] < floor or len(out) >= 8:
            break
        if all(abs(k - j) > ex for j in out):
            out.append(int(k))
    return out


def _window_advantage(me, ce, lag_a, lag_b):
    """Per-window Pearson r at lag a minus at lag b. Master regions off the song's ends count as silence."""
    B, L = ce.shape
    N = me.shape[1]
    w = int(WIN_S * SR / HOP)
    mpad = np.pad(me, ((0, 0), (L, L)))
    d = []
    for s0 in range(0, max(1, L - w + 1), w // 2):
        c = ce[:, s0:s0 + w]
        cz = (c - c.mean(1, keepdims=True)) / (c.std(1, keepdims=True) + 1e-9)
        rs = []
        for lag in (lag_a, lag_b):
            m = mpad[:, lag + L + s0:lag + L + s0 + c.shape[1]]
            mz = (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-9)
            rs.append(float((cz * mz).mean()))
        d.append(rs[0] - rs[1])
    return np.array(d)


def _t(d):
    """Mean window advantage in standard errors; 3 s windows at a 1.5 s step overlap by half, so n/2 count."""
    return float(d.mean() / (d.std() / np.sqrt(max(1.0, len(d) / 2)) + 1e-9))


def _beaten(d):
    """Does the best lag beat this competitor? A clear win somewhere, always; then either the 0.1.0 rule (lose no
    window by more than WIN_LOSS) or, for a long take, the whole-take rule (see WIN_LONG_N)."""
    if d.max() < WIN_CLEAR:
        return False
    if d.min() >= -WIN_LOSS:
        return True
    return bool(len(d) >= WIN_LONG_N and (d < -WIN_LOSS).mean() <= WIN_LONG_LOST and d.min() > -WIN_CLEAR
                and _t(d) >= WIN_LONG_T)


def measure_offset(master, clip, near_s=None):
    """Arrays at SR -> dict(offset_s, r_peak, ...). Raises SyncRefused."""
    clip_s = len(clip) / SR
    if clip_s < MIN_CLIP_S:
        raise SyncRefused(f"the clip's audio is {clip_s:.1f}s; need at least {MIN_CLIP_S:.0f}s to sync")
    level = 20 * np.log10(np.sqrt(np.mean(clip ** 2)) + 1e-12)
    if level < SILENT_DBFS:
        raise SyncRefused(f"the clip's audio is silent ({level:.0f} dBFS); film with the song playing out loud")
    me, ce = band_envelopes(master), band_envelopes(clip)
    L = ce.shape[1]
    r, ov = _ncc(me, ce)
    r = np.where(ov >= MIN_OVERLAP, r, -1.0)
    lag_s = (np.arange(len(r)) - L) * HOP / SR
    if near_s is not None:
        r = np.where(np.abs(lag_s - near_s) <= NEAR_WINDOW_S, r, -1.0)
    kp = int(np.argmax(r))
    rp = float(r[kp])
    delta = 0.0
    if 0 < kp < len(r) - 1:
        a, b, c = r[kp - 1], r[kp], r[kp + 1]
        den = a - 2 * b + c
        if den < 0:
            delta = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    offset = (kp - L + delta) * HOP / SR
    ex = int(round(EXCLUDE_S * SR / HOP))
    far = r.copy()
    far[max(0, kp - ex):kp + ex + 1] = -1.0
    k2 = int(np.argmax(far))
    r2 = float(far[k2])
    res = {"offset_s": round(offset, 5), "offset_ms": round(offset * 1000, 1), "r_peak": round(rp, 4),
           "r_second": round(r2, 4), "margin": round(rp - r2, 4),
           "second_offset_s": round((k2 - L) * HOP / SR, 3), "clip_audio_s": round(clip_s, 3),
           "clip_level_dbfs": round(level, 1), "near_s": near_s, "resolved_by": "margin",
           "thresholds": {"r_min": R_MIN, "margin_min": MARGIN_MIN, "compete_r": COMPETE_R,
                          "win_clear": WIN_CLEAR, "win_loss": WIN_LOSS, "win_long_n": WIN_LONG_N,
                          "win_long_lost": WIN_LONG_LOST, "win_long_t": WIN_LONG_T}}
    if rp < R_MIN:
        raise SyncRefused(f"the clip's audio does not match the song (best r={rp:.2f} < {R_MIN}). Is this the "
                          f"right song, and was it playing out loud while filming?")
    if rp - r2 < MARGIN_MIN:
        comps = [k for k in _peaks(r, ex, rp - COMPETE_R) if abs(k - kp) > ex]
        verdicts = []
        for k in comps:
            d = _window_advantage(me, ce, kp - L, k - L)
            verdicts.append({"offset_s": round((k - L) * HOP / SR, 3), "r": round(float(r[k]), 4),
                             "best_window_win": round(float(d.max()), 3), "worst_window_loss": round(float(d.min()), 3),
                             "windows": len(d), "windows_lost": int((d < -WIN_LOSS).sum()),
                             "t": round(_t(d), 2), "beaten": _beaten(d)})
        res["competitors"] = verdicts
        if not all(v["beaten"] for v in verdicts):
            alt = next(v for v in verdicts if not v["beaten"])
            raise SyncRefused(f"ambiguous sync: r={rp:.2f} at {offset:+.3f}s but r={alt['r']:.2f} at "
                              f"{alt['offset_s']:+.3f}s, and no part of the clip tells them apart (it may sit inside "
                              f"a repeated loop). Use a take that crosses a section change, or pass --sync-near "
                              f"<song seconds where the take starts>.")
        res["resolved_by"] = "windows"
    return res


def sync_clip(song, clip, near_s=None):
    """Measure the offset between a phone clip (any container) and the master, as song_time = video_time + offset.
    The mic audio is measured on its own clock, then moved onto the video's clock by the container's start-time
    skew (padding the gap with digital silence instead would plant a fake onset where the audio starts)."""
    info = probe(clip)
    if stream(info, "audio") is None:
        raise SyncRefused(f"{clip} has no audio track; Real One syncs on the phone's own mic")
    master = load_audio(song, SR)
    mic = load_audio(clip, SR)
    res = measure_offset(master, mic, near_s)
    skew = av_skew(info)
    res["audio_offset_s"] = res["offset_s"]
    res["av_skew_s"] = round(skew, 5)
    res["offset_s"] = round(res["offset_s"] - skew, 5)
    res["offset_ms"] = round(res["offset_s"] * 1000, 1)
    vid = stream(info, "video") or {}
    res["clip_duration_s"] = round(float(info.get("format", {}).get("duration") or 0), 3)
    res["video_start_s"] = float(vid.get("start_time") or 0)
    res["audio_start_s"] = float((stream(info, "audio") or {}).get("start_time") or 0)
    return res
