"""Master-track analysis: beat grid, downbeats, sections, and which section is the hook.

Everything is computed from the master file (never the phone mic). Heuristics, stated plainly:
- Beats: librosa's dynamic-programming tracker on the onset envelope, then a constant-tempo grid is fitted to
  them; if the residuals are small (a DAW-tempo song) the fitted grid replaces the raw beats, otherwise the raw
  beats stay (`grid: "tracked"`).
- Downbeats: the bar phase whose beats carry the most low-band (kick) onset energy plus harmonic change (chords
  tend to change on the one). Bar = 4 beats unless 3 is clearly stronger, or `beats_per_bar` is given.
- Sections: bar-level features (chroma, MFCC, loudness, onset density), multi-scale novelty, boundaries on bar
  lines, segments labelled by agglomerative clustering. The hook is the repeated label with the most energy
  (count breaks near-ties). Real songs can fool this; `--section bars:A-B` overrides it.
"""
import numpy as np

from realone.common import Refuse
from realone.media import load_audio

SR = 22050
HOP = 256
ONSET_LAG_S = 0.021  # the centred onset envelope peaks 21 ms after a sharp attack (measured on the test song's
#                      exact onsets: +21 ms centred, -25.5 ms uncentred); tests/test_audio.py pins it


def _oenv(y, fmax=None, fmin=None):
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_fft=2048, n_mels=96, fmax=8000)
    freqs = librosa.mel_frequencies(n_mels=96, fmax=8000)
    keep = np.ones(len(freqs), bool)
    if fmax:
        keep &= freqs <= fmax
    if fmin:
        keep &= freqs >= fmin
    return librosa.onset.onset_strength(S=librosa.power_to_db(S[keep]), sr=SR, hop_length=HOP)


def beat_grid(y):
    """-> (beat_times, bpm, grid_kind, residual_ms)."""
    import librosa
    oenv = _oenv(y)
    _, frames = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=HOP, units="frames", trim=True)
    t = librosa.frames_to_time(frames, sr=SR, hop_length=HOP) - ONSET_LAG_S
    if len(t) < 8:
        raise Refuse(f"found only {len(t)} beats in the master; is it music with a pulse?")
    p = float(np.median(np.diff(t)))
    # integer beat index per detected beat, counted IBI by IBI (a skipped beat counts 2, a doubled one 0)
    k = np.r_[0, np.cumsum(np.maximum(np.round(np.diff(t) / p), 0))]
    A = np.stack([np.ones_like(k), k], 1)
    (a, p), *_ = np.linalg.lstsq(A, t, rcond=None)
    res = t - (a + p * k)
    res_ms = float(np.std(res) * 1000)
    bpm = 60.0 / p
    if res_ms < 15 and np.abs(res).max() < 0.06:
        kk = np.arange(int(k.min()), int(k.max()) + 1)
        return a + p * kk, bpm, "constant", res_ms
    return t, float(60.0 / np.median(np.diff(t))), "tracked", res_ms


def _beat_sync(feat, feat_times, beats):
    """Mean of feature columns inside each beat interval [b_i, b_i+1)."""
    idx = np.searchsorted(feat_times, beats)
    out = []
    for i in range(len(beats)):
        a = idx[i]
        b = idx[i + 1] if i + 1 < len(idx) else min(a + 8, feat.shape[1])
        out.append(feat[:, a:max(b, a + 1)].mean(axis=1))
    return np.array(out)


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def downbeat_phase(y, beats, beats_per_bar=None):
    """-> (beats_per_bar, phase, scores). phase = index of the first downbeat in `beats`."""
    import librosa
    low = _oenv(y, fmax=160)
    lt = librosa.frames_to_time(np.arange(len(low)), sr=SR, hop_length=HOP)
    win = int(0.035 * SR / HOP) + 1
    li = np.searchsorted(lt, beats)
    low_b = np.array([low[max(0, i - win):i + win + 1].max() if i < len(low) else 0 for i in li])
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=512)
    ct = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=SR, hop_length=512)
    cb = _beat_sync(chroma, ct, beats)
    cb = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-9)
    chg = np.r_[0, 1 - (cb[1:] * cb[:-1]).sum(axis=1)]
    acc = _z(low_b) + _z(chg)
    scores = {}
    for m in (3, 4):
        ph = np.array([acc[p::m].mean() for p in range(m)])
        best = int(ph.argmax())
        scores[m] = {"phase": best, "contrast": float(ph[best] - np.delete(ph, best).mean()),
                     "by_phase": [round(float(v), 3) for v in ph]}
    if beats_per_bar:
        m = int(beats_per_bar)
        if m not in scores:
            ph = np.array([acc[p::m].mean() for p in range(m)])
            scores[m] = {"phase": int(ph.argmax()), "contrast": 0.0, "by_phase": ph.round(3).tolist()}
    else:
        m = 3 if scores[3]["contrast"] > 1.5 * max(scores[4]["contrast"], 1e-6) else 4
    return m, scores[m]["phase"], scores


def bar_features(y, downbeats, end_s):
    """Per-bar features -> (X, energy). X is robust-scaled, weighted by how slowly each feature moves (a feature
    that flips every bar with the chord cycle says nothing about sections), and projected on the principal
    components that hold 80% of the variance (drops the noise floor of the many weak dimensions)."""
    import librosa
    edges = np.r_[downbeats, end_s]
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=512)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=512, n_mfcc=20)[1:]
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    on = _oenv(y)
    t512 = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=SR, hop_length=512)
    t256 = librosa.frames_to_time(np.arange(len(on)), sr=SR, hop_length=HOP)
    rows, energy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s, e = np.searchsorted(t512, [a, b])
        s2, e2 = np.searchsorted(t256, [a, b])
        e, e2 = max(e, s + 1), max(e2, s2 + 1)
        c = np.median(chroma[:, s:e], axis=1)  # medians: a crash on the one should not define the bar
        rows.append(np.r_[c / (np.linalg.norm(c) + 1e-9), np.median(mfcc[:, s:e], axis=1),
                          20 * np.log10(np.median(rms[s:e]) + 1e-6), np.median(on[s2:e2])])
        energy.append(float(rms[s:e].mean()))
    F = np.array(rows)
    med = np.median(F, 0)
    iqr = np.subtract(*np.percentile(F, [75, 25], 0)) / 1.35 + 1e-6
    Z = np.clip((F - med) / iqr, -4, 4)
    if len(Z) >= 8:
        sm = np.array([np.convolve(Z[:, j], np.ones(4) / 4, mode="valid") for j in range(Z.shape[1])]).T
        slow = sm.var(0) / (Z.var(0) + 1e-9)
    else:
        slow = np.ones(Z.shape[1])
    X = Z * slow
    X = X - X.mean(0)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = S ** 2 / max(np.sum(S ** 2), 1e-12)
    nc = int(np.searchsorted(np.cumsum(ev), 0.80) + 1)
    return X @ Vt[:nc].T, np.array(energy)


def novelty(F, scales=(2, 4)):
    """Novelty at each bar line b (between bar b-1 and b): distance between the means of the bars before and after."""
    nb = len(F)
    nov = np.zeros(nb)
    for L in scales:
        for b in range(1, nb):
            pre, post = F[max(0, b - L):b], F[b:min(nb, b + L)]
            nov[b] += np.linalg.norm(pre.mean(0) - post.mean(0))
    return nov / len(scales)


def candidate_bounds(nov, reach=2):
    """Over-segment on purpose: every local novelty peak above the median is a candidate. Labelling and merging
    decide which candidates are real section changes."""
    nb = len(nov)
    med = np.median(nov[1:]) if nb > 1 else 0
    cand = [b for b in range(1, nb) if nov[b] > med and nov[b] >= nov[max(1, b - reach):b + reach + 1].max()]
    keep = []
    for b in sorted(cand, key=lambda b: -nov[b]):
        if all(abs(b - k) >= 2 for k in keep):
            keep.append(b)
    return [0] + sorted(keep) + [nb]


def label_segments(F, bounds, rel=0.4):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    means = np.array([F[a:b].mean(0) for a, b in zip(bounds[:-1], bounds[1:])])
    if len(means) == 1:
        return [0]
    d = pdist(means)
    lab = fcluster(linkage(means, "average"), t=rel * np.median(d), criterion="distance")
    order, out = {}, []
    for x in lab:  # relabel in order of first appearance: 0, 1, 2 ...
        order.setdefault(x, len(order))
        out.append(order[x])
    return out


def _merge(segs):
    out = []
    for a, e, lab in segs:
        if out and out[-1][2] == lab:
            out[-1][1] = e
        else:
            out.append([a, e, lab])
    return out


def _refine(F, segs):
    """Move each boundary up to 1 bar so the transition bars (a crash on the one, a fill before it) go to whichever
    neighbouring label they actually resemble."""
    cent = {}
    for lab in {s[2] for s in segs}:
        rows = [F[a:e] for a, e, l2 in segs if l2 == lab]
        cent[lab] = np.concatenate(rows).mean(0)
    for i in range(1, len(segs)):
        a0, b, la = segs[i - 1]
        _, e1, lb = segs[i]
        best, best_cost = b, None
        for nb in (b - 1, b, b + 1):
            if nb - a0 < 2 or e1 - nb < 2:
                continue
            cost = sum(np.linalg.norm(F[j] - cent[la]) for j in range(b - 1, nb)) + \
                sum(np.linalg.norm(F[j] - cent[lb]) for j in range(nb, b + 1))
            if best_cost is None or cost < best_cost - 1e-9:
                best, best_cost = nb, cost
        segs[i - 1][1] = best
        segs[i][0] = best
    return segs


def segment(y, downbeats, end_s):
    """-> (list of section dicts, novelty). Bars are indices into `downbeats`."""
    F, energy = bar_features(y, downbeats, end_s)
    nov = novelty(F)
    b = candidate_bounds(nov)
    labs = label_segments(F, b)
    segs = _merge([[a, e, lab] for (a, e), lab in zip(zip(b[:-1], b[1:]), labs)])
    segs = _merge(_refine(F, segs))
    count = {}
    for _, _, lab in segs:
        count[lab] = count.get(lab, 0) + 1
    lab_energy = {lab: float(np.mean([energy[a:e].mean() for a, e, l2 in segs if l2 == lab])) for lab in count}
    emax, cmax = max(lab_energy.values()), max(count.values())
    repeated = [lab for lab in count if count[lab] >= 2] or list(count)
    hook = max(repeated, key=lambda lab: lab_energy[lab] / emax + 0.3 * count[lab] / cmax)
    others = [lab for lab in repeated if lab != hook]
    if not others:  # no repeated non-hook label (verse 2 added a layer): the biggest body section is the verse
        body = [(e - a, -i, lab) for i, (a, e, lab) in enumerate(segs) if lab != hook
                and not (i in (0, len(segs) - 1) and count[lab] == 1)]
        others = [max(body)[2]] if body else []  # longest; on a tie, the earliest
    verse = max(others, key=lambda lab: (count[lab], lab_energy[lab])) if others else None
    edges = np.r_[downbeats, end_s]
    out, seen = [], {}
    for i, (a, e, lab) in enumerate(segs):
        if lab == hook:
            role = "hook"
        elif lab == verse:
            role = "verse"
        elif i == 0 and count[lab] == 1:
            role = "intro"
        elif i == len(segs) - 1 and count[lab] == 1:
            role = "outro"
        else:
            role = "bridge"
        seen[role] = seen.get(role, 0) + 1
        out.append({"index": i, "label": chr(65 + lab), "role": role, "occurrence": seen[role],
                    "start_bar": int(a), "end_bar": int(e), "start_s": round(float(edges[a]), 4),
                    "end_s": round(float(edges[e]), 4), "bars": int(e - a),
                    "energy_db": round(float(20 * np.log10(energy[a:e].mean() + 1e-9)), 2),
                    "repeats": count[lab], "is_hook": lab == hook})
    return out, nov


def analyze(song_path, beats_per_bar=None):
    y = load_audio(song_path, SR)
    dur = len(y) / SR
    if dur < 8:
        raise Refuse(f"the master is {dur:.1f}s long; need a song")
    if np.sqrt(np.mean(y ** 2)) < 1e-4:
        raise Refuse("the master is silent")
    beats, bpm, grid, res_ms = beat_grid(y)
    bpb, phase, mscores = downbeat_phase(y, beats, beats_per_bar)
    downbeats = beats[phase::bpb]
    end_s = float(min(dur, downbeats[-1] + bpb * 60.0 / bpm))
    sections, nov = segment(y, downbeats, end_s)
    hook = next(s for s in sections if s["is_hook"])
    return {
        "song": str(song_path), "duration_s": round(dur, 3), "analysis_sr": SR, "hop": HOP,
        "bpm": round(float(bpm), 3), "beats_per_bar": int(bpb), "grid": grid, "grid_residual_ms": round(res_ms, 2),
        "meter_scores": mscores, "first_downbeat_s": round(float(downbeats[0]), 4),
        "beats": [round(float(t), 4) for t in beats], "downbeats": [round(float(t), 4) for t in downbeats],
        "bar_novelty": [round(float(v), 4) for v in nov],
        "sections": sections, "hook_index": hook["index"],
    }


def pick_section(tl, spec):
    """spec: hook | verse | intro | outro | bridge (optionally ':n' for the n-th) | bars:A-B | #i."""
    secs = tl["sections"]
    if spec.startswith("bars:"):
        a, b = (int(x) for x in spec[5:].split("-"))
        db = tl["downbeats"]
        if not (0 <= a < b <= len(db)):
            raise Refuse(f"bars {a}-{b} outside the song's {len(db)} bars")
        end = db[b] if b < len(db) else tl["duration_s"]
        return {"index": -1, "label": "-", "role": f"bars{a}-{b}", "occurrence": 1, "start_bar": a, "end_bar": b,
                "start_s": db[a], "end_s": end, "bars": b - a, "is_hook": False}
    if spec.startswith("#"):
        i = int(spec[1:])
        if not 0 <= i < len(secs):
            raise Refuse(f"section #{i} does not exist ({len(secs)} sections)")
        return secs[i]
    role, _, n = spec.partition(":")
    n = int(n or 1)
    hits = [s for s in secs if s["role"] == role]
    if len(hits) < n:
        found = ", ".join(f"#{s['index']} {s['role']}" for s in secs)
        raise Refuse(f"no {role} #{n} in this song (found: {found}). Use --section bars:A-B to choose by hand.")
    return hits[n - 1]
