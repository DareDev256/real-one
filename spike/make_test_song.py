#!/usr/bin/env python3
"""Synthesize a procedural test song with a KNOWN structure (no samples, no rights issues).

98 BPM, 4/4, a 0.40 s lead-in, then: intro 4 bars, verse 8, hook 8, verse 8, hook 8, one ending hit.
Kick, snare, hats, bass, pad; the hook adds claps, open hats, a lead line, a wide saw-chord layer and
+3.5 dB on the drums (a chorus that changes the arrangement, the way real ones do).
A formant-synth "vocal" sings a different line in each verse and the same hook line in both hooks, with
ad-libs in hook 2; hook 2's lead also sits an octave up with a new rhythm in its back half, and verse 2 adds a
shaker. Real songs differ between repeats the same way (different words, new takes), and it keeps sync honest:
a clip that only covers rhythm-identical repeats should refuse, not guess (tests build that case on purpose).

Ground truth (tempo, first downbeat, section bars and times) goes to <out>.truth.json.
Plumbing test only: never the demo.

    python spike/make_test_song.py footage/test_song.wav
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 44100
STRUCTURE = [("intro", 4), ("verse", 8), ("hook", 8), ("verse", 8), ("hook", 8)]
PROGRESSION = [(57, "m"), (53, "M"), (48, "M"), (55, "M")]  # Am F C G (MIDI roots)
LEAD = [  # (beat offset in bar, length in beats, semitones above A4) per bar, 4-bar phrase
    [(0, 1, 0), (1, 0.5, 3), (1.5, 0.5, 5), (2, 1.5, 7), (3.5, 0.5, 5)],
    [(0, 1, 3), (1, 1, 0), (2, 1, -2), (3, 1, 0)],
    [(0, 0.5, 7), (0.5, 0.5, 10), (1, 1, 12), (2, 1, 10), (3, 1, 7)],
    [(0, 2, 5), (2, 1, 3), (3, 1, 2)],
]
LEAD_B = [  # hook 2, bars 5-8: new rhythm
    [(0, 0.5, 12), (0.5, 0.5, 12), (1, 0.5, 10), (1.5, 0.5, 7), (2, 2, 5)],
    [(0, 1.5, 3), (1.5, 0.5, 5), (2, 1, 7), (3, 1, 10)],
    [(0, 0.5, 12), (0.5, 0.5, 15), (1, 1, 12), (2, 0.5, 10), (2.5, 0.5, 7), (3, 1, 5)],
    [(0, 3, 0), (3, 1, -5)],
]


VOWELS = {"a": ((800, 90), (1200, 110), (2500, 160)), "i": ((300, 60), (2300, 140), (3000, 200)),
          "o": ((500, 70), (900, 90), (2400, 160)), "e": ((450, 70), (1900, 120), (2600, 170))}


def midi_hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def _filt(x, kind, f, order=2):
    sos = butter(order, f, btype=kind, fs=SR, output="sos")
    return sosfilt(sos, x)


class Song:
    def __init__(self, bpm, lead_in, seed, structure=STRUCTURE, vocals_only=False, jitter_s=0.0):
        self.structure = structure
        self.vocals_only, self.jitter_s = vocals_only, jitter_s
        self.bpm, self.lead_in = bpm, lead_in
        self.beat = 60.0 / bpm
        self.bar = 4 * self.beat
        self.rng = np.random.default_rng(seed)
        self.n_bars = sum(n for _, n in structure)
        self.dur = lead_in + self.n_bars * self.bar + 2.5
        self.buf = np.zeros(int(self.dur * SR) + SR, np.float64)

    def t(self, bar, beat=0.0):
        return self.lead_in + bar * self.bar + beat * self.beat

    def add(self, when, sig, gain=1.0, vocal=False):
        if self.vocals_only and not vocal:
            return
        if self.jitter_s:
            when += self.rng.uniform(-self.jitter_s, self.jitter_s)
        i = int(round(when * SR))
        sig = sig * gain * (1 + self.rng.uniform(-0.08, 0.08))  # velocity humanize; timing stays exact
        self.buf[i:i + len(sig)] += sig[:max(0, len(self.buf) - i)]

    # ------------------------------------------------------------------ one-shots
    def kick(self):
        n = int(0.38 * SR)
        tt = np.arange(n) / SR
        f = 46 + 80 * np.exp(-tt / 0.028)
        ph = 2 * np.pi * np.cumsum(f) / SR
        return np.sin(ph) * np.exp(-tt / 0.22) + 0.25 * np.exp(-tt / 0.002) * self.rng.standard_normal(n)

    def snare(self):
        n = int(0.25 * SR)
        tt = np.arange(n) / SR
        noise = _filt(self.rng.standard_normal(n), "bandpass", [1200, 7000]) * np.exp(-tt / 0.09)
        return 0.75 * noise + 0.5 * np.sin(2 * np.pi * 190 * tt) * np.exp(-tt / 0.06)

    def clap(self):
        n = int(0.22 * SR)
        tt = np.arange(n) / SR
        env = sum(np.exp(-np.clip(tt - d, 0, None) / 0.012) * (tt >= d) for d in (0, 0.011, 0.023))
        env += 0.6 * np.exp(-np.clip(tt - 0.03, 0, None) / 0.08) * (tt >= 0.03)
        return _filt(self.rng.standard_normal(n), "bandpass", [900, 3200]) * env * 0.9

    def hat(self, open_=False):
        n = int((0.3 if open_ else 0.07) * SR)
        tt = np.arange(n) / SR
        return _filt(self.rng.standard_normal(n), "highpass", 7000) * np.exp(-tt / (0.12 if open_ else 0.018))

    def shaker(self):
        n = int(0.06 * SR)
        tt = np.arange(n) / SR
        return _filt(self.rng.standard_normal(n), "bandpass", [4000, 9000]) * np.sin(np.pi * tt / 0.06) ** 2

    def crash(self):
        n = int(2.2 * SR)
        tt = np.arange(n) / SR
        return _filt(self.rng.standard_normal(n), "highpass", 3500) * np.exp(-tt / 0.6)

    def tone(self, hz, secs, kind):
        n = int(secs * SR)
        tt = np.arange(n) / SR
        if kind == "saw":  # band-limited saw
            s = sum(np.sin(2 * np.pi * hz * k * tt) / k for k in range(1, 12) if hz * k < 8000)
            env = np.minimum(1, tt / 0.004) * np.exp(-tt / 0.35)
        elif kind == "square":
            s = sum(np.sin(2 * np.pi * hz * k * tt) / k for k in range(1, 14, 2) if hz * k < 9000)
            env = np.minimum(1, tt / 0.006) * np.clip(1 - (tt - secs + 0.03) / 0.03, 0, 1) * (0.8 + 0.2 * np.exp(-tt / 0.1))
        elif kind == "padsaw":  # sustained saw, soft attack and release
            s = sum(np.sin(2 * np.pi * hz * k * tt) / k for k in range(1, 10) if hz * k < 7000)
            env = np.minimum(1, tt / 0.05) * np.clip((secs - tt) / 0.15, 0, 1)
        else:  # pad: sine stack, slow attack
            s = np.sin(2 * np.pi * hz * tt) + 0.3 * np.sin(4 * np.pi * hz * tt)
            env = np.minimum(1, tt / 0.25) * np.clip((secs - tt) / 0.3, 0, 1)
        return s * env

    def syllable(self, hz, secs, vowel):
        """Formant-filtered harmonic tone: a sung syllable, not a person's voice."""
        n = int(secs * SR)
        tt = np.arange(n) / SR
        vib = 1 + 0.006 * np.sin(2 * np.pi * 5.5 * tt) * np.minimum(1, tt / 0.15)
        ph = 2 * np.pi * np.cumsum(hz * vib) / SR
        s = np.zeros(n)
        for k in range(1, int(5000 / hz)):
            amp = sum(np.exp(-((k * hz - F) / bw) ** 2) for F, bw in VOWELS[vowel]) + 0.02
            s += amp * np.sin(k * ph) / np.sqrt(k)
        env = np.minimum(1, tt / 0.012) * np.clip((secs - tt) / 0.04, 0, 1)
        consonant = _filt(self.rng.standard_normal(min(n, int(0.02 * SR))), "highpass", 3000) * 0.3
        s[:len(consonant)] += consonant
        return s * env

    def vocal_line(self, bar0, nbars, seed, root_midi, adlib=False):
        r = np.random.default_rng(seed)
        scale = [0, 3, 5, 7, 10, 12]
        for b in range(nbars):
            phrase = np.random.default_rng(seed * 100 + b % 4)  # the line repeats every 4 bars, like lyrics
            pos = 0.0
            while pos < 3.75:
                ln = phrase.choice([0.25, 0.5, 0.5, 0.75, 1.0])
                if phrase.random() < 0.8:
                    note = root_midi + scale[phrase.integers(0, len(scale))]
                    v = "aioe"[phrase.integers(0, 4)]
                    self.add(self.t(bar0 + b, pos), self.syllable(midi_hz(note), ln * self.beat * 0.9, v), 0.09,
                             vocal=True)
                pos += ln
            if adlib and r.random() < 0.7:  # ad-libs: extra off-grid syllables an octave up
                at = r.choice([1.5, 2.5, 3.25])
                self.add(self.t(bar0 + b, at), self.syllable(midi_hz(root_midi + 12 + scale[r.integers(0, 5)]),
                                                             0.3 * self.beat, "a"), 0.06, vocal=True)

    # ------------------------------------------------------------------ arrangement
    def render(self):
        bar0 = 0
        hook_n = 0
        verse_n = 0
        for name, nbars in self.structure:
            if name == "hook":
                hook_n += 1
            if name == "verse":
                verse_n += 1
            loud = 1.5 if name == "hook" else 1.0
            if name == "verse":
                self.vocal_line(bar0, nbars, seed=100 + verse_n, root_midi=57)
            elif name == "hook":
                self.vocal_line(bar0, nbars, seed=200, root_midi=64, adlib=hook_n == 2)
            self.add(self.t(bar0), self.crash(), 0.2 if name != "intro" else 0.0)
            for b in range(nbars):
                gb = bar0 + b
                root, q = PROGRESSION[gb % 4]
                # pad: the chord changes on every downbeat (harmonic cue for downbeat detection)
                third = 4 if q == "M" else 3
                for n in (root, root + third, root + 7):
                    self.add(self.t(gb), self.tone(midi_hz(n), self.bar, "pad"), 0.07 if name == "intro" else 0.05)
                for k in range(8):  # eighth-note hats
                    self.add(self.t(gb, k * 0.5), self.hat(), (0.10 if k % 2 else 0.16) * (0.7 if name == "intro" else 1))
                if name == "intro":
                    self.add(self.t(gb), self.kick(), 0.6)
                    continue
                for beat in (0, 2):
                    self.add(self.t(gb, beat), self.kick(), 0.9 * loud)
                for beat in (1, 3):
                    self.add(self.t(gb, beat), self.snare(), 0.55 * loud)
                for k in range(8):  # bass: root eighths
                    self.add(self.t(gb, k * 0.5), self.tone(midi_hz(root - 24), self.beat * 0.45, "saw"), 0.22)
                if name == "verse" and verse_n == 2:
                    for k in range(16):
                        self.add(self.t(gb, k * 0.25), self.shaker(), 0.035)
                if name == "hook":
                    for n in (root, root + third, root + 7):  # wide saw chord, detuned pair
                        for det in (0.996, 1.004):
                            self.add(self.t(gb), self.tone(midi_hz(n - 12) * det, self.bar, "padsaw"), 0.035)
                    for beat in (1, 3):
                        self.add(self.t(gb, beat), self.clap(), 0.45)
                    for k in (1, 3, 5, 7):
                        self.add(self.t(gb, k * 0.5), self.hat(open_=True), 0.08)
                    phrase = LEAD_B if (hook_n == 2 and b >= 4) else LEAD
                    octave = 12 if hook_n == 2 else 0
                    for off, ln, semi in phrase[b % 4]:
                        self.add(self.t(gb, off), self.tone(midi_hz(69 + semi + octave), ln * self.beat * 0.92, "square"), 0.10)
                if b == nbars - 1:  # snare fill into the next section
                    for k in range(4):
                        self.add(self.t(gb, 3 + k * 0.25), self.snare(), 0.25 + 0.08 * k)
            bar0 += nbars
        self.add(self.t(bar0), self.kick(), 1.0)  # ending hit
        self.add(self.t(bar0), self.crash(), 0.5)
        y = self.buf[: int(self.dur * SR)]
        if self.vocals_only:
            return (y / (np.abs(y).max() + 1e-9) * 0.89).astype(np.float32)
        y = np.tanh(y * 0.9)  # glue
        return (y / (np.abs(y).max() + 1e-9) * 0.89).astype(np.float32)  # about -1 dBFS peak


def make_song(bpm=98.0, lead_in=0.40, seed=11, structure=STRUCTURE):
    """-> (mono float32 at SR, truth dict). Section names in `structure`: intro, verse, hook (anything else is
    arranged like a verse)."""
    s = Song(bpm, lead_in, seed, structure)
    y = s.render()
    sections, bar0 = [], 0
    for name, n in structure:
        sections.append({"name": name, "start_bar": bar0, "end_bar": bar0 + n,
                         "start_s": round(s.t(bar0), 4), "end_s": round(s.t(bar0 + n), 4)})
        bar0 += n
    truth = {"bpm": bpm, "beats_per_bar": 4, "lead_in_s": lead_in, "first_downbeat_s": lead_in,
             "bar_s": round(s.bar, 6), "n_bars": s.n_bars, "sections": sections,
             "hook": [x for x in sections if x["name"] == "hook"], "duration_s": round(len(y) / SR, 3),
             "sr": SR, "seed": seed, "note": "procedural test song, never the demo"}
    return y, truth


def make_live_vocal(bpm=98.0, lead_in=0.40, seed=11, structure=STRUCTURE, jitter_s=0.025):
    """The artist singing along on set: the song's vocal lines as a separate take (timing off by up to
    +-25 ms per syllable, different humanize, new ad-lib draws). Mono float32 at SR."""
    return Song(bpm, lead_in, seed + 1000, structure, vocals_only=True, jitter_s=jitter_s).render()


def phone_mic(song, offset_s, secs, seed=0, live_vocal=None, vocal_db=0.0, snr_db=20.0, sr=SR):
    """What a phone on set records while the song plays out loud: song[offset:offset+secs] band-limited
    (150 Hz to 7 kHz), a short room response, the artist singing live `vocal_db` relative to the playback,
    and room noise at `snr_db`. offset_s may be negative (the phone started before the song)."""
    rng = np.random.default_rng(seed)
    n = int(round(secs * sr))
    i0 = int(round(offset_s * sr))

    def window(x):
        out = np.zeros(n, np.float64)
        a, b = max(0, i0), min(len(x), i0 + n)
        if b > a:
            out[a - i0:b - i0] = x[a:b]
        return out

    play = sosfilt(butter(2, [150, 7000], btype="bandpass", fs=sr, output="sos"), window(song))
    ir_n = int(0.08 * sr)
    ir = rng.standard_normal(ir_n) * np.exp(-np.arange(ir_n) / (0.02 * sr)) * 0.05
    ir[0] = 1.0
    play = np.convolve(play, ir)[:n]
    p_rms = np.sqrt(np.mean(play ** 2)) + 1e-12
    mic = play
    if live_vocal is not None:
        v = sosfilt(butter(2, [120, 9000], btype="bandpass", fs=sr, output="sos"), window(live_vocal))
        v *= p_rms / (np.sqrt(np.mean(v ** 2)) + 1e-12) * 10 ** (vocal_db / 20)
        mic = mic + v
    noise = sosfilt(butter(1, 2000, fs=sr, output="sos"), rng.standard_normal(n))
    noise *= p_rms / (np.sqrt(np.mean(noise ** 2)) + 1e-12) * 10 ** (-snr_db / 20)
    mic = mic + noise
    return (mic / (np.abs(mic).max() + 1e-9) * 0.5).astype(np.float32)


def main(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    y, truth = make_song()
    sf.write(str(out), np.stack([y, y], axis=1), SR, subtype="PCM_16")
    out.with_suffix(".truth.json").write_text(json.dumps(truth, indent=2))
    print(f"wrote {out} ({truth['duration_s']}s, {truth['bpm']} BPM) + {out.with_suffix('.truth.json').name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "footage/test_song.wav")
