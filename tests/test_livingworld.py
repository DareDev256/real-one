"""Living world (realone/livingworld.py): the world's lamps flicker, swell on the kick and stutter on some bars, haze
drifts through their light; all of it a function of the world clock, so a frozen world is a frozen frame."""
import numpy as np

from realone import livingworld as LV
from realone.common import H, W

BEATS = [round(0.5 * k, 3) for k in range(40)]   # 120 BPM, 4/4
DOWNS = BEATS[::4]


def _world(lamps=True):
    w = np.full((H, W, 3), 30, np.float32)
    if lamps:
        w[600:640, 300:340] = (90, 190, 255)   # a warm lamp (BGR)
        w[1100:1130, 760:790] = (80, 170, 250)
    return w


def _ring(img):
    return img[560:700, 260:380].mean()


def test_pulse_on_the_kick():
    lw = LV.LivingWorld(BEATS, DOWNS, 4)
    assert lw.pulse(-0.1) == 0.0
    assert abs(lw.pulse(2.0) - LV.PULSE_DOWN) < 1e-9          # a downbeat
    assert abs(lw.pulse(2.5) - LV.PULSE_BACK) < 1e-9          # beat 2: a backbeat
    assert lw.pulse(3.0) == 0.0                                # beat 3: no swell
    assert lw.pulse(2.0) > lw.pulse(2.1) > lw.pulse(2.3) > 0  # it decays


def test_kick_brightens_the_lamps():
    lw = LV.LivingWorld(BEATS, DOWNS, 4)
    w = _world()
    on, off = lw.apply(w, 4.0), lw.apply(w, 4.45)
    assert _ring(on) > _ring(off) + 3


def test_world_clock_makes_it_deterministic():
    w = _world()
    a = LV.LivingWorld(BEATS, DOWNS, 4).apply(w, 3.7)
    b = LV.LivingWorld(BEATS, DOWNS, 4).apply(w, 3.7)
    assert np.array_equal(a, b)  # a frozen clock (the Hook Freeze) is a frozen frame
    assert not np.array_equal(a, LV.LivingWorld(BEATS, DOWNS, 4).apply(w, 3.9))


def test_no_lamps_no_change():
    w = _world(lamps=False)
    assert LV.LivingWorld(BEATS, DOWNS, 4).apply(w, 2.0) is w


def test_flicker_is_smooth_and_bounded():
    lw = LV.LivingWorld(BEATS, DOWNS, 4)
    f = [lw.flicker(1.0 + k / 30) for k in range(30)]
    assert all(abs(x.mean() - 1) < LV.FLICKER for x in f)
    assert max(np.abs(f[k + 1] - f[k]).max() for k in range(29)) < 0.08  # no frame-to-frame jump


def test_a_dying_lamp_on_some_bars():
    lw = LV.LivingWorld(BEATS, DOWNS, 4)
    dips = [lw.dropout(t / 30).min() for t in range(0, 600)]
    assert min(dips) < 0.3 and max(dips) == 1.0
    bars = [min(dips[int(DOWNS[b] * 30):int(DOWNS[b + 1] * 30)]) < 0.9 for b in range(len(DOWNS) - 1)]
    assert 0 < sum(bars) < 0.6 * len(bars)  # some bars, not every bar


def test_haze_drifts(monkeypatch):
    monkeypatch.setattr(LV, "FLICKER", 0.0)  # no flicker, no beats, no dropout: only the haze can move
    lw = LV.LivingWorld([], [], 4)
    w = _world()
    assert not np.array_equal(lw.apply(w, 1.0), lw.apply(w, 1.5))
