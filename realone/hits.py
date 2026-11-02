"""Hits on the beat (`--hits`): the whole finished frame kicks on the song's beats, so the video moves with the
music instead of only cutting with it (James, Nov 2: "it should have hit or effects based off the beat").

- every downbeat: a zoom punch (scale up, easing back over a few frames) and a short exposure kick;
- the backbeats (beats 2 and 4 in 4/4, where the snare sits): a smaller punch, no flash;
- hook sections hit harder and add a short camera shake on each downbeat.

Deterministic (the shake comes from the frame index, not a random seed), applied after the compositor like the
Hook Freeze hit, so the artist's pixels move with the frame but are never generated, and the Real Receipt's filmed
share is unchanged. No chromatic aberration, ever (James's rule). The frames a hit changes are returned as designed
changes so the render's cut check does not read a downbeat punch as an unplanned cut."""
import numpy as np

PUNCH_DOWN = 0.045      # scale kick on a downbeat (1 + this), easing back over PUNCH_FRAMES
PUNCH_BACK = 0.018      # on a backbeat
PUNCH_FRAMES = 6
FLASH_DOWN = 0.10       # share toward white on a downbeat, over FLASH_FRAMES
FLASH_FRAMES = 3
FLASH_GAIN = 2.2        # exposure x(1 + this * flash): a downbeat is x1.22, a hook downbeat x1.35
HOOK_GAIN = 1.6         # hooks hit this much harder
SHAKE_PX = 9.0          # hook downbeats: the frame jolts this far, settling over SHAKE_FRAMES
SHAKE_FRAMES = 5


def _ease(k, length):
    """1 at the hit frame, smoothly to 0 after `length` frames (a decaying kick, no hard step at the end)."""
    u = np.clip(k / max(1, length), 0.0, 1.0)
    return (1 - u) ** 2


def profile(timeline, s0, n, fps=30):
    """-> dict of per-frame arrays: punch (scale, 1.0 = none), flash (0..1), dx, dy (pixels), and `designed`: the
    frames whose whole picture a hit changes (for the cut check)."""
    punch = np.ones(n)
    flash = np.zeros(n)
    dx, dy = np.zeros(n), np.zeros(n)
    designed = set()
    beats = np.asarray(timeline.get("beats") or [], float)
    downs = set(round(float(x), 3) for x in timeline.get("downbeats") or [])
    bpb = int(timeline.get("beats_per_bar") or 4)
    hooks = [(float(s["start_s"]), float(s["end_s"])) for s in timeline.get("sections", []) if s.get("is_hook")]

    def in_hook(t):
        return any(a <= t < b for a, b in hooks)

    k_in_bar = 0
    for b in beats:
        is_down = round(float(b), 3) in downs
        k_in_bar = 0 if is_down else k_in_bar + 1
        f0 = int(round((b - s0) * fps))
        if f0 < 0 or f0 >= n:
            continue
        backbeat = bpb == 4 and k_in_bar in (1, 3)
        if not (is_down or backbeat):
            continue
        gain = HOOK_GAIN if in_hook(float(b)) else 1.0
        amp = (PUNCH_DOWN if is_down else PUNCH_BACK) * gain
        for k in range(PUNCH_FRAMES + 1):
            i = f0 + k
            if i < n:
                punch[i] = max(punch[i], 1.0 + amp * _ease(k, PUNCH_FRAMES))
        designed.update(range(f0, min(n, f0 + 2)))
        if is_down:
            for k in range(FLASH_FRAMES + 1):
                i = f0 + k
                if i < n:
                    flash[i] = max(flash[i], FLASH_DOWN * gain * _ease(k, FLASH_FRAMES))
            if gain > 1.0:  # the hook's jolt: a direction from the frame index, settling back
                ang = (f0 * 2.399963) % (2 * np.pi)  # the golden angle: no two downbeats jolt the same way
                for k in range(SHAKE_FRAMES + 1):
                    i = f0 + k
                    if i < n:
                        e = _ease(k, SHAKE_FRAMES) * (1 if k % 2 == 0 else -0.6)
                        dx[i], dy[i] = SHAKE_PX * np.cos(ang) * e, SHAKE_PX * np.sin(ang) * e
    return {"punch": punch, "flash": flash, "dx": dx, "dy": dy, "designed": designed}


def apply(frame, punch, flash, dx=0.0, dy=0.0):
    """Punch (scale about the centre), jolt (translate) and flash on a finished uint8 frame."""
    import cv2
    if punch == 1.0 and flash == 0.0 and dx == 0.0 and dy == 0.0:
        return frame
    out = frame
    if punch != 1.0 or dx or dy:
        h, w = frame.shape[:2]
        M = np.float32([[punch, 0, (1 - punch) * w / 2 + dx], [0, punch, (1 - punch) * h / 2 + dy]])
        out = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if flash > 0:  # an exposure kick: highlights bloom, blacks stay black (mixing toward white read as haze)
        out = (out.astype(np.float32) * (1 + FLASH_GAIN * flash)).clip(0, 255).astype(np.uint8)
    return out
