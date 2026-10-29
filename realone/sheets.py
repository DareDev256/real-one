"""Contact sheets, read back from the ENCODED video (what a viewer actually gets), not from memory."""
import cv2
import numpy as np

from realone.common import H, W
from realone.media import read_frames

INK = (236, 236, 236)
DIM = (150, 150, 150)
VERMILION = (43, 65, 232)  # BGR
BG = 14
FONT = cv2.FONT_HERSHEY_SIMPLEX


def grab(video, idxs):
    want = {int(i): None for i in idxs}
    top = max(want)
    for i, fr in enumerate(read_frames(video)):
        if i in want:
            want[i] = fr.copy()
        if i >= top:
            break
    return {i: (f if f is not None else np.zeros((H, W, 3), np.uint8)) for i, f in want.items()}


def text_bar(width, lines, height=None):
    lh = 19
    h = height or (10 + lh * len(lines))
    bar = np.full((h, width, 3), BG, np.uint8)
    for j, (txt, col, scale) in enumerate(lines):
        cv2.putText(bar, txt, (8, 18 + j * lh), FONT, scale, col, 1, cv2.LINE_AA)
    return bar


def edge_crop(frame, bbox, size=300):
    """1:1 crop around the chin/collar (where concave pockets and fringes live)."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, y0 + (y1 - y0) * 0.24
    xs = int(np.clip(cx - size / 2, 0, W - size))
    ys = int(np.clip(cy - size / 2, 0, H - size))
    return frame[ys:ys + size, xs:xs + size]


def contact_sheet(video, plan, bbox_out, look, path, title, world_names):
    shots = plan["shots"]
    tw, th = 300, 533
    picks = []
    for sh in shots:
        picks += [sh["start_frame"], (sh["start_frame"] + sh["end_frame"]) // 2]
    frames = grab(video, picks)
    cols = []
    for sh in shots:
        f_cut, f_mid = sh["start_frame"], (sh["start_frame"] + sh["end_frame"]) // 2
        cut_tile = cv2.resize(frames[f_cut], (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(cut_tile, (0, 0), (tw - 1, th - 1), VERMILION, 3)
        mid_tile = cv2.resize(frames[f_mid], (tw, th), interpolation=cv2.INTER_AREA)
        crop = edge_crop(frames[f_mid], bbox_out[f_mid])
        lab1 = text_bar(tw, [
            (f"SHOT {sh['index'] + 1}  CUT {sh['cut_time_s']:.3f}s  (frame {f_cut})", VERMILION, 0.45),
            (f"bar {sh['bar'] + 1} beat 1 = downbeat {sh['downbeat_s']:.3f}s", INK, 0.45),
            (f"cut error {sh['cut_error_ms']:+.1f} ms   beat #{sh['beat_index']}", INK, 0.45),
            (f"{sh['framing']} x{sh['zoom']:.1f}   world: {world_names[sh['world']]}", DIM, 0.45)])
        lab2 = text_bar(tw, [(f"mid-shot  t={plan['start_s'] + f_mid / 30:.2f}s  (frame {f_mid})", DIM, 0.45)])
        lab3 = text_bar(tw, [("edge, 1:1 pixels (chin / collar)", DIM, 0.45)])
        cols.append(np.vstack([cut_tile, lab1, mid_tile, lab2, crop, lab3]))
    gut = np.full((cols[0].shape[0], 8, 3), BG, np.uint8)
    body = np.hstack([gut] + [x for c in cols for x in (c, gut)])
    head = text_bar(body.shape[1], [(title, INK, 0.6), (f"look: {look}   frames read back from {video.name}", DIM, 0.45)],
                    height=56)
    sheet = np.vstack([head, body, np.full((8, body.shape[1], 3), BG, np.uint8)])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def looks_sheet(videos, plan, path, title):
    """Rows = looks, columns = the mid frame of each shot, same frames in every row."""
    tw, th = 270, 480
    mids = [(s["start_frame"] + s["end_frame"]) // 2 for s in plan["shots"]]
    rows = []
    for look, video in videos.items():
        frames = grab(video, mids)
        tiles = []
        for sh, f in zip(plan["shots"], mids):
            t = cv2.resize(frames[f], (tw, th), interpolation=cv2.INTER_AREA)
            tiles.append(np.vstack([t, text_bar(tw, [(f"{look}  shot {sh['index'] + 1}  {sh['framing']}", INK, 0.45)])]))
        gut = np.full((tiles[0].shape[0], 8, 3), BG, np.uint8)
        rows.append(np.hstack([gut] + [x for t in tiles for x in (t, gut)]))
    sep = np.full((8, rows[0].shape[1], 3), BG, np.uint8)
    head = text_bar(rows[0].shape[1], [(title, INK, 0.55)], height=34)
    sheet = np.vstack([head] + [x for r in rows for x in (r, sep)])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def freeze_sheet(video, plan, freeze, wf, path, title):
    """One freeze, read back from the encoded file: frames before / inside / after it, each labelled with its
    world clock, plus a tile of what changed between two frozen frames (grain ignored): only the artist."""
    f0, f1, n = freeze["start_frame"], freeze["end_frame"], plan["n_frames"]
    a_in, b_in = min(f1 - 1, f0 + 3), max(f0, f1 - 4)
    picks = [("before", f0 - 6), ("frozen", a_in), ("frozen", (f0 + f1) // 2), ("frozen", b_in),
             ("released", f1 + 2), ("released", f1 + 30)]
    picks = [(k, i) for k, i in picks if 0 <= i < n]
    frames = grab(video, [i for _, i in picks])
    tw, th = 300, 533
    tiles = []
    for kind, i in picks:
        t = cv2.resize(frames[i], (tw, th), interpolation=cv2.INTER_AREA)
        col = VERMILION if kind == "frozen" else INK
        if kind == "frozen":
            cv2.rectangle(t, (0, 0), (tw - 1, th - 1), VERMILION, 3)
        lab = text_bar(tw, [(f"{kind.upper()}  frame {i}  t={plan['start_s'] + i / 30:.2f}s", col, 0.45),
                            (f"world clock {int(wf[i])}  (frame index {i})", DIM, 0.45)])
        tiles.append(np.vstack([t, lab]))
    d = np.abs(frames[a_in].astype(np.int16) - frames[b_in].astype(np.int16)).max(axis=2)
    moved = (d > 28).astype(np.uint8) * 255
    moved = cv2.cvtColor(cv2.resize(moved, (tw, th), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    lab = text_bar(tw, [(f"changed: frame {a_in} vs {b_in}", VERMILION, 0.45),
                        ("white = moved (the artist); black = held", DIM, 0.45)])
    tiles.append(np.vstack([moved, lab]))
    gut = np.full((tiles[0].shape[0], 8, 3), BG, np.uint8)
    body = np.hstack([gut] + [x for t in tiles for x in (t, gut)])
    head = text_bar(body.shape[1], [(title, INK, 0.55),
                                    (f"freeze {freeze['role']} bars {freeze['start_bar'] + 1}-{freeze['end_bar']}: "
                                     f"frames {f0}-{f1 - 1}, world clock held at {int(wf[f0])}; "
                                     f"read back from {video.name}", DIM, 0.45)], height=56)
    sheet = np.vstack([head, body, np.full((8, body.shape[1], 3), BG, np.uint8)])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def receipt_artist_region(frame, erode=9):
    """In the receipt look only the filmed artist has colour: saturated pixels, eroded away from the matte."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] > 40).astype(np.uint8)
    return cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)) > 0


def laplacian_var(frame, region):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F)[region].var())


def punchin_ab_sheet(video_new, video_old, frames, path, title, crop_boxes):
    """Native punch-in vs the 0.1.0 upscaled proxy, same frames read back from both encodes: full frame and a
    1:1 crop each, with the Laplacian variance of the artist (receipt look: the coloured pixels)."""
    a, b = grab(video_new, frames), grab(video_old, frames)
    rows, stats = [], []
    for f, (x0, y0, x1, y1) in zip(frames, crop_boxes):
        reg = receipt_artist_region(a[f]) & receipt_artist_region(b[f])
        va, vb = laplacian_var(a[f], reg), laplacian_var(b[f], reg)
        stats.append((f, va, vb))
        tiles = []
        for img, lab, v in ((b[f], "0.1.0: proxy upscaled", vb), (a[f], "0.2.0: native 4K", va)):
            full = cv2.resize(img, (270, 480), interpolation=cv2.INTER_AREA)
            cv2.rectangle(full, (x0 // 4, y0 // 4), (x1 // 4, y1 // 4), VERMILION, 1)
            crop = cv2.resize(img[y0:y1, x0:x1], (480, 480), interpolation=cv2.INTER_NEAREST)
            bar = text_bar(270 + 8 + 480, [(f"{lab}   frame {f}", INK, 0.45),
                                           (f"Laplacian variance of the artist: {v:.1f}   (1:1 crop shown 1.5x, "
                                            f"nearest neighbour)", DIM, 0.42)])
            tiles.append(np.vstack([np.hstack([full, np.full((480, 8, 3), BG, np.uint8), crop]), bar]))
        gut = np.full((tiles[0].shape[0], 8, 3), BG, np.uint8)
        rows.append(np.hstack([gut, tiles[0], gut, tiles[1], gut]))
    sep = np.full((8, rows[0].shape[1], 3), BG, np.uint8)
    head = text_bar(rows[0].shape[1], [(title, INK, 0.55)], height=34)
    cv2.imwrite(str(path), np.vstack([head] + [x for r in rows for x in (r, sep)]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path, stats
