"""Stage depth: relative depth of a generated world still, on the Mac's Neural Engine.

Depth Anything V2 Small (Apple's Core ML build, Apache-2.0, huggingface.co/apple/coreml-depth-anything-v2-small)
through Vision. The model takes a 518 x 392 landscape image, so a 9:16 world is reflect-padded to that aspect
first (squashing it would bend every floor line) and the padding is cut off the result. Output is relative inverse
depth (bigger = nearer), cached next to the world by the image's hash. The artist is never run through it."""
import hashlib
from pathlib import Path

import cv2
import numpy as np

from realone.common import Refuse

MODEL_DIR = Path.home() / ".cache" / "realone" / "models"
MODEL_NAME = "DepthAnythingV2SmallF16"
MODEL_URL = ("https://huggingface.co/apple/coreml-depth-anything-v2-small/resolve/main/"
             f"{MODEL_NAME}.mlpackage")
MODEL_FILES = ("Manifest.json", "Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin")
WEIGHT_SHA256 = "fa60d9b6a155734f59029ebb882fd54e549bfaee3539c1a9cbd2cbbab64a0fed"
IN_W, IN_H = 518, 392
_model = None


def model_path():
    return MODEL_DIR / f"{MODEL_NAME}.mlpackage"


def ensure_model():
    """Download the Core ML package once (48 MB) and refuse unless the weights hash matches the pinned one."""
    import urllib.request
    p = model_path()
    w = p / MODEL_FILES[2]
    if not w.exists():
        for f in MODEL_FILES:
            dst = p / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"depth model: downloading {f}")
            urllib.request.urlretrieve(f"{MODEL_URL}/{f}", dst)
    h = hashlib.sha256(w.read_bytes()).hexdigest()
    if h != WEIGHT_SHA256:
        raise Refuse(f"depth model: weight.bin sha256 {h[:12]}... is not the pinned {WEIGHT_SHA256[:12]}...")
    return p


def _load():
    global _model
    if _model is None:
        import CoreML
        import Foundation
        import Vision
        url = Foundation.NSURL.fileURLWithPath_(str(ensure_model()))
        compiled, err = CoreML.MLModel.compileModelAtURL_error_(url, None)
        if compiled is None:
            raise Refuse(f"depth model: Core ML compile failed: {err}")
        ml, err = CoreML.MLModel.modelWithContentsOfURL_error_(compiled, None)
        if ml is None:
            raise Refuse(f"depth model: load failed: {err}")
        vm, err = Vision.VNCoreMLModel.modelForMLModel_error_(ml, None)
        if vm is None:
            raise Refuse(f"depth model: Vision wrap failed: {err}")
        _model = vm
    return _model


def pad_to_model_aspect(img):
    """Reflect-pad left/right (a tall image) or top/bottom (a wide one) to 518:392. -> (padded, (x0, y0, w, h))."""
    h, w = img.shape[:2]
    target = IN_W / IN_H
    if w / h < target:
        pw = int(round(h * target))
        l = (pw - w) // 2
        out = cv2.copyMakeBorder(img, 0, 0, l, pw - w - l, cv2.BORDER_REFLECT_101)
        return out, (l, 0, w, h)
    ph = int(round(w / target))
    t = (ph - h) // 2
    out = cv2.copyMakeBorder(img, t, ph - h - t, 0, 0, cv2.BORDER_REFLECT_101)
    return out, (0, t, w, h)


def _run(img_bgr):
    import objc
    import Quartz
    import Vision
    from Foundation import NSData
    vm = _load()
    with objc.autorelease_pool():
        small = cv2.resize(img_bgr, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", small)
        data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        req = Vision.VNCoreMLRequest.alloc().initWithModel_(vm)
        req.setImageCropAndScaleOption_(Vision.VNImageCropAndScaleOptionScaleFill)
        okp, err = handler.performRequests_error_([req], None)
        res = req.results() if okp else None
        if not res:
            raise Refuse(f"depth model: no result ({err})")
        pb = res[0].pixelBuffer()
        Quartz.CVPixelBufferLockBaseAddress(pb, 1)
        w, h = Quartz.CVPixelBufferGetWidth(pb), Quartz.CVPixelBufferGetHeight(pb)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
        fmt = Quartz.CVPixelBufferGetPixelFormatType(pb)
        raw = Quartz.CVPixelBufferGetBaseAddress(pb).as_buffer(bpr * h)
        if fmt == Quartz.kCVPixelFormatType_OneComponent16Half:
            arr = np.frombuffer(raw, np.float16).reshape(h, bpr // 2)[:, :w].astype(np.float32)
        elif fmt == Quartz.kCVPixelFormatType_OneComponent32Float:
            arr = np.frombuffer(raw, np.float32).reshape(h, bpr // 4)[:, :w].copy()
        elif fmt == Quartz.kCVPixelFormatType_OneComponent8:
            arr = np.frombuffer(raw, np.uint8).reshape(h, bpr)[:, :w].astype(np.float32) / 255
        else:
            Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
            raise Refuse(f"depth model: unexpected pixel format {fmt}")
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
    return arr


def world_disparity(img_bgr, cache_dir=None):
    """-> float32 (h, w) relative inverse depth for the world still, same size as the still (bigger = nearer),
    edge-aware upsampled from the model's 518 x 392 output. Cached by image hash in cache_dir."""
    key = hashlib.sha1(img_bgr.tobytes()).hexdigest()[:16]
    if cache_dir is not None:
        c = Path(cache_dir) / f"disp_{key}.npy"
        if c.exists():
            return np.load(c)
    padded, (x0, y0, w, h) = pad_to_model_aspect(img_bgr)
    d = _run(padded)
    d = cv2.resize(d, (padded.shape[1], padded.shape[0]), interpolation=cv2.INTER_CUBIC)[y0:y0 + h, x0:x0 + w]
    if not np.isfinite(d).all() or float(d.max() - d.min()) <= 1e-6:
        raise Refuse("depth model: flat or non-finite output")
    # snap depth edges to the image's own edges (a joint bilateral pass guided by the still)
    guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        d = cv2.ximgproc.jointBilateralFilter(guide, d.astype(np.float32), 9, 0.05 * float(d.max() - d.min()), 5)
    except AttributeError:
        pass
    d = d.astype(np.float32)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(cache_dir) / f"disp_{key}.npy", d)
    return d
