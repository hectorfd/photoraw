"""Mascaras IA: segmentacion de sujeto con u2net (ONNX + CUDA).

Detecta el sujeto principal de la foto y devuelve un mapa de peso 0..1.
El "fondo" es simplemente el mapa invertido. La segmentacion se calcula a
320x320 (suficiente para una mascara suave) y se reescala.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from photoraw.ai import MODEL_DIR, _make_options, _providers, _setup_dll_paths

U2NET_MODEL = MODEL_DIR / "u2net.onnx"
U2NET_URL = ("https://github.com/danielgatis/rembg/releases/download"
             "/v0.0.0/u2net.onnx")

_session = None


def model_available():
    return U2NET_MODEL.exists() and U2NET_MODEL.stat().st_size > 10_000_000


def download_model(progress_cb=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = U2NET_MODEL.with_suffix(".onnx.tmp")

    def hook(blocks, block_size, total):
        if progress_cb and total > 0:
            progress_cb(min(blocks * block_size / total, 1.0))

    urllib.request.urlretrieve(U2NET_URL, tmp, hook)
    tmp.replace(U2NET_MODEL)


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        _setup_dll_paths()
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        _session = ort.InferenceSession(
            str(U2NET_MODEL), sess_options=_make_options(),
            providers=_providers())
    return _session


def subject_mask(img):
    """img: float32 RGB 0..1. Devuelve mapa float32 h x w en 0..1
    (1 = sujeto, 0 = fondo)."""
    sess = _get_session()
    h, w = img.shape[:2]
    small = cv2.resize(np.clip(img, 0.0, 1.0), (320, 320),
                       interpolation=cv2.INTER_AREA)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = (small.astype(np.float32) - mean) / std
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    out = (out - out.min()) / max(float(out.max() - out.min()), 1e-6)
    m = cv2.resize(out.astype(np.float32), (w, h),
                   interpolation=cv2.INTER_LINEAR)
    # borde ligeramente suavizado para fusiones naturales
    m = cv2.GaussianBlur(m, (0, 0), max(max(h, w) / 800.0, 1.0))
    return np.clip(m, 0.0, 1.0)
