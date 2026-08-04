"""Superresolucion con IA: Real-ESRGAN x4 (ONNX + CUDA).

Aumenta la resolucion reconstruyendo detalle real (no un simple estirado).
Se procesa por mosaicos con margen y se pega solo el interior de cada uno,
asi la VRAM no se agota y no quedan costuras.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from photoraw.ai import MODEL_DIR, _make_options, _providers, _setup_dll_paths

SR_MODEL = MODEL_DIR / "real_esrgan_x4.onnx"
SR_URL = ("https://huggingface.co/facefusion/models-3.0.0"
          "/resolve/main/real_esrgan_x4.onnx")

_session = None


def model_available():
    return SR_MODEL.exists() and SR_MODEL.stat().st_size > 10_000_000


def download_model(progress_cb=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SR_MODEL.with_suffix(".onnx.tmp")

    def hook(blocks, block_size, total):
        if progress_cb and total > 0:
            progress_cb(min(blocks * block_size / total, 1.0))

    urllib.request.urlretrieve(SR_URL, tmp, hook)
    tmp.replace(SR_MODEL)


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
            str(SR_MODEL), sess_options=_make_options(),
            providers=_providers())
    return _session


def upscale(img, scale=4, tile=256, margin=16, progress_cb=None):
    """img: float32 RGB 0..1. Devuelve uint8 RGB con `scale`x mas pixeles
    (2 o 4). El modelo trabaja a 4x; para 2x se reduce el resultado."""
    sess = _get_session()
    input_name = sess.get_inputs()[0].name
    h, w = img.shape[:2]
    out = np.zeros((h * 4, w * 4, 3), np.uint8)

    ys = list(range(0, h, tile))
    xs = list(range(0, w, tile))
    total = len(ys) * len(xs)
    done = 0
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            # ventana con margen de contexto alrededor
            my0, mx0 = max(y0 - margin, 0), max(x0 - margin, 0)
            my1, mx1 = min(y1 + margin, h), min(x1 + margin, w)
            patch = np.ascontiguousarray(
                img[my0:my1, mx0:mx1].transpose(2, 0, 1)[None],
                dtype=np.float32)
            y_out = sess.run(None, {input_name: patch})[0][0]
            y_out = np.clip(y_out.transpose(1, 2, 0), 0.0, 1.0)
            # pegar solo el interior (sin el margen), ya ampliado 4x
            iy0, ix0 = (y0 - my0) * 4, (x0 - mx0) * 4
            iy1, ix1 = iy0 + (y1 - y0) * 4, ix0 + (x1 - x0) * 4
            out[y0 * 4:y1 * 4, x0 * 4:x1 * 4] = \
                (y_out[iy0:iy1, ix0:ix1] * 255.0 + 0.5).astype(np.uint8)
            done += 1
            if progress_cb:
                progress_cb(done / total)

    if scale == 2:
        out = cv2.resize(out, (w * 2, h * 2), interpolation=cv2.INTER_AREA)
    return out
