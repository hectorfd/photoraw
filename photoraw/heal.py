"""Pincel corrector: inpainting con LaMa (ONNX + CUDA).

El usuario pinta una mascara y LaMa rellena la zona con contenido coherente
con el entorno. El modelo trabaja a 512x512 fijo, asi que se recorta la zona
con contexto alrededor, se procesa y se reintegra con borde desvanecido.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from photoraw.ai import MODEL_DIR, _make_options, _providers, _setup_dll_paths
from photoraw.generative import _sharpen_patch

LAMA_MODEL = MODEL_DIR / "lama_fp32.onnx"
LAMA_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"

_session = None


def model_available():
    return LAMA_MODEL.exists() and LAMA_MODEL.stat().st_size > 50_000_000


def download_model(progress_cb=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LAMA_MODEL.with_suffix(".onnx.tmp")

    def hook(blocks, block_size, total):
        if progress_cb and total > 0:
            progress_cb(min(blocks * block_size / total, 1.0))

    urllib.request.urlretrieve(LAMA_URL, tmp, hook)
    tmp.replace(LAMA_MODEL)


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
            str(LAMA_MODEL), sess_options=_make_options(),
            providers=_providers())
    return _session


def rasterize_strokes(strokes, h, w):
    """Convierte trazos normalizados [[radio, [[x, y], ...]], ...] en una
    mascara uint8 (255 = borrar) del tamano pedido."""
    mask = np.zeros((h, w), np.uint8)
    scale = max(h, w)
    for radius, points in strokes:
        thickness = max(int(radius * scale * 2), 3)
        pts = np.array([[int(x * w), int(y * h)] for x, y in points], np.int32)
        if len(pts) == 1:
            cv2.circle(mask, tuple(pts[0]), max(thickness // 2, 2), 255, -1)
        else:
            cv2.polylines(mask, [pts], False, 255, thickness=thickness,
                          lineType=cv2.LINE_8)
    return mask


# trazos menores a esta fraccion del lado mayor toleran el relleno clasico
# de respaldo cuando el modelo LaMa no esta descargado (ver needs_model)
FAST_FRAC = 0.06


def _is_small(y0, y1, x0, x1, h, w):
    return max(y1 - y0, x1 - x0) <= FAST_FRAC * max(h, w)


def needs_model(strokes, h, w):
    """True si algun trazo es lo bastante grande para requerir LaMa."""
    mask = rasterize_strokes(strokes, h, w)
    n_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    for i in range(1, n_labels):
        ys, xs = np.where(labels == i)
        if not _is_small(ys.min(), ys.max(), xs.min(), xs.max(), h, w):
            return True
    return False


def _fast_inpaint_region(img, mask, y0, y1, x0, x1):
    """Relleno clasico instantaneo (Telea) para manchas pequenas."""
    crop = img[y0:y1, x0:x1]
    mcrop = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    hard = cv2.dilate(mcrop, np.ones((3, 3), np.uint8))
    crop8 = (np.clip(crop, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    out8 = cv2.inpaint(crop8, hard, 5, cv2.INPAINT_TELEA)
    out = out8.astype(np.float32) / 255.0
    blend = cv2.GaussianBlur((mcrop > 0).astype(np.float32), (0, 0), 2)[..., None]
    img[y0:y1, x0:x1] = crop * (1.0 - blend) + out * blend


def _inpaint_region(img, mask, y0, y1, x0, x1):
    """Procesa un recorte y lo funde de vuelta en img (in place)."""
    sess = _get_session()
    names = [i.name for i in sess.get_inputs()]
    crop = img[y0:y1, x0:x1]
    mcrop = mask[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]

    crop512 = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
    mask512 = cv2.resize(mcrop, (512, 512), interpolation=cv2.INTER_NEAREST)
    mask512 = cv2.dilate(mask512, np.ones((5, 5), np.uint8))

    x_img = np.ascontiguousarray(crop512.transpose(2, 0, 1)[None], dtype=np.float32)
    x_mask = (mask512 > 127).astype(np.float32)[None, None]
    out = sess.run(None, {names[0]: x_img, names[1]: x_mask})[0][0]
    out = out.transpose(1, 2, 0)
    if float(out.max()) > 2.0:  # este export devuelve 0..255
        out = out / 255.0
    out = np.clip(out, 0.0, 1.0)

    # en fotos grandes (sobre todo al exportar a resolucion completa) el
    # recorte puede ser mucho mayor a 512: si se estira con interpolacion
    # simple, el parche sale borroso y "inventado" frente al detalle real
    # de alrededor. Se re-amplia con Real-ESRGAN, igual que el borrado
    # generativo, para que la nitidez combine con el resto de la foto.
    out = _sharpen_patch(out, cw, ch)
    blend = cv2.GaussianBlur((mcrop > 0).astype(np.float32), (0, 0), 4)[..., None]
    img[y0:y1, x0:x1] = crop * (1.0 - blend) + out * blend


def inpaint(img, mask, progress_cb=None):
    """img: float32 RGB 0..1; mask: uint8 (255 = borrar).
    Devuelve una copia con las zonas marcadas rellenadas."""
    if mask.max() == 0:
        return img
    result = img.copy()
    h, w = img.shape[:2]

    # cada zona conexa se procesa con su propio contexto alrededor
    n_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    for i in range(1, n_labels):
        ys, xs = np.where(labels == i)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        region_mask = np.where(labels == i, mask, 0)
        if not model_available():
            # sin modelo descargado: relleno clasico (mejor que nada)
            pad = 24
            _fast_inpaint_region(result, region_mask,
                                 max(y0 - pad, 0), min(y1 + pad + 1, h),
                                 max(x0 - pad, 0), min(x1 + pad + 1, w))
        else:
            # siempre LaMa, tambien en manchas pequenas: el relleno clasico
            # dejaba un borron liso que desentona en piel o zonas con grano
            side = max(y1 - y0, x1 - x0)
            pad = max(int(side * 0.8), 96)
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            half = min(max(side // 2 + pad, 192), max(h, w) // 2)
            ry0, ry1 = max(cy - half, 0), min(cy + half, h)
            rx0, rx1 = max(cx - half, 0), min(cx + half, w)
            _inpaint_region(result, region_mask, ry0, ry1, rx0, rx1)
        if progress_cb:
            progress_cb(i / (n_labels - 1))
    return np.clip(result, 0.0, 1.0)
