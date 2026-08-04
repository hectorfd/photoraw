"""Mascaras de retrato: BiSeNet divide cada cara en zonas (piel, cejas,
ojos, labios, dientes, pelo) para usarlas como mascaras locales.

YuNet (el mismo detector del retoque de rostros) encuentra las caras; cada
una se alinea a 512x512 como en CelebAMask-HQ, BiSeNet la clasifica pixel a
pixel en 19 zonas y las etiquetas se reintegran en un mapa del tamano de la
foto. Las zonas siguen la numeracion de CelebAMask-HQ (ver
engine.FACE_PART_IDS para la traduccion zona -> mascara).
"""
import urllib.request

import cv2
import numpy as np

from photoraw import faces
from photoraw.ai import MODEL_DIR, _make_options, _providers, _setup_dll_paths

BISENET_MODEL = MODEL_DIR / "bisenet_face.onnx"
BISENET_URL = ("https://huggingface.co/facefusion/models-3.0.0/resolve/main/"
               "bisenet_resnet_34.onnx")

_session = None


def model_available():
    return (BISENET_MODEL.exists() and BISENET_MODEL.stat().st_size > 10_000_000
            and faces.YUNET_MODEL.exists()
            and faces.YUNET_MODEL.stat().st_size > 100_000)


def download_model(progress_cb=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in ((faces.YUNET_MODEL, faces.YUNET_URL),
                      (BISENET_MODEL, BISENET_URL)):
        if path.exists():
            continue
        tmp = path.with_suffix(".onnx.tmp")

        def hook(blocks, block_size, total):
            if progress_cb and total > 0:
                progress_cb(min(blocks * block_size / total, 1.0))

        urllib.request.urlretrieve(url, tmp, hook)
        tmp.replace(path)


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
            str(BISENET_MODEL), sess_options=_make_options(),
            providers=_providers())
    return _session


def parse_labels(img):
    """img: float32 RGB 0..1. Devuelve un mapa uint8 h x w con la zona de
    cada pixel (0 = nada) uniendo todas las caras detectadas, o un mapa de
    ceros si no hay caras."""
    h, w = img.shape[:2]
    labels = np.zeros((h, w), np.uint8)
    bgr = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8),
                       cv2.COLOR_RGB2BGR)
    pts_list = faces._detect_faces(bgr)
    if not pts_list:
        return labels
    sess = _get_session()
    input_name = sess.get_inputs()[0].name
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    for pts in pts_list:
        matrix, _ = cv2.estimateAffinePartial2D(pts, faces.TEMPLATE_512,
                                                method=cv2.LMEDS)
        if matrix is None:
            continue
        crop = cv2.warpAffine(np.clip(img, 0.0, 1.0), matrix, (512, 512),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
        x = (crop.astype(np.float32) - mean) / std
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)
        out = sess.run(None, {input_name: x})[0][0]      # 19 x 512 x 512
        face_labels = out.argmax(axis=0).astype(np.uint8)
        # las etiquetas vuelven a su sitio en la foto SIN interpolar
        # (interpolar mezclaria numeros de zonas vecinas)
        inv = cv2.invertAffineTransform(matrix)
        back = cv2.warpAffine(face_labels, inv, (w, h),
                              flags=cv2.INTER_NEAREST)
        np.copyto(labels, back, where=back > 0)
    return labels
