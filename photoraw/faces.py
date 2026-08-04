"""Retoque de rostros con IA: YuNet detecta las caras y GFPGAN v1.4 las
restaura en la GPU (CUDA). Cada cara se alinea a 512x512, se procesa y se
reintegra en la foto con una fusion suave.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from photoraw.ai import MODEL_DIR, _make_options, _providers, _setup_dll_paths

# Dos restauradores: GFPGAN embellece mas pero alucina con oclusiones
# (gafas, manos); CodeFormer es mas fiel a la foto original.
FACE_MODELS = {
    "GFPGAN": (MODEL_DIR / "gfpgan_1.4.onnx",
               "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx"),
    "CodeFormer": (MODEL_DIR / "codeformer.onnx",
                   "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx"),
}
YUNET_MODEL = MODEL_DIR / "yunet_face.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")

_model_name = "GFPGAN"


def set_model(name):
    global _model_name
    if name in FACE_MODELS:
        _model_name = name


def current_model():
    return _model_name

# Plantilla clasica de GFPGAN/FFHQ a 512 px: ojo izq, ojo der, nariz, boca izq, boca der
TEMPLATE_512 = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.19360],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
], np.float32)

_sessions = {}
_detector = None


def models_available():
    model_path = FACE_MODELS[_model_name][0]
    return (model_path.exists() and model_path.stat().st_size > 10_000_000
            and YUNET_MODEL.exists() and YUNET_MODEL.stat().st_size > 100_000)


def download_models(progress_cb=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path, model_url = FACE_MODELS[_model_name]
    for path, url in ((YUNET_MODEL, YUNET_URL), (model_path, model_url)):
        if path.exists():
            continue
        tmp = path.with_suffix(".onnx.tmp")

        def hook(blocks, block_size, total):
            if progress_cb and total > 0:
                progress_cb(min(blocks * block_size / total, 1.0))

        urllib.request.urlretrieve(url, tmp, hook)
        tmp.replace(path)


def _get_session():
    if _model_name not in _sessions:
        import onnxruntime as ort
        _setup_dll_paths()
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        _sessions[_model_name] = ort.InferenceSession(
            str(FACE_MODELS[_model_name][0]), sess_options=_make_options(),
            providers=_providers())
    return _sessions[_model_name]


def _detect_faces(bgr):
    """Devuelve lista de landmarks (5 puntos por cara) en coordenadas de la imagen."""
    global _detector
    h, w = bgr.shape[:2]
    # detectar sobre una copia reducida acelera y es igual de fiable
    scale = min(1280.0 / max(h, w), 1.0)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else bgr
    if _detector is None:
        _detector = cv2.FaceDetectorYN_create(str(YUNET_MODEL), "", (320, 320),
                                              score_threshold=0.7)
    _detector.setInputSize((small.shape[1], small.shape[0]))
    _found, faces = _detector.detect(small)
    result = []
    if faces is not None:
        for f in faces:
            pts = f[4:14].reshape(5, 2) / scale
            result.append(pts.astype(np.float32))
    return result


def enhance_faces(img, progress_cb=None):
    """img: float32 RGB 0..1. Restaura todas las caras detectadas.
    Devuelve (imagen resultante, numero de caras retocadas)."""
    bgr = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    faces = _detect_faces(bgr)
    if not faces:
        return img, 0

    sess = _get_session()
    inputs = sess.get_inputs()
    input_name = inputs[0].name
    # CodeFormer acepta un peso de fidelidad extra (0 = mas bello, 1 = mas fiel)
    extra = {}
    if len(inputs) > 1:
        dtype = np.float64 if "double" in inputs[1].type else np.float32
        extra[inputs[1].name] = np.array([0.8], dtype=dtype)
    h, w = img.shape[:2]
    out = img.copy()

    # mascara suave 512: elipse central desvanecida hacia los bordes
    mask = np.zeros((512, 512), np.float32)
    cv2.ellipse(mask, (256, 268), (190, 230), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 24)

    for i, pts in enumerate(faces):
        matrix, _ = cv2.estimateAffinePartial2D(pts, TEMPLATE_512, method=cv2.LMEDS)
        if matrix is None:
            continue
        crop = cv2.warpAffine(out, matrix, (512, 512), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
        x_in = (crop.transpose(2, 0, 1)[None] - 0.5) / 0.5  # RGB [-1, 1]
        y_out = sess.run(None, {input_name: x_in.astype(np.float32), **extra})[0][0]
        restored = np.clip(y_out.transpose(1, 2, 0) * 0.5 + 0.5, 0.0, 1.0)
        if not np.isfinite(restored).all():
            continue

        inv = cv2.invertAffineTransform(matrix)
        back = cv2.warpAffine(restored, inv, (w, h), flags=cv2.INTER_LINEAR)
        back_mask = cv2.warpAffine(mask, inv, (w, h), flags=cv2.INTER_LINEAR)
        back_mask = np.clip(back_mask, 0.0, 1.0)[..., None]
        out = out * (1.0 - back_mask) + back * back_mask
        if progress_cb:
            progress_cb((i + 1) / len(faces))

    return np.clip(out, 0.0, 1.0), len(faces)
