"""Reduccion de ruido con IA: SCUNet sobre ONNX Runtime + CUDA (RTX 4070).

Historial de decisiones (para no repetir el camino):
- SCUNet + DirectML: crash nativo (las ops de transformer rompen DirectML).
- NAFNet + DirectML/CPU/CUDA: ese export esta roto, explota con ruido fuerte
  (la "estatica" en fotos oscuras era el modelo, no el backend).
- SCUNet + CUDA: estable y de la mejor calidad. CPU como respaldo.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path.home() / ".photoraw" / "models"
DENOISE_MODEL = MODEL_DIR / "SCUNet-PSNR.onnx"
DENOISE_URL = ("https://huggingface.co/deepghs/image_restoration"
               "/resolve/main/SCUNet-PSNR.onnx")

_session = None


def model_available():
    return DENOISE_MODEL.exists() and DENOISE_MODEL.stat().st_size > 10_000_000


def download_model(progress_cb=None):
    """Descarga el modelo (~91 MB). progress_cb recibe 0..1."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DENOISE_MODEL.with_suffix(".onnx.tmp")

    def hook(blocks, block_size, total):
        if progress_cb and total > 0:
            progress_cb(min(blocks * block_size / total, 1.0))

    urllib.request.urlretrieve(DENOISE_URL, tmp, hook)
    tmp.replace(DENOISE_MODEL)


_cpu_session = None


def _make_options():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # oculta avisos internos del modelo
    return opts


def _providers():
    import onnxruntime as ort
    available = ort.get_available_providers()
    order = []
    if "CUDAExecutionProvider" in available:
        # HEURISTIC: elige los algoritmos de convolucion sin la calibracion
        # exhaustiva de cuDNN, que congelaba el primer uso de cada modelo
        # (Real-ESRGAN podia quedarse ~minutos "pensando" sin avisar)
        order.append(("CUDAExecutionProvider",
                      {"cudnn_conv_algo_search": "HEURISTIC"}))
    return order + ["CPUExecutionProvider"]


def _setup_dll_paths():
    """Anade las carpetas bin de los paquetes nvidia (pip) a la ruta de DLLs.
    cuDNN carga sus sublibrerias por PATH y preload_dlls no cubre eso."""
    import os
    import site
    roots = list(site.getsitepackages())
    roots.append(site.getusersitepackages())
    for sp in roots:
        nv = Path(sp) / "nvidia"
        if nv.is_dir():
            for bin_dir in nv.glob("*/bin"):
                try:
                    os.add_dll_directory(str(bin_dir))
                except OSError:
                    pass
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        # Carga las DLL de CUDA/cuDNN instaladas via pip
        _setup_dll_paths()
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        _session = ort.InferenceSession(
            str(DENOISE_MODEL), sess_options=_make_options(),
            providers=_providers())
        # Calentamiento
        name = _session.get_inputs()[0].name
        _session.run(None, {name: np.zeros((1, 3, 128, 128), np.float32)})
    return _session


def _get_cpu_session():
    global _cpu_session
    if _cpu_session is None:
        import onnxruntime as ort
        _cpu_session = ort.InferenceSession(
            str(DENOISE_MODEL), sess_options=_make_options(),
            providers=["CPUExecutionProvider"])
    return _cpu_session


def _tile_is_bad(x_in, y_out):
    """Detecta salidas corruptas de la GPU: NaN, un resultado que no se parece
    a la entrada, o MAS ruido de alta frecuencia del que entro (quitar ruido
    nunca puede aumentarlo; la estatica corrupta lo dispara)."""
    if not np.isfinite(y_out).all():
        return True
    if float(np.abs(y_out - x_in).mean()) > 0.15:
        return True
    lap_in = np.abs(cv2.Laplacian(x_in[0].mean(axis=0), cv2.CV_32F)).mean()
    lap_out = np.abs(cv2.Laplacian(y_out[0].mean(axis=0), cv2.CV_32F)).mean()
    return float(lap_out) > float(lap_in) * 1.5 + 1e-4


def _run_tile(x_in, input_name):
    """Ejecuta un mosaico con verificacion: GPU -> reintento -> CPU."""
    sess = _get_session()
    y = sess.run(None, {input_name: x_in})[0]
    if _tile_is_bad(x_in, y):
        y = sess.run(None, {input_name: x_in})[0]  # reintento en GPU
    if _tile_is_bad(x_in, y):
        y = _get_cpu_session().run(None, {input_name: x_in})[0]
    return y


def gpu_in_use():
    try:
        return _get_session().get_providers()[0] != "CPUExecutionProvider"
    except Exception:
        return False


def denoise(img, tile=512, overlap=48, progress_cb=None):
    """img: float32 RGB 0..1. Procesa por mosaicos para no agotar la VRAM.
    Devuelve float32 RGB 0..1."""
    sess = _get_session()
    input_name = sess.get_inputs()[0].name
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    weight = np.zeros((h, w, 1), np.float32)
    step = tile - overlap
    ys = list(range(0, h, step))
    xs = list(range(0, w, step))
    total = len(ys) * len(xs)
    done = 0

    for y in ys:
        for x in xs:
            y1, x1 = min(y + tile, h), min(x + tile, w)
            y0, x0 = max(y1 - tile, 0), max(x1 - tile, 0)
            patch = img[y0:y1, x0:x1]
            ph, pw = patch.shape[:2]
            # la red reduce resolucion por 2 varias veces: lados multiplos de 64
            pad_h = (64 - ph % 64) % 64
            pad_w = (64 - pw % 64) % 64
            if pad_h or pad_w:
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            x_in = np.ascontiguousarray(patch.transpose(2, 0, 1)[None], dtype=np.float32)
            y_out = _run_tile(x_in, input_name)[0]
            y_out = y_out.transpose(1, 2, 0)[:ph, :pw]
            out[y0:y1, x0:x1] += y_out
            weight[y0:y1, x0:x1] += 1.0
            done += 1
            if progress_cb:
                progress_cb(done / total)

    return np.clip(out / np.maximum(weight, 1e-6), 0.0, 1.0)
