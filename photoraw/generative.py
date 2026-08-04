"""Borrado generativo: difusion (Stable Diffusion inpainting) en la GPU.

Reconstruye de verdad el fondo detras de un objeto grande (una persona, un
coche...) inventando detalle plausible, cosa que el corrector clasico (LaMa)
no puede. Usa Realistic Vision 5.1 inpainting exportado a ONNX fp16, con el
pipeline escrito a mano sobre onnxruntime: VAE codifica la zona, la U-Net
la "revela" del ruido en ~25 pasos condicionada por la imagen enmascarada,
y el VAE decodifica el resultado. Sin texto (prompt vacio): el modelo solo
continua el fondo. Semilla fija: el mismo borrado siempre da lo mismo.

El modelo trabaja a 512x512, asi que (como el corrector) se recorta la zona
con contexto, se procesa y se funde de vuelta con borde suave.
"""
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from photoraw.ai import (MODEL_DIR, Cancelled, _make_options, _providers,
                         _setup_dll_paths)

SD_DIR = MODEL_DIR / "sd_inpaint"
SD_BASE_URL = ("https://huggingface.co/RanaLLC/"
               "Realistic_Vision_V5.1-inpainting-onnx-fp16/resolve/main/")
SD_FILES = [
    ("unet/model.onnx", "unet.onnx", 1_500_000_000),
    ("text_encoder/model.onnx", "text_encoder.onnx", 200_000_000),
    ("vae_encoder/model.onnx", "vae_encoder.onnx", 50_000_000),
    ("vae_decoder/model.onnx", "vae_decoder.onnx", 80_000_000),
    ("tokenizer/vocab.json", "vocab.json", 100_000),
    ("tokenizer/merges.txt", "merges.txt", 100_000),
    ("scheduler/scheduler_config.json", "scheduler_config.json", 10),
]

STEPS = 25          # pasos de difusion (calidad/velocidad)
BOS, EOS = 49406, 49407  # tokens de inicio/fin del CLIP
LATENT_SCALE = 0.18215
GUIDANCE = 4.5      # cuanto obedece al texto (CFG)
# Sin guia el modelo tiende a "inventar" objetos en medio del relleno.
# Estas instrucciones fijas lo obligan a continuar solo el fondo:
PROMPT = "background, empty scene"
NEGATIVE = ("person, people, human, face, object, animal, text, watermark, "
            "logo, blurry, artifact, distortion")

_sessions = {}
_tokenizer = None
_emb_cache = {}


def model_available():
    return all((SD_DIR / local).exists()
               and (SD_DIR / local).stat().st_size >= min_size
               for _r, local, min_size in SD_FILES)


def _download_resume(url, dst, hook=None):
    """Descarga con reanudacion: si la conexion se corta, reintenta y sigue
    donde quedo (imprescindible para archivos de este tamano)."""
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    for _try in range(30):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        size = 0
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and getattr(r, "status", 200) != 206:
                    have = 0  # el servidor no reanuda: desde el principio
                size = have + int(r.headers.get("Content-Length") or 0)
                with open(tmp, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if hook:
                            hook(have, size)
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(3)
            continue
        if size and have >= size:
            tmp.replace(dst)
            return
        time.sleep(3)
    raise OSError(f"No se pudo completar la descarga de {dst.name}")


def download_model(progress_cb=None):
    """Descarga ~2 GB (una sola vez). progress_cb recibe 0..1 global."""
    SD_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(s for _r, _l, s in SD_FILES)
    done = 0
    for remote, local, min_size in SD_FILES:
        dst = SD_DIR / local
        if not (dst.exists() and dst.stat().st_size >= min_size):

            def hook(have, size, base=done, want=min_size):
                if progress_cb and total > 0:
                    frac = have / size if size else 0.0
                    progress_cb(min((base + frac * want) / total, 1.0))

            _download_resume(SD_BASE_URL + remote, dst, hook)
        done += min_size
    if progress_cb:
        progress_cb(1.0)


def encode_map(wmap):
    """Mapa 0..1 -> PNG gris en base64 (lado largo <= 1024) para guardar la
    operacion de borrado en el JSON de ediciones. La mascara puede borrarse
    despues sin perder el borrado, y al exportar se re-aplica identica."""
    h, w = wmap.shape[:2]
    scale = min(1024.0 / max(h, w), 1.0)
    small = wmap
    if scale < 1.0:
        small = cv2.resize(wmap, (max(int(w * scale), 1),
                                  max(int(h * scale), 1)),
                           interpolation=cv2.INTER_AREA)
    _ok, buf = cv2.imencode(".png",
                            (np.clip(small, 0.0, 1.0) * 255).astype(np.uint8))
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_map(data, h, w):
    """PNG base64 -> mapa float32 0..1 reescalado a h x w."""
    raw = np.frombuffer(base64.b64decode(data), np.uint8)
    m = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = m.astype(np.float32) / 255.0
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return m


def _np_dtype(ort_type):
    return {"tensor(float)": np.float32, "tensor(float16)": np.float16,
            "tensor(int64)": np.int64, "tensor(int32)": np.int32,
            "tensor(double)": np.float64}.get(ort_type, np.float32)


def _get_session(name):
    if name not in _sessions:
        import onnxruntime as ort
        _setup_dll_paths()
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        _sessions[name] = ort.InferenceSession(
            str(SD_DIR / f"{name}.onnx"), sess_options=_make_options(),
            providers=_providers())
    return _sessions[name]


def _run1(name, x):
    """Sesion de una sola entrada / una salida, con el dtype que pida."""
    sess = _get_session(name)
    inp = sess.get_inputs()[0]
    x = x.astype(_np_dtype(inp.type), copy=False)
    return sess.run(None, {inp.name: x})[0]


def _get_tokenizer():
    """Tokenizador BPE del CLIP (suficiente para nuestras frases fijas en
    ingles: minusculas, palabras separadas por espacios/comas)."""
    global _tokenizer
    if _tokenizer is None:
        vocab = json.loads((SD_DIR / "vocab.json").read_text("utf-8"))
        merges = {}
        lines = (SD_DIR / "merges.txt").read_text("utf-8").splitlines()
        for rank, line in enumerate(lines[1:]):  # la primera es cabecera
            parts = line.split()
            if len(parts) == 2:
                merges[tuple(parts)] = rank
        _tokenizer = (vocab, merges)
    return _tokenizer


def _bpe_word(word, vocab, merges):
    """Aplica las fusiones BPE a una palabra (con marca de fin </w>)."""
    tokens = list(word[:-1]) + [word[-1] + "</w>"]
    while len(tokens) > 1:
        pairs = [(merges.get((a, b), 1 << 30), i)
                 for i, (a, b) in enumerate(zip(tokens, tokens[1:]))]
        rank, i = min(pairs)
        if rank >= 1 << 30:
            break
        tokens[i:i + 2] = [tokens[i] + tokens[i + 1]]
    return [vocab[t] for t in tokens if t in vocab]


def _prompt_embedding(text):
    """Texto -> embedding del CLIP (77 tokens), cacheado."""
    if text not in _emb_cache:
        vocab, merges = _get_tokenizer()
        ids = [BOS]
        for word in text.lower().replace(",", " ").split():
            ids.extend(_bpe_word(word, vocab, merges))
        ids = ids[:76] + [EOS] * (77 - min(len(ids), 76))
        _emb_cache[text] = _run1("text_encoder", np.array([ids]))
    return _emb_cache[text]


def _scheduler_alphas():
    cfg = json.loads((SD_DIR / "scheduler_config.json").read_text("utf-8"))
    t_train = int(cfg.get("num_train_timesteps", 1000))
    b0 = float(cfg.get("beta_start", 0.00085))
    b1 = float(cfg.get("beta_end", 0.012))
    if cfg.get("beta_schedule", "scaled_linear") == "scaled_linear":
        betas = np.linspace(b0 ** 0.5, b1 ** 0.5, t_train,
                            dtype=np.float64) ** 2
    else:
        betas = np.linspace(b0, b1, t_train, dtype=np.float64)
    return np.cumprod(1.0 - betas), t_train, int(cfg.get("steps_offset", 1))


def _diffuse(img_s, mask_s, seed, progress_cb=None):
    """img_s: float32 RGB 0..1 (cuadrada, lado multiplo de 64); mask_s:
    float32 0..1 (1=rellenar). Devuelve el relleno decodificado (0..1).
    El UNet exportado acepta cualquier tamano: a mas lado, mas fina la
    textura reconstruida (y mas tiempo/VRAM)."""
    side = img_s.shape[0]
    s8 = side // 8
    acp, t_train, offset = _scheduler_alphas()
    ratio = t_train // STEPS
    timesteps = (np.arange(STEPS) * ratio)[::-1] + offset

    # guia por texto: "continua el fondo" frente a "nada de objetos/personas"
    emb = np.concatenate([_prompt_embedding(NEGATIVE),
                          _prompt_embedding(PROMPT)], axis=0)
    pm1 = img_s.astype(np.float32) * 2.0 - 1.0           # a rango -1..1
    hard = (mask_s > 0.5).astype(np.float32)
    masked = pm1 * (1.0 - hard[..., None])
    lat_masked = _run1("vae_encoder",
                       masked.transpose(2, 0, 1)[None]).astype(np.float32)
    lat_masked *= LATENT_SCALE
    mask_lat = cv2.resize(hard, (s8, s8),
                          interpolation=cv2.INTER_NEAREST)[None, None]

    unet = _get_session("unet")
    u_in = {i.name: i for i in unet.get_inputs()}
    s_dt = _np_dtype(u_in["sample"].type)
    t_dt = _np_dtype(u_in["timestep"].type)
    e_dt = _np_dtype(u_in["encoder_hidden_states"].type)
    emb = emb.astype(e_dt, copy=False)

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, 4, s8, s8)).astype(np.float32)
    for i, t in enumerate(timesteps):
        sample = np.concatenate(
            [x, mask_lat.astype(np.float32), lat_masked], axis=1)
        if side <= 512:
            # lote de 2: sin texto (negativo) y con texto, en una pasada
            eps2 = unet.run(None, {
                "sample": np.repeat(sample, 2, axis=0).astype(s_dt),
                "timestep": np.array([t], dtype=t_dt),
                "encoder_hidden_states": emb,
            })[0].astype(np.float32)
        else:
            # en grande, una pasada por vez: el lote de 2 duplica el pico
            # de VRAM de la atencion y a 768 se queda sin memoria
            eps2 = np.concatenate([unet.run(None, {
                "sample": sample.astype(s_dt),
                "timestep": np.array([t], dtype=t_dt),
                "encoder_hidden_states": emb[k:k + 1],
            })[0].astype(np.float32) for k in (0, 1)], axis=0)
        eps = eps2[0:1] + GUIDANCE * (eps2[1:2] - eps2[0:1])
        # paso DDIM (deterministico): estima la imagen limpia y re-proyecta
        a_t = float(acp[t])
        t_prev = t - ratio
        a_prev = float(acp[t_prev]) if t_prev >= 0 else 1.0
        x0 = (x - np.sqrt(1.0 - a_t) * eps) / np.sqrt(a_t)
        x = np.sqrt(a_prev) * x0 + np.sqrt(1.0 - a_prev) * eps
        if progress_cb:
            progress_cb((i + 1) / STEPS)

    if side > 512:
        # el UNet ya no hace falta y tras correr en grande su reserva de
        # VRAM es enorme: se suelta para que el decodificador tenga sitio
        # (tambien las referencias locales, o la sesion sigue viva)
        _sessions.pop("unet", None)
        del unet, u_in
    out = _run1("vae_decoder", (x / LATENT_SCALE)).astype(np.float32)
    out = out[0].transpose(1, 2, 0)
    return np.clip(out * 0.5 + 0.5, 0.0, 1.0)


def _sharpen_patch(out_s, cw, ch, progress_cb=None):
    """Si el parche reconstruido va a estirarse bastante, se re-amplia con
    Real-ESRGAN (textura nitida) en vez de con interpolacion (borrosa).
    Asi el relleno no desentona con el detalle de alrededor."""
    if max(cw, ch) > max(out_s.shape[:2]) * 1.2:
        try:
            from photoraw import upscale
            if upscale.model_available():
                big = upscale.upscale(out_s, scale=4,
                                      progress_cb=progress_cb)
                big = big.astype(np.float32) / 255.0
                return cv2.resize(big, (cw, ch), interpolation=cv2.INTER_AREA)
        except Cancelled:
            raise  # Detener no es un fallo del afinado: aborta de verdad
        except Exception:
            pass  # sin superresolucion: se estira normal (mas blando)
    return cv2.resize(out_s, (cw, ch), interpolation=cv2.INTER_LINEAR)


def _free_vram_mb():
    """VRAM libre segun nvidia-smi, o None si no se puede saber."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _pick_size(win):
    """Resolucion de trabajo para una zona de `win` px: la mayor que quepa
    en la VRAM libre AHORA (la GPU se comparte con otras apps; si se
    desborda, Windows tira de RAM y todo va 10-100x mas lento)."""
    free = _free_vram_mb()
    if free is not None:
        # medido en la RTX 4070 de 8 GB: 640 corre limpio con ~7.7 GB
        # libres (19.8 s); 768 desborda y tarda minutos. 768 queda para
        # tarjetas con mas memoria.
        for size, need in ((768, 10000), (640, 6000)):
            if win > size * 0.85 and free >= need:
                return size
    return 512


def release_sessions():
    """Suelta los modelos de difusion (~2 GB de VRAM). El borrado es
    ocasional y dejarlos cargados frena al resto de la IA: con ellos
    residentes, el afinado Real-ESRGAN pasaba de ~2 s a minutos. Se
    recargan solos en el proximo borrado (unos segundos)."""
    _sessions.clear()


def erase(img, mask_map, seed=0, progress_cb=None):
    """img: float32 RGB 0..1; mask_map: float32 0..1 (que borrar).
    Devuelve una copia con las zonas marcadas reconstruidas con difusion.
    Cada zona conexa se procesa con su propio contexto alrededor.

    Dos fases: primero la difusion de todas las zonas (85% del progreso),
    despues se liberan los modelos de difusion y se afina la textura con
    Real-ESRGAN (15% restante) ya sin presion de VRAM."""
    m8 = ((mask_map > 0.5).astype(np.uint8)) * 255
    if m8.max() == 0:
        return img
    result = img.copy()
    h, w = img.shape[:2]
    n_labels, labels = cv2.connectedComponents((m8 > 0).astype(np.uint8))
    regions = [i for i in range(1, n_labels)
               if (labels == i).sum() >= 64]  # motas sueltas fuera
    n = len(regions)
    done = [0.0]  # unidades de trabajo completadas (por zona: 0.85 + 0.15)

    def report(extra):
        if progress_cb:
            progress_cb(min((done[0] + extra) / max(n, 1), 1.0))

    pend = []
    for idx, i in enumerate(regions):
        ys, xs = np.where(labels == i)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        side = max(y1 - y0, x1 - x0)
        # ventana cuadrada con contexto alrededor de la zona
        half = min(max(int(side * 0.85), 224), max(h, w) // 2)
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        ry0, ry1 = max(cy - half, 0), min(cy + half, h)
        rx0, rx1 = max(cx - half, 0), min(cx + half, w)
        crop = result[ry0:ry1, rx0:rx1].copy()
        mcrop = np.where(labels[ry0:ry1, rx0:rx1] == i,
                         mask_map[ry0:ry1, rx0:rx1], 0.0)
        ch, cw = crop.shape[:2]

        # resolucion de trabajo: a mas lado, mas fina la textura que puede
        # inventar la difusion (pasto, agua, pelo), pero cuesta VRAM. Se
        # elige segun la memoria libre AHORA, con escalera de reintentos.
        sizes = [s for s in (768, 640, 512)
                 if s <= _pick_size(max(ch, cw))]
        for size in sizes:
            crop_s = cv2.resize(crop, (size, size),
                                interpolation=cv2.INTER_AREA)
            mask_s = cv2.resize((mcrop > 0.5).astype(np.float32),
                                (size, size),
                                interpolation=cv2.INTER_NEAREST)
            mask_s = cv2.dilate(mask_s, np.ones((9, 9), np.uint8))
            try:
                out_s = _diffuse(crop_s, mask_s, seed + i,
                                 lambda p: report(p * 0.85))
                break
            except Cancelled:
                raise  # el usuario paro: no bajar de tamano y reintentar
            except Exception:
                if size == 512:
                    raise
                release_sessions()  # sin VRAM para este lado: uno menor
        done[0] += 0.85

        blend = cv2.resize(mask_s, (cw, ch), interpolation=cv2.INTER_LINEAR)
        blend = cv2.GaussianBlur(blend, (0, 0), max(max(ch, cw) / 256.0, 2.0))
        blend = np.clip(blend, 0.0, 1.0)[..., None]
        # fusion provisional (relleno blando): las zonas siguientes ya ven
        # esta reconstruccion como contexto (reescalada no se nota el filo)
        soft = cv2.resize(out_s, (cw, ch), interpolation=cv2.INTER_LINEAR)
        result[ry0:ry1, rx0:rx1] = crop * (1.0 - blend) + soft * blend
        pend.append((ry0, ry1, rx0, rx1, crop, blend, out_s))

    release_sessions()  # deja la VRAM libre para el afinado y la demas IA

    for ry0, ry1, rx0, rx1, crop, blend, out_s in pend:
        ch, cw = crop.shape[:2]
        out = _sharpen_patch(out_s, cw, ch,
                             progress_cb=lambda p: report(p * 0.15))
        done[0] += 0.15
        result[ry0:ry1, rx0:rx1] = crop * (1.0 - blend) + out * blend
        report(0.0)
    return np.clip(result, 0.0, 1.0)
