"""Cache en disco de vistas previas y miniaturas (estilo Lightroom).

La primera vez que se abre una foto se decodifica el RAW (lo lento) y el
resultado queda guardado aqui; las siguientes veces carga en milisegundos.
La clave incluye la fecha y el tamano del archivo original, asi que si la
foto cambia el cache se invalida solo. El tamano total se limita borrando
lo menos usado (LRU).
"""
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from photoraw import loader

CACHE_DIR = Path.home() / ".photoraw" / "cache"
LIMIT_FILE = Path.home() / ".photoraw" / "cache_limit.json"
DEFAULT_MAX_GB = 4
THUMB_SIDE = 320
# GB "de los que pone en el disco duro" (1000, no 1024), para que el tope que
# eliges en la ventana sea el numero que luego se te ensena
GB = 1_000_000_000

# Categorias del cache, por extension. Sirven para el desglose que se ensena
# en la ventana de Modelos de IA.
KINDS = {
    ".jpg": "miniaturas de la tira",
    ".npy": "fotos ya decodificadas",
    ".npz": "trabajos de IA y revelados",
}


def _load_limit():
    try:
        gb = float(json.loads(LIMIT_FILE.read_text("utf-8"))["max_gb"])
        return max(int(gb * GB), 256 * 1_000_000)   # nunca menos de 256 MB
    except Exception:
        return DEFAULT_MAX_GB * GB


MAX_BYTES = _load_limit()


def limit_gb():
    """Tope del cache en GB."""
    return MAX_BYTES / GB


def set_limit_gb(gb):
    """Cambia el tope y, si el cache ya lo pasa, recorta ya mismo."""
    global MAX_BYTES
    MAX_BYTES = max(int(float(gb) * GB), 256 * 1_000_000)
    try:
        LIMIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        LIMIT_FILE.write_text(json.dumps({"max_gb": float(gb)}), encoding="utf-8")
    except Exception:
        pass
    _evict_if_needed()


def stats():
    """(bytes usados, tope, {categoria: (n archivos, bytes)}) para la interfaz."""
    por_tipo = {}
    total = 0
    try:
        for f in CACHE_DIR.iterdir():
            if not f.is_file():
                continue
            size = f.stat().st_size
            total += size
            nombre = KINDS.get(f.suffix.lower(), "Otros")
            n, b = por_tipo.get(nombre, (0, 0))
            por_tipo[nombre] = (n + 1, b + size)
    except OSError:
        pass
    return total, MAX_BYTES, por_tipo


def _entry(path, kind, ext):
    """Archivo de cache para `path`; None si el original no es accesible."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    tag = f"{Path(path).resolve()}|{st.st_mtime_ns}|{st.st_size}|{kind}"
    name = hashlib.sha1(tag.encode("utf-8", "replace")).hexdigest()[:24]
    return CACHE_DIR / f"{name}.{ext}"


def _touch(p):
    """Marca el archivo como recien usado (para el borrado LRU)."""
    try:
        os.utime(p, None)
    except OSError:
        pass


def _write_atomic(entry, writer):
    """Escribe via archivo temporal para no dejar caches a medias."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = entry.with_name(entry.name + ".tmp")
        with open(tmp, "wb") as f:
            writer(f)
        tmp.replace(entry)
        _evict_if_needed()
    except Exception:
        pass


def _evict_if_needed():
    try:
        files = []
        for f in CACHE_DIR.iterdir():
            if f.is_file():
                st = f.stat()
                files.append((st.st_mtime, st.st_size, f))
    except OSError:
        return
    total = sum(size for _m, size, _f in files)
    if total <= MAX_BYTES:
        return
    for _m, size, f in sorted(files):  # los mas antiguos primero
        try:
            f.unlink()
            total -= size
        except OSError:
            pass
        if total <= MAX_BYTES:
            return


def load_preview(path, max_side=2200):
    """Como loader.load_preview pero con cache en disco. Se guarda a 16 bits
    (uint16) para no perder el margen real que da la decodificacion RAW."""
    # El sufijo "icc" invalida las vistas previas de JPG/HEIC guardadas antes
    # de que la conversion de perfil de color funcionara. Los RAW no pasan por
    # ese camino, asi que conservan su cache (decodificarlos de nuevo es lento)
    kind = f"prev{max_side}16" if loader.is_raw(path) else f"prev{max_side}16icc"
    entry = _entry(path, kind, "npy")
    if entry is not None and entry.exists():
        try:
            arr = np.load(entry)
            _touch(entry)
            return np.ascontiguousarray(arr).astype(np.float32) / 65535.0
        except Exception:
            pass
    base = loader.load_preview(path, max_side)
    if entry is not None:
        data = (np.clip(base, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        _write_atomic(entry, lambda f: np.save(f, data))
    return base


def result_key(*parts):
    """Huella corta de los ingredientes de un resultado IA (la imagen de
    entrada, los trazos, el modelo...). Si cualquier ingrediente cambia,
    la huella cambia y el resultado guardado deja de valer."""
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(p.tobytes())
        else:
            h.update(repr(p).encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def load_result(path, kind):
    """Resultado IA cacheado como (array, meta) o None si no esta.
    Los float 0..1 se guardaron como uint8 (invisible: la fuente es de 8 bits)."""
    entry = _entry(path, kind, "npz")
    if entry is None or not entry.exists():
        return None
    try:
        with np.load(entry) as z:
            arr, f01, meta = z["arr"], bool(z["f01"]), int(z["meta"])
        _touch(entry)
        if f01:
            arr = np.ascontiguousarray(arr).astype(np.float32) / 255.0
        return arr, meta
    except Exception:
        return None


def save_result(path, kind, arr, meta=0):
    """Guarda un resultado IA (float32 0..1 o uint8) para no recalcularlo."""
    if arr is None:
        return
    f01 = arr.dtype != np.uint8
    if f01:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    entry = _entry(path, kind, "npz")
    if entry is not None:
        _write_atomic(entry, lambda f: np.savez(f, arr=arr, f01=f01, meta=meta))


def thumb_jpeg(path):
    """Bytes JPEG de la miniatura (320 px), con cache en disco.
    Devuelve None si la foto no se puede leer."""
    # las miniaturas de RAW salen del JPEG incrustado y no cambian; las de
    # JPG/HEIC si, porque ahora se les aplica el perfil de color
    entry = _entry(path, "thumb2" if loader.is_raw(path) else "thumb2icc", "jpg")
    if entry is not None and entry.exists():
        try:
            data = entry.read_bytes()
            _touch(entry)
            return data
        except OSError:
            pass

    arr = None
    data, flip = loader.load_thumb_bytes(path)  # JPEG incrustado del RAW (rapido)
    if data is not None:
        arr = loader.decode_thumb(data, flip)
    if arr is None:
        arr = loader.load_thumb_array(path, THUMB_SIDE)
    if arr is None:
        return None

    h, w = arr.shape[:2]
    if max(h, w) > THUMB_SIDE:
        scale = THUMB_SIDE / max(h, w)
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    data = buf.tobytes()
    if entry is not None:
        _write_atomic(entry, lambda f: f.write(data))
    return data


def cache_size():
    """Tamano total del cache en bytes."""
    try:
        return sum(f.stat().st_size for f in CACHE_DIR.iterdir() if f.is_file())
    except OSError:
        return 0


def clear():
    """Vacia el cache por completo."""
    try:
        for f in CACHE_DIR.iterdir():
            if f.is_file():
                f.unlink()
    except OSError:
        pass
