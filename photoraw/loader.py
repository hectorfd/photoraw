"""Carga de archivos RAW e imagenes normales."""
import io
from pathlib import Path
import numpy as np
import cv2
import rawpy
from PIL import Image, ImageOps

# HEIC/HEIF (fotos de iPhone y de moviles Android recientes): registra el
# decodificador en Pillow para que Image.open los abra como cualquier JPEG
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:
    HEIF_OK = False

RAW_EXTS = {".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".raf", ".rw2",
            ".orf", ".pef", ".srw", ".x3f", ".erf", ".kdc", ".3fr", ".iiq"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
HEIF_EXTS = {".heic", ".heif", ".hif"}
if HEIF_OK:
    IMG_EXTS |= HEIF_EXTS
ALL_EXTS = RAW_EXTS | IMG_EXTS


def is_raw(path):
    return Path(path).suffix.lower() in RAW_EXTS


def list_photos(folder):
    folder = Path(folder)
    files = [p for p in sorted(folder.iterdir(), key=lambda p: p.name.lower())
             if p.is_file() and p.suffix.lower() in ALL_EXTS]
    return files


def _resize_max(img, max_side):
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img
    scale = max_side / side
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _load_tiff16(path):
    """TIFF de 16 bits por canal como uint16 RGB, o None si no lo es.

    Pillow abre estos TIFF pero los baja a 8 bits sin avisar, y ahi se
    pierde justo el margen que hace util un archivo de 16 bits (el resultado
    de una fusion HDR, por ejemplo, que se revela estirando mucho sombras y
    luces). OpenCV si los lee enteros.
    """
    if Path(path).suffix.lower() not in (".tif", ".tiff"):
        return None
    try:
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if bgr is None or bgr.dtype != np.uint16 or bgr.ndim != 3:
        return None
    if bgr.shape[2] == 4:
        bgr = bgr[:, :, :3]
    elif bgr.shape[2] != 3:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_rgb(path, half_size):
    """Devuelve uint8 RGB (o uint16 en RAW y TIFF de 16 bits)."""
    path = Path(path)
    if is_raw(path):
        from photoraw import dng
        # Los DNG que fabrica la fusion HDR ya vienen expuestos: son el
        # resultado que se aprobo en la vista previa. Dejar que LibRaw les
        # estire el histograma como a un RAW recien salido de la camara los
        # aclara casi un paso entero (medido en IMG_3621-23: de 93 a 120 de
        # media), o sea que la foto no se abre como se guardo.
        propio = dng.es_nuestro(path)
        with rawpy.imread(str(path)) as raw:
            return raw.postprocess(use_camera_wb=True, half_size=half_size,
                                   output_bps=16, no_auto_bright=propio,
                                   highlight_mode=rawpy.HighlightMode.Blend)
    tiff16 = _load_tiff16(path)
    if tiff16 is not None:
        return tiff16
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        # Aplicar perfil ICC embebido (como hace Windows). Las fotos de
        # iPhone (HEIC y JPG) vienen en Display P3: sin esto los rojos y
        # verdes salen sobresaturados
        try:
            from PIL import ImageCms
            icc = im.info.get("icc_profile")
            if icc:
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")
                profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                srgb = ImageCms.createProfile("sRGB")
                im = ImageCms.profileToProfile(im, profile, srgb, outputMode="RGB")
        except Exception:
            pass
        return np.asarray(im.convert("RGB"))


def _to_float01(rgb):
    """uint8 o uint16 (RAW a 16 bits) -> float32 0..1."""
    peak = 65535.0 if rgb.dtype == np.uint16 else 255.0
    return np.ascontiguousarray(rgb).astype(np.float32) / peak


def load_preview(path, max_side=2200):
    """Imagen para editar en pantalla: float32 RGB 0..1, lado max limitado."""
    rgb = _load_rgb(path, half_size=True)
    rgb = _resize_max(rgb, max_side)
    return _to_float01(rgb)


def load_full(path):
    """Imagen a resolucion completa para exportar: float32 RGB 0..1."""
    rgb = _load_rgb(path, half_size=False)
    return _to_float01(rgb)


def load_thumb_bytes(path):
    """Intenta extraer la miniatura JPEG incrustada del RAW (rapido).
    Devuelve (bytes JPEG, orientacion LibRaw) o (None, 0)."""
    if not is_raw(path):
        return None, 0
    try:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
            flip = int(getattr(raw.sizes, "flip", 0) or 0)
        if thumb.format == rawpy.ThumbFormat.JPEG:
            return thumb.data, flip
    except Exception:
        pass
    return None, 0


def decode_thumb(data, flip):
    """Decodifica la miniatura incrustada y la endereza: con su propio EXIF
    si lo trae y, si no, con la orientacion que LibRaw leyo del RAW
    (convencion dcraw: bit 0 espejo horizontal, bit 1 vertical, bit 2
    transponer). Los iPhone guardan la miniatura sin enderezar."""
    import io
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.getexif().get(274, 1) != 1:
                return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
            arr = np.asarray(im.convert("RGB"))
    except Exception:
        return None
    if flip & 1:
        arr = arr[:, ::-1]
    if flip & 2:
        arr = arr[::-1]
    if flip & 4:
        arr = np.transpose(arr, (1, 0, 2))
    return np.ascontiguousarray(arr)


def load_thumb_array(path, max_side=320):
    """Miniatura como array uint8 RGB (camino lento, sirve para todo formato)."""
    try:
        rgb = _load_rgb(path, half_size=True)
        rgb = _resize_max(rgb, max_side)
        if rgb.dtype == np.uint16:
            rgb = (rgb >> 8).astype(np.uint8)
        return rgb
    except Exception:
        return None
