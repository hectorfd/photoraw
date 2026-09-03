"""Correccion de lente: distorsion geometrica, aberracion cromatica y
vineteado, automatica a partir del EXIF de cada foto.

No hay mandos para esto (a proposito): se lee la camara y el objetivo del
EXIF, se busca su perfil en la base de datos de lensfun -viene empaquetada
con lensfunpy, sin descargas aparte- y si aparece se corrige. Sin perfil no
hay nada que hacer y la foto sale tal cual, sin avisar: es lo normal en
moviles y en muchas compactas, que no traen esos datos en el EXIF."""
from pathlib import Path

import cv2
import numpy as np

_db = None
_available = None
# (fabricante, modelo, fabricante objetivo, modelo objetivo, focal, apertura,
#  ancho, alto) -> Modifier ya inicializado, o None si no hay perfil
_mod_cache = {}


def available():
    """Si la libreria de correccion esta instalada."""
    global _available
    if _available is None:
        try:
            import lensfunpy  # noqa: F401
            _available = True
        except Exception:
            _available = False
    return _available


def _get_db():
    global _db
    if _db is None:
        import lensfunpy
        _db = lensfunpy.Database()
    return _db


def _exif_info(path):
    """(fabricante camara, modelo camara, fabricante objetivo o None, modelo
    objetivo, focal en mm, apertura) o None si falta camara, objetivo o
    focal -- sin esos tres datos no hay con que buscar un perfil."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            ifd0 = im.getexif()
            exif = ifd0.get_ifd(0x8769) or {}
    except Exception:
        return None
    maker = ifd0.get(0x010F)
    model = ifd0.get(0x0110)
    lens_model = exif.get(0xA434)
    lens_maker = exif.get(0xA433)
    focal = exif.get(0x920A)
    aperture = exif.get(0x829D)
    if not maker or not model or not lens_model or not focal:
        return None
    try:
        focal = float(focal)
        aperture = float(aperture) if aperture else 0.0
    except (TypeError, ValueError):
        return None
    if focal <= 0:
        return None
    return (str(maker).strip(), str(model).strip(),
            str(lens_maker).strip() if lens_maker else None,
            str(lens_model).strip(), focal, aperture)


def _find_modifier(info, width, height):
    maker, model, lens_maker, lens_model, focal, aperture = info
    key = (maker, model, lens_maker, lens_model, focal,
           round(aperture, 1), width, height)
    if key in _mod_cache:
        return _mod_cache[key]
    import lensfunpy
    mod = None
    try:
        db = _get_db()
        cams = db.find_cameras(maker, model, loose_search=True)
        if cams:
            cam = cams[0]
            lenses = db.find_lenses(cam, lens_maker, lens_model,
                                    loose_search=True)
            if lenses:
                m = lensfunpy.Modifier(lenses[0], cam.crop_factor,
                                       width, height)
                # aperture a proposito con respaldo (f/2.8): sin ella el
                # vineteado no se puede calcular, pero la distorsion y la
                # aberracion cromatica casi no dependen del diafragma
                m.initialize(focal, aperture or 2.8, pixel_format=np.float32)
                mod = m
    except Exception:
        mod = None
    # cache pequena: cada entrada son solo parametros, no los mapas de
    # remapeo (esos se calculan al vuelo en cada `correct`)
    if len(_mod_cache) > 64:
        _mod_cache.pop(next(iter(_mod_cache)))
    _mod_cache[key] = mod
    return mod


def correct(rgb, path):
    """Corrige distorsion, aberracion cromatica y vineteado sobre `rgb`
    (mismo array que devuelve el resto de `loader`: uint8, uint16 o
    float32). Si la camara u objetivo no estan en la base de datos, o
    cualquier paso falla, devuelve `rgb` sin tocar -- nunca por esto se
    debe quedar una foto sin poder abrirse."""
    if not available():
        return rgb
    try:
        info = _exif_info(path)
        if info is None:
            return rgb
        h, w = rgb.shape[:2]
        mod = _find_modifier(info, w, h)
        if mod is None:
            return rgb

        top = (65535.0 if rgb.dtype == np.uint16 else
              255.0 if rgb.dtype == np.uint8 else None)
        img = rgb.astype(np.float32) / top if top else np.array(rgb, np.float32)

        # el vineteado se calcula sobre la geometria ORIGINAL (la caida de
        # luz hacia las esquinas sigue el objetivo tal y como disparo, no
        # la foto ya enderezada); por eso va antes del remapeo
        mod.apply_color_modification(img)

        # distorsion + aberracion cromatica en un solo mapa por canal: cada
        # canal se desplaza un pelo distinto (eso es la aberracion) ademas
        # de corregir el barril/cojin del objetivo
        coords = mod.apply_subpixel_geometry_distortion()
        if coords is not None:
            out = np.empty_like(img)
            for c in range(3):
                out[..., c] = cv2.remap(
                    img[..., c], coords[..., c, :], None,
                    cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101)
            img = out
        else:
            coords = mod.apply_geometry_distortion()
            if coords is not None:
                img = cv2.remap(img, coords, None, cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_REFLECT101)

        img = np.clip(img, 0.0, 1.0)
        if top:
            img = (img * top + 0.5).astype(rgb.dtype)
        return np.ascontiguousarray(img)
    except Exception:
        return rgb
