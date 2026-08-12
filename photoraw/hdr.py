"""Fusion de un bracketing (HDR).

Tu camara, con el bracketing activado (AEB), dispara la misma escena varias
veces cambiando la exposicion: una oscura que conserva el cielo, una normal
y una clara que abre las sombras. Aqui se juntan en una sola foto que tiene
detalle en todas las zonas. Hay dos formas de hacerlo, y no son variantes de
lo mismo: se parecen a dos oficios distintos.

NATURAL (fusion de exposiciones, metodo de Mertens)
    No intenta saber cuanta luz habia en la escena. Mira las tomas zona por
    zona y se queda con la mejor de cada una, pesando tres cosas: cuanto
    contraste local tiene, cuanto color, y lo cerca que esta del gris medio.
    Es el "HDR natural" tipo Lightroom. No necesita EXIF y es rapido.

HDR REAL (Debevec + mapeo tonal)
    Reconstruye cuanta luz recibio de verdad cada punto de la escena. Para
    eso necesita saber con que exposicion se tomo cada foto (del EXIF), y
    deduce la curva de respuesta del sensor comparando las tomas entre si.
    El resultado intermedio es un mapa de luz real que puede abarcar 15 pasos
    o mas —imposible de ensenar en una pantalla—, asi que despues se comprime
    a algo visible (mapeo tonal de Reinhard). Es el HDR "de manual": conserva
    mejor las luces altas y da mas margen de interpretacion, a cambio de que
    pasarse con los mandos deje el tipico aspecto artificial.

En los dos casos el resultado sale plano a proposito: sin negros puros ni
blancos quemados, con todo el margen recogido. Eso no es un defecto, es la
materia prima — luego lo revelas en PhotoRAW como cualquier otro archivo
(punto negro, contraste, curva) partiendo de una base que ya no tiene nada
quemado.
"""
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import Image

from photoraw import diskcache, loader

NATURAL = "natural"
HDR = "hdr"

# Valores de fabrica. Los tres pesos de la fusion natural: contraste y color
# son los de OpenCV; el equilibrio (well-exposedness) viene de fabrica a 0 y
# se sube un poco porque con paisajes suaviza los cortes entre tomas.
# Los del HDR real son los del mapeo de Reinhard, con gamma 2,2 porque el
# mapa de luz que sale de la fusion es lineal y hay que pasarlo a pantalla.
DEFAULT_PARAMS = {
    "method": NATURAL,
    "align": True,
    # natural (Mertens)
    "contrast": 1.0, "saturation": 1.0, "exposure": 0.5,
    # HDR real (Reinhard)
    "gamma": 2.2, "intensity": 0.0, "light": 1.0, "color": 0.0,
}

# que mandos usa cada metodo, para que la interfaz ensene solo los que tocan
METHOD_KEYS = {
    NATURAL: ("contrast", "saturation", "exposure"),
    HDR: ("intensity", "gamma", "light", "color"),
}

SUFFIX = "_hdr"


# ---------- carga de las tomas ----------

def load_shot(path, half_size=False, max_side=None):
    """Una toma del bracketing como uint8 RGB, conservando su exposicion.

    OJO con `no_auto_bright`: el revelado normal de PhotoRAW deja que LibRaw
    estire el histograma de cada foto para que salga bien expuesta. Aqui eso
    seria fatal — igualaria el brillo de las tres tomas y la fusion no
    tendria nada que elegir. Asi que se apaga: las diferencias de exposicion
    tienen que llegar intactas.
    """
    path = Path(path)
    if loader.is_raw(path):
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True, half_size=half_size, output_bps=8,
                no_auto_bright=True,
                # sin recuperacion de altas luces: lo quemado de la toma
                # clara se descarta solo al pesar las zonas, y reconstruirlo
                # aqui solo inventaria color que la toma oscura ya trae bien
                highlight_mode=rawpy.HighlightMode.Clip)
    else:
        rgb = loader._load_rgb(path, half_size=False)
        if rgb.dtype == np.uint16:
            rgb = (rgb >> 8).astype(np.uint8)
    if max_side is not None:
        rgb = loader._resize_max(rgb, max_side)
    return np.ascontiguousarray(rgb)


def load_group(paths, half_size=False, max_side=None, progress_cb=None):
    """Carga toda la tanda. Todas las tomas se recortan al tamano de la mas
    pequena: si una salio girada o de otra camara, mejor perder un borde que
    petar al fusionar."""
    shots = []
    for i, p in enumerate(paths):
        shots.append(load_shot(p, half_size=half_size, max_side=max_side))
        if progress_cb:
            progress_cb((i + 1) / len(paths))
    h = min(s.shape[0] for s in shots)
    w = min(s.shape[1] for s in shots)
    return [np.ascontiguousarray(s[:h, :w]) for s in shots]


# ---------- datos de exposicion (para el HDR real) ----------

def _exif(path):
    """EXIF de la foto como (ifd0, subifd Exif). Vacios si no se puede leer.

    Los RAW tambien pasan por aqui: un DNG (y casi cualquier RAW) es un TIFF
    por dentro, y Pillow lo abre lo justo para leerle el EXIF aunque no sepa
    revelarlo. rawpy, que si lo revela, no expone estos datos.
    """
    try:
        with Image.open(path) as im:
            ifd0 = im.getexif()
            return ifd0, ifd0.get_ifd(0x8769)
    except Exception:
        return {}, {}


def read_exposure(path):
    """Cuanta luz recogio esta toma, en unidades relativas, o None.

    No vale con el tiempo de exposicion: muchas camaras (el iPhone entre
    ellas) hacen el bracketing subiendo tambien el ISO. En la tanda de
    ejemplo, del 1/59 al 1/17 hay 1,8 pasos de luz, pero contando el ISO
    (200 -> 1000) son 4,1 de verdad. Fusionar con los tiempos a secas dejaria
    el HDR mal calibrado.

    Lo que importa es la senal que sale del sensor para una misma escena, que
    va con el tiempo por el ISO y por la inversa del diafragma al cuadrado.
    La constante da igual: la fusion solo usa las proporciones entre tomas.
    """
    _ifd0, exif = _exif(path)
    if not exif:
        return None
    try:
        t = float(exif.get(0x829A))         # ExposureTime
    except (TypeError, ValueError):
        return None
    if not t > 0:
        return None
    try:
        n = float(exif.get(0x829D) or 1.0)  # FNumber
    except (TypeError, ValueError):
        n = 1.0
    iso = exif.get(0x8827)                  # ISOSpeedRatings
    if isinstance(iso, (list, tuple)):
        iso = iso[0] if iso else None
    try:
        iso = float(iso or 100.0)
    except (TypeError, ValueError):
        iso = 100.0
    if not n > 0:
        n = 1.0
    return t * max(iso, 1.0) / (n * n)


def exposures(paths):
    """Exposiciones de la tanda, o None si a alguna le falta el dato.

    Se exige que todas lo tengan y que no sean todas iguales: con tomas
    "iguales" de exposicion el HDR real no tiene con que reconstruir nada.
    """
    vals = [read_exposure(p) for p in paths]
    if any(v is None for v in vals):
        return None
    if len(set(vals)) < 2:
        return None
    return np.array(vals, np.float32)


def can_do_hdr(paths):
    """Si esta tanda trae lo necesario para el HDR real."""
    return exposures(paths) is not None


# ---------- alineado y fusion ----------

def align(shots):
    """Corrige el desplazamiento entre tomas de un bracketing a pulso.

    AlignMTB compara las tomas por su mediana de brillo, asi que no le
    molesta que unas sean mas claras que otras; solo corrige el movimiento
    de la camara (desplazamiento), no lo que se movio dentro de la escena.
    Si algo falla se devuelven las tomas tal cual: mejor un HDR con un pelo
    de doble contorno que ninguno.

    OJO: AlignMTB escribe dentro de los arrays que se le pasan como destino
    (el ejemplo tipico, `process(lista, lista)`, machaca la entrada). Aqui se
    le dan copias, porque quien llama conserva sus tomas: el dialogo las
    guarda para refusionar al mover un ajuste, y alinear encima de lo ya
    alineado va acumulando desplazamiento y estropeando la vista previa.
    """
    if len(shots) < 2:
        return shots
    try:
        out = [s.copy() for s in shots]
        cv2.createAlignMTB().process(list(shots), out)
        return out
    except Exception:
        return shots


def fuse_natural(shots, p):
    """Fusion de exposiciones (Mertens). float32 RGB 0..1.

    OJO si algun dia comparas dos fusiones: MergeMertens no da dos veces el
    mismo resultado exacto con la misma entrada (reparte las piramides entre
    hilos y las sumas en float32 salen en otro orden). El baile es de 2e-07,
    o sea cero niveles de los 255 que se guardan; solo cambia de valor algun
    pixel que caia justo en la frontera del redondeo, dos de cada millon.
    El HDR real si es determinista.
    """
    merger = cv2.createMergeMertens(float(p["contrast"]),
                                    float(p["saturation"]),
                                    float(p["exposure"]))
    out = merger.process([cv2.cvtColor(s, cv2.COLOR_RGB2BGR) for s in shots])
    # Mertens se pasa un poco de 1 en las luces altas
    return np.clip(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), 0.0, 1.0)


def merge_light_map(shots, exps):
    """Mapa de luz real de la escena (HDR lineal), sin comprimir.

    Devuelve float32 BGR que puede pasar de 1 con holgura: en la tanda de
    ejemplo llega a 23, unos 15 pasos de luz entre lo mas oscuro y lo mas
    claro. Ninguna pantalla ensena eso, de ahi el mapeo tonal de despues.

    Antes de fusionar hay que saber como responde el sensor: la relacion
    entre la luz que entra y el numero que sale no es una linea recta (lleva
    la curva del revelado). CalibrateDebevec la deduce comparando las mismas
    zonas en tomas con exposicion distinta.
    """
    bgr = [cv2.cvtColor(s, cv2.COLOR_RGB2BGR) for s in shots]
    times = np.asarray(exps, np.float32).copy()   # OpenCV escribe en el array
    response = cv2.createCalibrateDebevec().process(bgr, times.copy())
    return cv2.createMergeDebevec().process(bgr, times, response)


def fuse_hdr(shots, exps, p):
    """HDR real: reconstruye la luz de la escena y la mapea a pantalla."""
    light = merge_light_map(shots, exps)
    # Reinhard: comprime segun el brillo de cada zona, no con una curva fija.
    # `light` (adaptacion local) es lo que decide si manda el contraste de la
    # zona o el de la foto entera; `color` mantiene el color medio en vez de
    # tratar cada canal por su cuenta.
    mapper = cv2.createTonemapReinhard(
        max(float(p["gamma"]), 0.1), float(p["intensity"]),
        float(np.clip(p["light"], 0.0, 1.0)),
        float(np.clip(p["color"], 0.0, 1.0)))
    out = mapper.process(light)
    del light
    # el mapeo puede escupir NaN donde el mapa de luz venia a cero
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), 0.0, 1.0)


def fuse(shots, params=None, exps=None):
    """Junta las tomas en una sola. Devuelve float32 RGB 0..1.

    `exps` son las exposiciones de cada toma, en el mismo orden; hacen falta
    para el HDR real. Sin ellas se usa la fusion natural, que no las necesita.
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    if p.get("align"):
        shots = align(shots)
    if p.get("method") == HDR and exps is not None and len(shots) >= 2:
        return fuse_hdr(shots, exps, p)
    return fuse_natural(shots, p)


def fuse_paths(paths, params=None, half_size=False, max_side=None,
               progress_cb=None):
    """Camino completo: carga las tomas del disco y las fusiona."""
    p = dict(DEFAULT_PARAMS, **(params or {}))
    exps = exposures(paths) if p.get("method") == HDR else None
    shots = load_group(paths, half_size=half_size, max_side=max_side,
                       progress_cb=progress_cb)
    out = fuse(shots, p, exps=exps)
    del shots
    return out


# ---------- guardado ----------

def output_path(paths, folder=None):
    """Nombre libre para el resultado: el de la primera toma con `_hdr`.

    "Primera" por nombre, no por el orden en que llega la lista (que va de
    oscura a clara), para que el HDR caiga junto a sus tomas en la tira.
    """
    first = min((Path(p) for p in paths), key=lambda p: p.name.lower())
    folder = Path(folder) if folder else first.parent
    dest = folder / f"{first.stem}{SUFFIX}.tif"
    n = 2
    while dest.exists():
        dest = folder / f"{first.stem}{SUFFIX}{n}.tif"
        n += 1
    return dest


def save(rgb01, dest):
    """Guarda el resultado como TIFF de 16 bits.

    16 bits y no JPEG porque la fusion sale a proposito plana y con todo el
    margen recogido: al revelarla despues se estiran mucho las sombras y las
    luces, y en 8 bits eso deja escalones (banding) en los degradados de un
    cielo. Comprimido sin perdida cuando se puede.
    """
    dest = Path(dest)
    data = (np.clip(rgb01, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    # 8 = ADOBE_DEFLATE; no todas las compilaciones de OpenCV lo traen
    if not cv2.imwrite(str(dest), bgr, [cv2.IMWRITE_TIFF_COMPRESSION, 8]):
        if not cv2.imwrite(str(dest), bgr):
            raise OSError(f"no se pudo escribir {dest}")
    return dest


# ---------- deteccion automatica de tandas ----------

def shot_time(path):
    """Momento del disparo, en segundos.

    Manda el EXIF, tambien en los RAW: un DNG es un TIFF por dentro y Pillow
    le lee la fecha aunque no sepa revelarlo. Si no hay EXIF se usa la fecha
    del archivo, que suele ser la del disparo (la camara la escribe al grabar
    la tarjeta y Windows la conserva al copiar), pero se pierde si las fotos
    pasan por sitios que la reescriben.
    """
    path = Path(path)
    ifd0, exif = _exif(path)
    # DateTimeOriginal vive en el sub-IFD Exif; DateTime en el principal
    stamp = (exif.get(0x9003) if exif else None) or \
            (ifd0.get(0x0132) if ifd0 else None)
    if stamp:
        try:
            base = datetime.strptime(str(stamp).strip(),
                                     "%Y:%m:%d %H:%M:%S").timestamp()
            # SubsecTimeOriginal: las rafagas caen en el mismo segundo y sin
            # las decimas todas las tomas empatan al ordenarlas
            sub = str(exif.get(0x9291) or "").strip() if exif else ""
            if sub.isdigit():
                base += float(f"0.{sub}")
            return base
        except (ValueError, OSError):
            pass
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


def _thumb_luma(path):
    """Brillo medio (0..1) de la miniatura. Sale del JPEG que incrusto la
    camara, asi que refleja la exposicion real de la toma."""
    try:
        data = diskcache.thumb_jpeg(path)
        if not data:
            return None
        gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        return float(gray.mean()) / 255.0
    except Exception:
        return None


def detect_groups(files, max_gap=3.0, min_spread=0.05, min_shots=2,
                  max_shots=9, progress_cb=None):
    """Busca tandas de bracketing entre las fotos de la carpeta.

    Dos senales, y hacen falta las dos:

    1. Se dispararon seguidas (menos de `max_gap` segundos entre una y la
       siguiente). El bracketing sale en rafaga.
    2. Su brillo cambia de verdad (`min_spread`). Esto es lo que distingue
       una tanda de bracketing de una rafaga normal de tres fotos al niño
       corriendo, donde las tres estan igual de expuestas.

    Devuelve una lista de tandas, cada una ordenada de la mas oscura a la
    mas clara, y solo con lo que parece bracketing de verdad.
    """
    files = [Path(f) for f in files]
    if len(files) < min_shots:
        return []

    stamped = sorted(((shot_time(f), f) for f in files), key=lambda t: t[0])

    # 1. trocear por huecos de tiempo
    blocks, block = [], [stamped[0]]
    for prev, cur in zip(stamped, stamped[1:]):
        if cur[0] - prev[0] <= max_gap:
            block.append(cur)
        else:
            blocks.append(block)
            block = [cur]
    blocks.append(block)

    # 2. de los bloques con tamano de tanda, quedarse con los que cambian
    #    de brillo (la miniatura solo se mira aqui: es la parte lenta)
    candidates = [b for b in blocks if min_shots <= len(b) <= max_shots]
    groups = []
    for i, b in enumerate(candidates):
        if progress_cb:
            progress_cb((i + 1) / len(candidates))
        lumas = [(_thumb_luma(f), f) for _t, f in b]
        if any(l is None for l, _f in lumas):
            continue
        if max(l for l, _f in lumas) - min(l for l, _f in lumas) < min_spread:
            continue
        groups.append([f for _l, f in sorted(lumas, key=lambda t: t[0])])
    return groups


def describe_group(paths):
    """Resumen de una tanda para la interfaz: 'IMG_0031 … IMG_0033 (3 tomas)'."""
    paths = sorted((Path(p) for p in paths), key=lambda p: p.name.lower())
    if len(paths) == 1:
        return paths[0].name
    return f"{paths[0].stem} … {paths[-1].stem} ({len(paths)} tomas)"
