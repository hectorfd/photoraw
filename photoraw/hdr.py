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

HDR REAL (fusion lineal + mapeo tonal)
    Reconstruye cuanta luz recibio de verdad cada punto de la escena. Para
    eso necesita saber con que exposicion se tomo cada foto (del EXIF) y
    partir de las tomas en LINEAL, tal y como salen del sensor. Cada toma se
    divide por su exposicion —asi las tres quedan en la misma escala de luz—
    y se promedian quedandose de cada una con la parte que capto bien.
    El resultado intermedio es un mapa de luz que puede abarcar 20 pasos
    o mas —imposible de ensenar en una pantalla—, asi que despues se comprime
    a algo visible (mapeo tonal). Es el HDR "de manual": conserva mejor las
    luces altas y da mas margen de interpretacion.

En los dos casos el resultado sale algo plano a proposito, con casi todo el
margen recogido y las luces sin quemar. Eso no es un defecto, es la materia
prima — luego lo revelas en PhotoRAW como cualquier otro archivo (contraste,
curva, color) partiendo de una base que no ha perdido nada por arriba.

Plano, eso si, no es lechoso: los dos metodos dejan un punto negro de
verdad (ver `_punto_negro`). Una foto en la que la zona mas oscura sea gris
medio no es materia prima, es una foto velada.

POR QUE NO SE USA DEBEVEC (que es lo que venia antes)
    El metodo clasico de Debevec parte de las tomas ya reveladas y deduce la
    curva del sensor comparandolas. Con tres tomas y separaciones de 2 pasos
    la deduccion sale mal: medido en la tanda IMG_3577-79, la curva estimada
    no era ni siquiera creciente (un valor 224 se leia como MENOS luz que un
    160) y los tres canales se separaban hasta 3x en la zona alta —al nivel
    192, el rojo valia 2,62 y el verde 0,92—. Eso es exactamente lo que
    pintaba de verde y magenta zonas que no lo son. Partiendo del lineal no
    hay ninguna curva que deducir: el sensor ya es lineal por definicion, y
    el EXIF dice el resto.
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
# Los del HDR real son los del mapeo tonal: gamma 2,2 porque el mapa de luz
# que sale de la fusion es lineal y hay que pasarlo a pantalla.
# `color` viene a 1,25 y no a 1: comprimir el brillo dejando el color quieto
# desatura por su cuenta —el ojo lee menos color cuanto menos contraste hay
# alrededor—, y sin ese empujon el HDR real sale notablemente mas soso que
# el natural con la misma tanda (38,9 de saturacion media contra 58,3).
DEFAULT_PARAMS = {
    "method": NATURAL,
    "align": True,
    # natural (Mertens)
    "contrast": 1.0, "saturation": 1.0, "exposure": 0.5,
    # HDR real (mapeo tonal propio)
    "gamma": 2.2, "intensity": 0.0, "light": 0.5, "color": 1.25,
}

# A donde va el gris medio de la foto al mapearla. 0,18 es el gris 18 % de
# toda la vida: el punto al que mide una camara. El mando de brillo mueve
# esto arriba y abajo en pasos de luz.
MID_GRAY = 0.18

# que mandos usa cada metodo, para que la interfaz ensene solo los que tocan
METHOD_KEYS = {
    NATURAL: ("contrast", "saturation", "exposure"),
    HDR: ("intensity", "gamma", "light", "color"),
}

SUFFIX = "_hdr"


# ---------- carga de las tomas ----------

def load_shot(path, half_size=False, max_side=None, linear=False):
    """Una toma del bracketing, conservando su exposicion.

    Normalmente devuelve uint8 RGB ya revelado, que es lo que come la fusion
    natural. Con `linear=True` devuelve float32 RGB 0..1 SIN la curva de
    revelado, o sea el numero de fotones tal cual: el doble de luz, el doble
    de valor. Eso es lo que necesita el HDR real para poder sumar tomas de
    exposiciones distintas — con la curva puesta, las cuentas no salen.

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
                use_camera_wb=True, half_size=half_size,
                output_bps=16 if linear else 8,
                no_auto_bright=True,
                # gamma (1, 1) = sin curva; el sensor tal cual
                **({"gamma": (1, 1)} if linear else {}),
                # sin recuperacion de altas luces: lo quemado de la toma
                # clara se descarta solo al pesar las zonas, y reconstruirlo
                # aqui solo inventaria color que la toma oscura ya trae bien
                highlight_mode=rawpy.HighlightMode.Clip)
    else:
        rgb = loader._load_rgb(path, half_size=False)
        if linear:
            # un JPEG llega con la curva de pantalla puesta; se le quita para
            # dejarlo en la misma escala lineal que un RAW
            top = 65535.0 if rgb.dtype == np.uint16 else 255.0
            rgb = _srgb_to_linear(rgb.astype(np.float32) / top)
        elif rgb.dtype == np.uint16:
            rgb = (rgb >> 8).astype(np.uint8)
    if max_side is not None:
        rgb = loader._resize_max(rgb, max_side)
    if linear and rgb.dtype == np.uint16:
        rgb = rgb.astype(np.float32) / 65535.0
    return np.ascontiguousarray(rgb)


def _srgb_to_linear(x):
    """Quita la curva de pantalla (sRGB) y deja luz lineal. Entra y sale 0..1."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92,
                    ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def load_group(paths, half_size=False, max_side=None, progress_cb=None,
               linear=False):
    """Carga toda la tanda. Todas las tomas se recortan al tamano de la mas
    pequena: si una salio girada o de otra camara, mejor perder un borde que
    petar al fusionar."""
    shots = []
    for i, p in enumerate(paths):
        shots.append(load_shot(p, half_size=half_size, max_side=max_side,
                               linear=linear))
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

    Las tomas del HDR real llegan en lineal (float32), y AlignMTB solo sabe
    mirar imagenes de 8 bits. Para esas se le pide solo el desplazamiento,
    calculado sobre una copia en 8 bits, y se aplica aparte.
    """
    if len(shots) < 2:
        return shots
    try:
        if shots[0].dtype == np.uint8:
            out = [s.copy() for s in shots]
            cv2.createAlignMTB().process(list(shots), out)
            return out
        return _align_linear(shots)
    except Exception:
        return shots


def _align_linear(shots):
    """Alinea tomas en lineal. La referencia es la del medio (la de en medio
    de exposicion suele ser la que mejor se ve entera)."""
    mtb = cv2.createAlignMTB()
    grays = []
    for s in shots:
        lum = s[:, :, 0] * 0.2126 + s[:, :, 1] * 0.7152 + s[:, :, 2] * 0.0722
        # a 8 bits con curva: MTB compara por mediana y en lineal casi todo
        # el histograma se le apelotona abajo
        grays.append(np.clip(lum, 0.0, 1.0) ** (1 / 2.2) * 255.0)
    grays = [g.astype(np.uint8) for g in grays]
    ref = len(shots) // 2
    h, w = shots[0].shape[:2]
    out = []
    for i, s in enumerate(shots):
        dx, dy = (0, 0) if i == ref else mtb.calculateShift(grays[ref], grays[i])
        if not dx and not dy:
            out.append(s)
            continue
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        out.append(cv2.warpAffine(s, m, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REPLICATE))
    return out


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

    `shots` tienen que venir en LINEAL (float32 0..1, de load_shot con
    linear=True). Devuelve float32 RGB que pasa de 1 con holgura: en la tanda
    de ejemplo llega a 17, unos 21 pasos de luz entre lo mas oscuro y lo mas
    claro. Ninguna pantalla ensena eso, de ahi el mapeo tonal de despues.

    La cuenta es directa: como el sensor es lineal, dividir una toma por su
    exposicion la lleva a la escala de luz de la escena, y todas las tomas
    caen encima de la misma. Solo queda promediarlas quedandose de cada una
    con lo que capto bien, que es lo que hace el peso: cae a cero en lo
    quemado (arriba ya no hay informacion, solo tope) y en lo que es puro
    ruido (abajo), y manda en la zona media.

    EL PESO ES POR PIXEL, NO POR CANAL, y no es un detalle: si cada canal
    eligiera toma por su cuenta, en una zona donde el verde esta quemado pero
    el rojo no, el pixel saldria con el verde de una toma y el rojo de otra.
    Eso es color inventado — el sillon verde.

    LA CAMPANA DE PESO TIENE QUE SER ESTRECHA, y esto si costo un disgusto
    real: con una campana ancha, casi toda la zona media de la foto recibe
    peso de las TRES tomas a la vez (no solo de la mejor expuesta), y
    "align" solo corrige un desplazamiento entero de camara, no lo que se
    movio DENTRO de la escena (hojas y flores con viento, gente, el pulso a
    mano). En cuanto se mezclan tomas que no encajan pixel a pixel el
    resultado se ve derretido, sin detalle fino, aunque cada toma por
    separado estuviera nitida (comparar un HDR de un rosal con la toma RAW
    suelta lo ensena clarisimo). Estrechar la campana hace que casi todo
    pixel salga de UNA sola toma, la mejor expuesta ahi, y solo se mezcle
    con otra justo en la franja de transicion donde ninguna domina — igual
    de nitido que la fusion natural, sin perder rango real.
    """
    t = np.asarray(exps, np.float32)
    t = t / t.max()                      # relativas; la escala absoluta da igual
    acc = np.zeros_like(shots[0])
    wsum = np.zeros(shots[0].shape[:2], np.float32)
    for shot, ti in zip(shots, t):
        top = shot.max(axis=2)           # el canal que antes se quema
        # campana centrada en el gris medio, mirada como la ve el ojo
        w = np.exp(-32.0 * (np.sqrt(top) - 0.5) ** 2).astype(np.float32)
        w[top > 0.99] = 0.0              # quemado: fuera el pixel entero
        w[top < 0.002] = 0.0             # por debajo de esto solo hay ruido
        acc += (shot / ti) * w[..., None]
        wsum += w
    huerfanos = wsum < 1e-6              # quemado o negro en TODAS las tomas
    if huerfanos.any():
        # de la mas oscura, que es la que menos probable que lo tenga quemado
        oscura = int(np.argmin(t))
        acc[huerfanos] = shots[oscura][huerfanos] / t[oscura]
        wsum[huerfanos] = 1.0
    return acc / wsum[..., None]


def tonemap(light, p):
    """Comprime el mapa de luz a algo que quepa en una pantalla. RGB 0..1.

    Se trabaja sobre el BRILLO y el color se lleva de paseo: se calcula
    cuanto hay que bajar cada zona y se aplica esa misma proporcion a los
    tres canales. Asi un rojo sigue siendo el mismo rojo despues de
    comprimir, solo mas oscuro. (Comprimir cada canal por su lado, que es lo
    que hace el Reinhard de OpenCV, tuerce los colores: el canal mas alto se
    frena antes que los otros y el tono se va.)

    El anclaje es la media geometrica del brillo, o sea el gris medio de la
    escena: se lleva al 18 % y el mando de brillo lo sube o lo baja en pasos.
    Da igual lo brillante que fuera la escena o como se llamen las unidades.
    """
    lum = np.maximum(_luma(light), 1e-4)
    # media geometrica: el "gris medio" de esta escena
    gris = float(np.exp(np.log(lum).mean()))
    escala = MID_GRAY * (2.0 ** float(p["intensity"])) / max(gris, 1e-6)
    x = lum * escala
    # Reinhard con punto de blanco: `light` dice a partir de cuanta luz algo
    # se da por blanco. OJO CON LA DIRECCION, es al reves de lo que suena:
    # bajo pone el listón cerca, asi que las luces llegan a blanco y se
    # recortan antes (medido en IMG_3577-79: 3,2 % de pixeles quemados a 0 %
    # contra 1,2 % a 100 %); alto lo aleja y las altas luces se comprimen sin
    # llegar nunca a quemarse, a cambio de un cielo mas apagado.
    blanco = 2.0 ** (1.0 + 6.0 * float(np.clip(p["light"], 0.0, 1.0)))
    comprimido = x * (1.0 + x / (blanco * blanco)) / (1.0 + x)

    out = light * (comprimido / lum)[..., None]
    color = float(np.clip(p["color"], 0.0, 2.0))
    if color != 1.0:
        g = _luma(out)[..., None]
        out = g + (out - g) * color
    out = np.clip(out, 0.0, 1.0) ** (1.0 / max(float(p["gamma"]), 0.1))
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return _punto_negro(out)


# Cuanta foto se deja caer a negro del todo. Medio punto porcentual es lo que
# hace cualquier revelado, y es lo que separa "plano" de "lechoso": sin esto
# el HDR real no bajaba de 34 sobre 255 en su parte mas oscura —o sea, ni un
# negro en toda la foto— mientras la fusion natural de la misma tanda llegaba
# a 6. Ese velo gris es lo que hacia que el HDR real pareciera roto.
NEGRO_PCT = 0.5


def _punto_negro(rgb01):
    """Baja la sombra mas profunda hasta negro, como el revelado de un RAW.

    La luz real de una escena no llega a cero nunca —siempre hay algo de luz
    rebotada— y el mapa de luz lo refleja fielmente. Pero una foto sin ningun
    negro se ve velada, asi que aqui se hace lo mismo que hace la camara al
    revelar: se decide un punto negro y lo que cae por debajo es negro.

    El resto de la foto se estira para llenar el hueco, y por eso esto tambien
    devuelve algo de color: comprimir el rango apaga la saturacion y bajar el
    negro la recupera (medido en tres tandas de interior: de 24 a 39, de 44 a
    64 y de 97 a 123 de saturacion media).
    """
    muestra = rgb01[::4, ::4]
    negro = float(np.percentile(muestra, NEGRO_PCT)) if muestra.size else 0.0
    negro = min(max(negro, 0.0), 0.25)     # tope por si la escena es rarisima
    if negro <= 1e-4:
        return np.clip(rgb01, 0.0, 1.0).astype(np.float32)
    return np.clip((rgb01 - negro) / (1.0 - negro), 0.0, 1.0).astype(np.float32)


def _luma(rgb):
    """Brillo percibido de un RGB lineal (pesos Rec.709)."""
    return (rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152
            + rgb[:, :, 2] * 0.0722)


def fuse_hdr(shots, exps, p):
    """HDR real: reconstruye la luz de la escena y la mapea a pantalla.

    `shots` en lineal (ver merge_light_map).
    """
    light = merge_light_map(shots, exps)
    out = tonemap(light, p)
    del light
    return out


def wants_linear(params, exps):
    """Si estos ajustes van a acabar en el HDR real, que come tomas lineales.

    Quien carga las tomas tiene que saberlo ANTES de cargarlas, porque cada
    metodo necesita un revelado distinto: sin curva para el HDR real y el
    revelado normal para el natural. Una tanda sin datos de exposicion cae al
    natural aunque se pida HDR real, y entonces tampoco quiere lineal.
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    return p.get("method") == HDR and exps is not None


def fuse(shots, params=None, exps=None):
    """Junta las tomas en una sola. Devuelve float32 RGB 0..1.

    `exps` son las exposiciones de cada toma, en el mismo orden; hacen falta
    para el HDR real. Sin ellas se usa la fusion natural, que no las necesita.

    OJO: las tomas tienen que venir cargadas para el metodo que se pide —
    uint8 revelado para el natural, float32 lineal para el HDR real (ver
    `wants_linear`). Si no cuadran se fusiona con lo que se pueda hacer con
    lo que hay, en vez de sacar una foto con el color estropeado.
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    hdr_real = (p.get("method") == HDR and exps is not None
                and len(shots) >= 2 and shots[0].dtype != np.uint8)
    if p.get("align"):
        shots = align(shots)
    if hdr_real:
        return fuse_hdr(shots, exps, p)
    if shots[0].dtype != np.uint8:
        # tomas lineales pero toca fusion natural: se les pone la curva de
        # pantalla, que es con lo que Mertens sabe pesar las zonas
        shots = [(np.clip(s, 0, 1) ** (1 / 2.2) * 255.0 + 0.5).astype(np.uint8)
                 for s in shots]
    return fuse_natural(shots, p)


def fuse_paths(paths, params=None, half_size=False, max_side=None,
               progress_cb=None):
    """Camino completo: carga las tomas del disco y las fusiona."""
    p = dict(DEFAULT_PARAMS, **(params or {}))
    exps = exposures(paths) if p.get("method") == HDR else None
    shots = load_group(paths, half_size=half_size, max_side=max_side,
                       progress_cb=progress_cb,
                       linear=p.get("method") == HDR and exps is not None)
    out = fuse(shots, p, exps=exps)
    del shots
    return out


# ---------- guardado ----------

DNG = "dng"
TIFF = "tif"
DEFAULT_FORMAT = DNG


def output_path(paths, folder=None, fmt=DEFAULT_FORMAT):
    """Nombre libre para el resultado: el de la primera toma con `_hdr`.

    "Primera" por nombre, no por el orden en que llega la lista (que va de
    oscura a clara), para que el HDR caiga junto a sus tomas en la tira.
    """
    ext = "dng" if fmt == DNG else "tif"
    first = min((Path(p) for p in paths), key=lambda p: p.name.lower())
    folder = Path(folder) if folder else first.parent
    dest = folder / f"{first.stem}{SUFFIX}.{ext}"
    n = 2
    while dest.exists():
        dest = folder / f"{first.stem}{SUFFIX}{n}.{ext}"
        n += 1
    return dest


def save(rgb01, dest, fmt=None):
    """Guarda el resultado. Por defecto DNG lineal; si no, TIFF de 16 bits.

    Los dos son de 16 bits, y no de 8, porque la fusion sale a proposito
    plana: al revelarla despues se estiran mucho las sombras y las luces, y
    en 8 bits eso deja escalones (banding) en los degradados de un cielo.

    El DNG lineal es lo que entrega Lightroom al fusionar un HDR y es el de
    fabrica aqui: pesa lo mismo pero PhotoRAW lo trata como un RAW, con el
    balance de blancos todavia por decidir. El TIFF sigue estando para
    llevarselo a un programa que no entienda DNG.
    """
    dest = Path(dest)
    if fmt is None:
        fmt = DNG if dest.suffix.lower() == ".dng" else TIFF
    if fmt == DNG:
        from photoraw import dng
        return dng.save(rgb01, dest)
    data = (np.clip(rgb01, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    # cv2.imwrite falla en Windows con rutas con tildes o enes sin avisar
    # (ni excepcion ni False fiable); codificar en memoria y escribir con
    # Path si funciona con cualquier ruta.
    # 8 = ADOBE_DEFLATE; no todas las compilaciones de OpenCV lo traen
    ok, buf = cv2.imencode(".tiff", bgr, [cv2.IMWRITE_TIFF_COMPRESSION, 8])
    if not ok:
        ok, buf = cv2.imencode(".tiff", bgr)
    if not ok:
        raise OSError(f"no se pudo escribir {dest}")
    dest.write_bytes(buf.tobytes())
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
