"""Balance de blancos en Kelvin y matiz reales, como en Lightroom.

Un RAW guarda con que balance de blancos disparo la camara (en un DNG, la
etiqueta AsShotNeutral: que color del sensor era "blanco"). Con las matrices
de color del archivo (ColorMatrix1/2: como ve el sensor la luz de tungsteno y
la de dia) ese blanco se pasa a coordenadas de color y de ahi a temperatura
en Kelvin y matiz verde/magenta. Las conversiones son las del SDK de DNG de
Adobe (tabla de Robertson), asi que los numeros se parecen a los de Lightroom.

Cambiar la temperatura es rehacer el balance de blancos: se calcula que
color tendria en el sensor el blanco de la luz elegida y se reequilibran los
canales DEL SENSOR para que ese color salga neutro. Se hace en luz lineal,
sobre la foto recien revelada y antes de cualquier curva — no es un tinte
encima de la foto, que es lo que hacia el deslizador relativo.
"""
import functools
from pathlib import Path

import numpy as np

# (1e6/K, u, v, pendiente de la isoterma): tabla de Robertson del SDK de DNG
_TABLA = np.array([
    (0, 0.18006, 0.26352, -0.24341), (10, 0.18066, 0.26589, -0.25479),
    (20, 0.18133, 0.26846, -0.26876), (30, 0.18208, 0.27119, -0.28539),
    (40, 0.18293, 0.27407, -0.30470), (50, 0.18388, 0.27709, -0.32675),
    (60, 0.18494, 0.28021, -0.35156), (70, 0.18611, 0.28342, -0.37915),
    (80, 0.18740, 0.28668, -0.40955), (90, 0.18880, 0.28997, -0.44278),
    (100, 0.19032, 0.29326, -0.47888), (125, 0.19462, 0.30141, -0.58204),
    (150, 0.19962, 0.30921, -0.70471), (175, 0.20525, 0.31647, -0.84901),
    (200, 0.21142, 0.32312, -1.0182), (225, 0.21807, 0.32909, -1.2168),
    (250, 0.22511, 0.33439, -1.4512), (275, 0.23247, 0.33904, -1.7298),
    (300, 0.24010, 0.34308, -2.0637), (325, 0.24792, 0.34655, -2.4681),
    (350, 0.25591, 0.34951, -2.9641), (375, 0.26400, 0.35200, -3.5814),
    (400, 0.27218, 0.35407, -4.3633), (425, 0.28039, 0.35577, -5.3762),
    (450, 0.28863, 0.35714, -6.7262), (475, 0.29685, 0.35823, -8.5955),
    (500, 0.30505, 0.35907, -11.324), (525, 0.31320, 0.35968, -15.628),
    (550, 0.32129, 0.36011, -23.325), (575, 0.32931, 0.36038, -40.770),
    (600, 0.33724, 0.36051, -116.45)])
_TINT_SCALE = -3000.0

TEMP_MIN, TEMP_MAX = 2000.0, 50000.0
TINT_MIN, TINT_MAX = -150.0, 150.0

# sRGB lineal (D65) -> XYZ
_XYZ_RGB = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])

# Iluminantes EXIF de las etiquetas CalibrationIlluminant -> Kelvin
_ILUMINANTES = {1: 5500, 2: 4200, 3: 2856, 4: 5500, 9: 5500, 10: 6500,
                11: 7500, 12: 6430, 13: 5000, 14: 4150, 15: 3450, 17: 2856,
                18: 4874, 19: 6774, 20: 5503, 21: 6504, 22: 7504, 23: 5003,
                24: 3200}


def xy_to_temp(x, y):
    """Coordenadas xy -> (Kelvin, matiz) como Lightroom."""
    d = 1.5 - x + 6.0 * y
    u, v = 2.0 * x / d, 3.0 * y / d
    last_dt = last_du = last_dv = 0.0
    for i in range(1, len(_TABLA)):
        du, dv = 1.0, _TABLA[i, 3]
        n = np.hypot(du, dv)
        du, dv = du / n, dv / n
        uu, vv = u - _TABLA[i, 1], v - _TABLA[i, 2]
        dt = -uu * dv + vv * du
        if dt <= 0.0 or i == len(_TABLA) - 1:
            dt = -min(dt, 0.0)
            f = 0.0 if i == 1 else dt / (last_dt + dt)
            temp = 1e6 / (_TABLA[i - 1, 0] * f + _TABLA[i, 0] * (1 - f))
            uu = u - (_TABLA[i - 1, 1] * f + _TABLA[i, 1] * (1 - f))
            vv = v - (_TABLA[i - 1, 2] * f + _TABLA[i, 2] * (1 - f))
            du, dv = du * (1 - f) + last_du * f, dv * (1 - f) + last_dv * f
            n = np.hypot(du, dv)
            return float(temp), float((uu * du + vv * dv) / n * _TINT_SCALE)
        last_dt, last_du, last_dv = dt, du, dv
    return 6500.0, 0.0


def temp_to_xy(temp, tint):
    """(Kelvin, matiz) -> coordenadas xy (la inversa de `xy_to_temp`)."""
    r = 1e6 / temp
    offset = tint / _TINT_SCALE
    for i in range(len(_TABLA) - 1):
        if r < _TABLA[i + 1, 0] or i == len(_TABLA) - 2:
            f = (_TABLA[i + 1, 0] - r) / (_TABLA[i + 1, 0] - _TABLA[i, 0])
            u = _TABLA[i, 1] * f + _TABLA[i + 1, 1] * (1 - f)
            v = _TABLA[i, 2] * f + _TABLA[i + 1, 2] * (1 - f)
            a = np.array([1.0, _TABLA[i, 3]]); a /= np.hypot(*a)
            b = np.array([1.0, _TABLA[i + 1, 3]]); b /= np.hypot(*b)
            c = a * f + b * (1 - f); c /= np.hypot(*c)
            u += c[0] * offset
            v += c[1] * offset
            d = u - 4.0 * v + 2.0
            return 1.5 * u / d, v / d
    return 0.3127, 0.3290


def _xy_to_xyz(x, y):
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


class CameraColor:
    """Lo que hace falta de un RAW para hablar en Kelvin: su blanco al
    disparar y como ve el sensor la luz (matrices XYZ -> camara)."""

    def __init__(self, neutral, cm1, cm2=None, t1=2856.0, t2=6504.0):
        self.neutral = np.asarray(neutral, float)
        self.cm1 = np.asarray(cm1, float)
        self.cm2 = self.cm1 if cm2 is None else np.asarray(cm2, float)
        self.t1, self.t2 = (t1, t2) if t1 <= t2 else (t2, t1)
        if t1 > t2:
            self.cm1, self.cm2 = self.cm2, self.cm1
        # la camara->sRGB con que LibRaw revelo la foto (convencion dcraw:
        # matriz de luz de dia, filas normalizadas para que el blanco del
        # sensor salga blanco)
        cam_rgb = self.cm2 @ _XYZ_RGB
        cam_rgb /= cam_rgb.sum(axis=1, keepdims=True)
        self.cam_rgb = cam_rgb
        self.rgb_cam = np.linalg.inv(cam_rgb)
        self.as_shot = self._neutral_to_temp(self.neutral)

    def _matrix(self, temp):
        """Matriz XYZ -> camara interpolada para esa luz (en 1/K, como DNG)."""
        if self.t1 == self.t2:
            return self.cm1
        g = (1.0 / temp - 1.0 / self.t2) / (1.0 / self.t1 - 1.0 / self.t2)
        g = float(np.clip(g, 0.0, 1.0))
        return self.cm1 * g + self.cm2 * (1.0 - g)

    def _neutral_to_temp(self, neutral):
        temp = 5000.0
        for _ in range(8):
            xyz = np.linalg.solve(self._matrix(temp), neutral)
            x, y = xyz[0] / xyz.sum(), xyz[1] / xyz.sum()
            nuevo, tint = xy_to_temp(x, y)
            if abs(nuevo - temp) < 1.0:
                break
            temp = nuevo
        return round(float(np.clip(nuevo, TEMP_MIN, TEMP_MAX))), \
            round(float(np.clip(tint, TINT_MIN, TINT_MAX)))

    def correction(self, temp, tint):
        """Matriz 3x3 en sRGB lineal que cambia el balance de la foto
        revelada (con el blanco de disparo) al de (temp, tint)."""
        temp = float(np.clip(temp, TEMP_MIN, TEMP_MAX))
        tint = float(np.clip(tint, TINT_MIN, TINT_MAX))
        x, y = temp_to_xy(temp, tint)
        blanco = self._matrix(temp) @ _xy_to_xyz(x, y)   # en el sensor
        # multiplicadores nuevos / los de disparo, canal a canal del sensor
        r = self.neutral / blanco
        r /= r[1]
        m = self.rgb_cam @ np.diag(r) @ self.cam_rgb
        # que el blanco no cambie de brillo, solo de color
        m /= (_XYZ_RGB[1] @ (m @ np.ones(3)))
        return m.astype(np.float32)


def _read_dng(path):
    from PIL import Image
    with Image.open(path) as im:
        t = im.tag_v2
        neutral = t.get(50728)
        cm1, cm2 = t.get(50721), t.get(50722)
        i1, i2 = t.get(50778), t.get(50779)
    if not neutral or not cm1 or len(cm1) != 9:
        return None
    cm1 = np.array(cm1, float).reshape(3, 3)
    cm2 = np.array(cm2, float).reshape(3, 3) if cm2 and len(cm2) == 9 else None
    t1 = _ILUMINANTES.get(int(i1 or 17), 2856)
    t2 = _ILUMINANTES.get(int(i2 or 21), 6504)
    return CameraColor(np.array(neutral[:3], float), cm1, cm2, t1, t2)


def _read_libraw(path):
    import rawpy
    with rawpy.imread(str(path)) as raw:
        mult = np.array(raw.camera_whitebalance[:3], float)
        cm = np.array(raw.rgb_xyz_matrix[:3], float)
    if not np.any(cm) or np.any(mult <= 0):
        return None
    return CameraColor(1.0 / mult, cm)


@functools.lru_cache(maxsize=64)
def _info_cached(path, mtime):
    try:
        info = _read_dng(path) if path.lower().endswith(".dng") else None
        return info or _read_libraw(path)
    except Exception:
        return None


def camera_color(path):
    """`CameraColor` de un RAW, o None si no se puede leer (JPG, DNG propio
    sin etiquetas...)."""
    try:
        return _info_cached(str(path), Path(path).stat().st_mtime)
    except Exception:
        return None


# --- aplicar -------------------------------------------------------------
# La foto llega con la gamma BT.709 con que la revela LibRaw; el balance se
# cambia en luz lineal. Ida y vuelta por tablas: pow() sobre 3 megapixeles
# por canal es lo que mas costaria.
_N = 4096
_g = np.linspace(0.0, 1.0, _N)
_A_LINEAL = np.where(_g < 0.081, _g / 4.5,
                     ((_g + 0.099) / 1.099) ** (1 / 0.45)).astype(np.float32)
_NL = 65536
_l = np.linspace(0.0, 1.0, _NL)
_A_GAMMA = np.where(_l < 0.018, _l * 4.5,
                    1.099 * _l ** 0.45 - 0.099).astype(np.float32)


def apply(img, m):
    """Aplica la matriz `m` (de `CameraColor.correction`) a la foto."""
    idx = np.clip(img, 0.0, 1.0)
    idx *= _N - 1
    idx += 0.5
    lin = _A_LINEAL[idx.astype(np.uint16)]
    lin = lin.reshape(-1, 3) @ m.T
    np.clip(lin, 0.0, 1.0, out=lin)
    lin *= _NL - 1
    lin += 0.5
    return _A_GAMMA[lin.astype(np.uint32)].reshape(img.shape)
