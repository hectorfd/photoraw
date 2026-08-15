"""Escritura de DNG lineal.

Un DNG normal, el que sale de tu camara, guarda lo que leyo el sensor tal
cual: un solo numero por pixel, con el mosaico de color todavia por
interpretar. Una fusion HDR ya no puede ser eso — al juntar tres tomas el
mosaico ya se ha interpretado y hay tres numeros por pixel, y no hay forma de
volver atras.

Para eso existe el DNG LINEAL, que es lo que Lightroom entrega al fusionar un
HDR: un DNG de verdad, con su matriz de color y su punto blanco, pero con la
imagen ya demosaicada. Sigue comportandose como un RAW —balance de blancos
reinterpretable, 16 bits, margen para revelar— sin fingir un mosaico que ya
no existe.

Por dentro un DNG es un TIFF con etiquetas de mas, asi que se escribe a mano:
son cuatro estructuras contadas y nos ahorra meter una dependencia entera
solo para esto.
"""
import struct
from pathlib import Path

import cv2
import numpy as np

# tipos de dato de TIFF
BYTE, ASCII, SHORT, LONG, RATIONAL, SRATIONAL = 1, 2, 3, 4, 5, 10

# Matriz de color: pasa de XYZ (con luz de dia D65) al color de "la camara".
# Aqui la camara es sRGB, porque la fusion ya entrega los colores ahi, asi
# que es la matriz XYZ->sRGB lineal de toda la vida. Sin esto el programa que
# abra el archivo no sabria que significan los numeros y el color bailaria.
XYZ_A_SRGB = [3.2406, -1.5372, -0.4986,
              -0.9689, 1.8758, 0.0415,
              0.0557, -0.2040, 1.0570]

# Marca de fabrica. `loader` la usa para reconocer sus propios DNG y no
# volver a exponerlos al abrirlos (ver photoraw/loader.py).
CAMARA = "PhotoRAW HDR"

THUMB_ANCHO = 256


def bt709_a_lineal(y):
    """Quita la curva de pantalla que pone LibRaw al revelar. Entra/sale 0..1.

    Es la inversa EXACTA de la curva que aplica rawpy por defecto (la de
    television, BT.709), y tiene que serlo: dentro del DNG la foto va en
    lineal, y al abrirlo LibRaw le pondra esa curva. Si aqui se usara la
    curva de sRGB —que se le parece pero no es igual— la foto volveria a
    salir 12 niveles mas oscura de como la dejaste en la vista previa.
    """
    y = np.clip(y, 0.0, 1.0)
    return np.where(y < 0.081, y / 4.5,
                    ((y + 0.099) / 1.099) ** (1 / 0.45)).astype(np.float32)


class _IFD:
    """Un directorio de etiquetas del TIFF."""

    def __init__(self):
        self.tags = {}

    def set(self, tag, typ, value):
        if not isinstance(value, (list, tuple)):
            value = [value]
        self.tags[tag] = (typ, list(value))

    def size(self):
        """Lo que ocupa: 12 bytes por etiqueta, mas la cuenta y el enlace."""
        return 2 + 12 * len(self.tags) + 4

    def build(self, extra_base):
        """(bytes del directorio, bytes de los valores largos).

        En TIFF, un valor de mas de 4 bytes no cabe dentro de su etiqueta: se
        guarda aparte y en la etiqueta queda su posicion. `extra_base` es
        donde empieza esa zona.
        """
        out, extra = bytearray(), bytearray()
        out += struct.pack("<H", len(self.tags))
        for tag in sorted(self.tags):           # el TIFF las quiere ordenadas
            typ, vals = self.tags[tag]
            if typ == ASCII:
                raw = vals[0].encode("ascii", "replace") + b"\0"
                count = len(raw)
            elif typ in (RATIONAL, SRATIONAL):
                fmt = "<ii" if typ == SRATIONAL else "<II"
                raw = b"".join(struct.pack(fmt, *v) for v in vals)
                count = len(vals)
            else:
                fmt = {BYTE: "<B", SHORT: "<H", LONG: "<I"}[typ]
                raw = b"".join(struct.pack(fmt, v) for v in vals)
                count = len(vals)
            out += struct.pack("<HHI", tag, typ, count)
            if len(raw) <= 4:
                out += raw + b"\0" * (4 - len(raw))
            else:
                out += struct.pack("<I", extra_base + len(extra))
                extra += raw
                if len(extra) % 2:              # las posiciones van pares
                    extra += b"\0"
        out += struct.pack("<I", 0)             # no hay otro directorio detras
        return bytes(out), bytes(extra)


def _rat(v, den=10000):
    return int(round(v * den)), den


def save(rgb01, dest, gamma=2.2):
    """Guarda RGB 0..1 (ya revelado, como se ve) en un DNG lineal de 16 bits.

    Entra la foto tal y como la dejaste en la vista previa y dentro del
    archivo se guarda en lineal, que es lo que espera un DNG. Al abrirlo,
    el revelado le devuelve la curva y sale igual que la viste.
    """
    dest = Path(dest)
    visible = np.clip(np.asarray(rgb01, np.float32), 0.0, 1.0)
    lineal = bt709_a_lineal(visible)
    data = (lineal * 65535.0 + 0.5).astype("<u2")
    h, w = data.shape[:2]

    # la previa incrustada: lo que ensena el explorador de archivos
    alto = max(1, int(round(THUMB_ANCHO * h / w)))
    thumb = cv2.resize((visible * 255.0 + 0.5).astype(np.uint8),
                       (THUMB_ANCHO, alto), interpolation=cv2.INTER_AREA)
    thumb_bytes = thumb.tobytes()
    raw_bytes = data.tobytes()

    ifd0, sub = _IFD(), _IFD()

    # IFD0: la vista previa. El DNG manda que la imagen de verdad no vaya
    # aqui sino en un sub-directorio, y que esto sea solo la miniatura.
    ifd0.set(254, LONG, 1)                       # esto es una version reducida
    ifd0.set(256, LONG, thumb.shape[1])
    ifd0.set(257, LONG, thumb.shape[0])
    ifd0.set(258, SHORT, [8, 8, 8])
    ifd0.set(259, SHORT, 1)                      # sin comprimir
    ifd0.set(262, SHORT, 2)                      # RGB
    ifd0.set(273, LONG, 0)                       # donde estan los datos
    ifd0.set(277, SHORT, 3)
    ifd0.set(278, LONG, thumb.shape[0])
    ifd0.set(279, LONG, len(thumb_bytes))
    ifd0.set(284, SHORT, 1)
    ifd0.set(271, ASCII, "PhotoRAW")             # Make
    ifd0.set(272, ASCII, CAMARA)                 # Model
    ifd0.set(305, ASCII, CAMARA)                 # Software
    ifd0.set(330, LONG, 0)                       # donde esta la imagen buena
    ifd0.set(50706, BYTE, [1, 4, 0, 0])          # DNGVersion
    ifd0.set(50707, BYTE, [1, 1, 0, 0])          # version minima para leerlo
    ifd0.set(50708, ASCII, CAMARA)               # UniqueCameraModel
    ifd0.set(50721, SRATIONAL, [_rat(v) for v in XYZ_A_SRGB])
    ifd0.set(50778, SHORT, 21)                   # la matriz es para luz D65
    ifd0.set(50728, RATIONAL, [_rat(1.0)] * 3)   # el blanco ya esta puesto

    # el sub-directorio con la foto de verdad
    sub.set(254, LONG, 0)                        # esta es la principal
    sub.set(256, LONG, w)
    sub.set(257, LONG, h)
    sub.set(258, SHORT, [16, 16, 16])
    sub.set(259, SHORT, 1)
    sub.set(262, SHORT, 34892)                   # LinearRaw: ya demosaicada
    sub.set(273, LONG, 0)
    sub.set(277, SHORT, 3)
    sub.set(278, LONG, h)
    sub.set(279, LONG, len(raw_bytes))
    sub.set(284, SHORT, 1)
    sub.set(339, SHORT, [1, 1, 1])               # enteros sin signo
    sub.set(50714, LONG, 0)                      # el negro esta en 0
    sub.set(50717, LONG, 65535)                  # y el blanco arriba del todo

    # Colocacion. Hay que armarlo dos veces: la primera para saber cuanto
    # ocupa cada trozo y la segunda ya con las posiciones definitivas.
    off_ifd0 = 8
    off_x0 = off_ifd0 + ifd0.size()
    _, x0 = ifd0.build(off_x0)
    off_sub = off_x0 + len(x0)
    off_x1 = off_sub + sub.size()
    _, x1 = sub.build(off_x1)
    off_thumb = off_x1 + len(x1)
    off_raw = off_thumb + len(thumb_bytes)

    ifd0.set(273, LONG, off_thumb)
    ifd0.set(330, LONG, off_sub)
    sub.set(273, LONG, off_raw)
    b0, x0 = ifd0.build(off_x0)
    b1, x1 = sub.build(off_x1)

    tmp = dest.with_suffix(dest.suffix + ".parcial")
    try:
        with open(tmp, "wb") as f:
            f.write(struct.pack("<2sHI", b"II", 42, off_ifd0))
            f.write(b0)
            f.write(x0)
            f.write(b1)
            f.write(x1)
            f.write(thumb_bytes)
            f.write(raw_bytes)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def es_nuestro(path):
    """Si este archivo es un DNG de los que escribe PhotoRAW.

    Se mira la etiqueta Software del TIFF, sin abrir la foto entera.
    """
    path = Path(path)
    if path.suffix.lower() != ".dng":
        return False
    try:
        with open(path, "rb") as f:
            cabecera = f.read(8)
            if cabecera[:2] not in (b"II", b"MM"):
                return False
            orden = "<" if cabecera[:2] == b"II" else ">"
            off = struct.unpack(orden + "I", cabecera[4:8])[0]
            f.seek(off)
            n = struct.unpack(orden + "H", f.read(2))[0]
            for _ in range(min(n, 200)):
                tag, typ, count = struct.unpack(orden + "HHI", f.read(8))
                valor = f.read(4)
                if tag != 305:                   # Software
                    continue
                if count <= 4:
                    texto = valor[:count]
                else:
                    pos = struct.unpack(orden + "I", valor)[0]
                    aqui = f.tell()
                    f.seek(pos)
                    texto = f.read(count)
                    f.seek(aqui)
                return texto.split(b"\0")[0].decode("ascii", "ignore") == CAMARA
    except (OSError, struct.error):
        return False
    return False
