"""Catalogo de los modelos de IA de PhotoRAW.

Una sola lista con todo lo que la app puede descargar: que es, para que
herramienta sirve, cuanto pesa y si esta en el disco. La ventana "Modelos
de IA" se dibuja sola a partir de aqui, asi que para anadir una IA nueva
solo hay que sumar una entrada a AI_MODELS.

Los modelos NO van en el repositorio (son ~3,3 GB): se clona, se instalan
los requirements y desde esa ventana se baja lo que se quiera usar.
"""
import shutil
import urllib.request
from pathlib import Path

from photoraw import ai, face_parse, faces, generative, heal, masks_ai, upscale

MODEL_DIR = ai.MODEL_DIR


def _download_file(path, url, cb=None):
    """Descarga un modelo suelto a un temporal y lo renombra al final, para
    que un corte de red no deje un archivo a medias que parezca valido."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    def hook(blocks, block_size, total):
        if cb and total > 0:
            cb(min(blocks * block_size / total, 1.0))

    urllib.request.urlretrieve(url, tmp, hook)
    tmp.replace(path)


class Model:
    """Una IA descargable: lo que la ventana necesita saber de ella."""

    def __init__(self, key, name, tool, what, size_mb, paths,
                 installed, download, needed_by=None):
        self.key = key
        self.name = name            # nombre del modelo
        self.tool = tool            # herramienta de PhotoRAW que lo usa
        self.what = what            # que hace, en cristiano
        self.size_mb = size_mb      # peso aproximado de la descarga
        self.paths = tuple(paths)   # archivos/carpetas en disco
        self._installed = installed
        self._download = download
        self.needed_by = needed_by or ()   # otras herramientas que lo usan

    def installed(self):
        try:
            return bool(self._installed())
        except Exception:
            return False

    def download(self, cb=None):
        self._download(cb)

    def disk_bytes(self):
        total = 0
        for p in self.paths:
            if p.is_dir():
                total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            elif p.exists():
                total += p.stat().st_size
        return total

    def remove(self):
        """Borra el modelo del disco (se puede volver a descargar)."""
        for p in self.paths:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
        # y que la app no siga usando la copia que tenia en la GPU
        ai.release_all_sessions()


def _file_model(key, name, tool, what, size_mb, path, url, min_mb=10,
                needed_by=None):
    return Model(
        key, name, tool, what, size_mb, [path],
        installed=lambda: path.exists() and path.stat().st_size > min_mb * 1_000_000,
        download=lambda cb=None: _download_file(path, url, cb),
        needed_by=needed_by)


AI_MODELS = [
    _file_model(
        "scunet", "SCUNet", "Reducción de ruido (IA)",
        "Limpia el ruido de ISO alto conservando el detalle fino. Lo que "
        "hace el deslizador «Ruido IA».",
        88, ai.DENOISE_MODEL, ai.DENOISE_URL),

    _file_model(
        "codeformer", "CodeFormer", "Retoque de rostros",
        "Reconstruye caras con mano firme: rescata ojos y piel en fotos "
        "movidas o pequeñas. El más agresivo de los dos.",
        360, *faces.FACE_MODELS["CodeFormer"]),

    _file_model(
        "gfpgan", "GFPGAN 1.4", "Retoque de rostros",
        "Alternativa más suave: respeta más los rasgos originales. Elige "
        "uno u otro en el desplegable de la herramienta.",
        325, *faces.FACE_MODELS["GFPGAN"]),

    _file_model(
        "yunet", "YuNet", "Detector de caras",
        "No retoca nada: solo encuentra dónde están las caras. Hace falta "
        "para el retoque de rostros y para las máscaras de retrato.",
        1, faces.YUNET_MODEL, faces.YUNET_URL, min_mb=0.1,
        needed_by=("Retoque de rostros", "Máscaras de retrato")),

    _file_model(
        "bisenet", "BiSeNet", "Máscaras de retrato",
        "Divide cada cara en zonas: piel, cejas, ojos, labios, dientes y "
        "pelo, para ajustar cada una por separado.",
        90, face_parse.BISENET_MODEL, face_parse.BISENET_URL),

    _file_model(
        "u2net", "u2net", "Máscaras Sujeto / Fondo",
        "Recorta a las personas o al objeto principal de un clic, para "
        "editar sujeto y fondo por separado.",
        168, masks_ai.U2NET_MODEL, masks_ai.U2NET_URL),

    _file_model(
        "lama", "LaMa", "Pincel corrector",
        "Rellena lo que pintas continuando la textura de alrededor: granos, "
        "motas de sensor, cables. Rápido (~0,2 s). Sin él, el corrector usa "
        "el relleno clásico, que deja un borrón liso.",
        199, heal.LAMA_MODEL, heal.LAMA_URL, min_mb=50),

    _file_model(
        "esrgan", "Real-ESRGAN x4", "Superresolución",
        "Exporta a 2× o 4× reconstruyendo detalle real. También afina el "
        "parche del borrado generativo.",
        67, upscale.SR_MODEL, upscale.SR_URL),

    Model(
        "sd_inpaint", "Realistic Vision 5.1", "Borrar con IA (generativo)",
        "Hace desaparecer personas u objetos grandes imaginándose el fondo "
        "que había detrás. El más pesado y el más lento (~15-20 s), pero es "
        "el único que reconstruye escena de verdad.",
        2000, [generative.SD_DIR],
        installed=generative.model_available,
        download=lambda cb=None: generative.download_model(cb)),
]


def summary():
    """(instalados, total, bytes en disco) para la cabecera de la ventana."""
    hechos = [m for m in AI_MODELS if m.installed()]
    return len(hechos), len(AI_MODELS), sum(m.disk_bytes() for m in AI_MODELS)


def missing():
    return [m for m in AI_MODELS if not m.installed()]
