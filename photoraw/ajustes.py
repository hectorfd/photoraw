"""Ajustes de la aplicacion que sobreviven a cerrarla.

Cosas pequenas que no son de ninguna foto en concreto sino de como te dejaste
el programa: sobre todo, en que carpeta estabas trabajando. Van a un JSON en
`~/.photoraw`, donde ya viven la cache y los modelos de IA.

No confundir con `edits.py`, que guarda el revelado de cada foto: eso vive en
la carpeta de las fotos y viaja con ellas. Esto es del ordenador, no del
trabajo.
"""
import json
from pathlib import Path

ARCHIVO = Path.home() / ".photoraw" / "ajustes.json"


def _leer():
    try:
        with open(ARCHIVO, encoding="utf-8") as f:
            datos = json.load(f)
        return datos if isinstance(datos, dict) else {}
    except (OSError, ValueError):
        return {}


def get(clave, defecto=None):
    return _leer().get(clave, defecto)


def set(clave, valor):
    """Guarda un ajuste. Si no se puede escribir, no pasa nada: son
    comodidades, y ninguna vale una ventana de error al cerrar."""
    datos = _leer()
    datos[clave] = valor
    try:
        ARCHIVO.parent.mkdir(parents=True, exist_ok=True)
        tmp = ARCHIVO.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        tmp.replace(ARCHIVO)
    except OSError:
        pass


def ultima_carpeta():
    """La carpeta donde estabas la ultima vez, si todavia existe.

    Se comprueba que siga ahi porque es muy normal que estuviera en una
    tarjeta de memoria o en un disco USB que ya no esta conectado.
    """
    valor = get("ultima_carpeta")
    if not valor:
        return None
    carpeta = Path(valor)
    try:
        return carpeta if carpeta.is_dir() else None
    except OSError:
        return None


def guardar_carpeta(carpeta):
    set("ultima_carpeta", str(Path(carpeta)))
