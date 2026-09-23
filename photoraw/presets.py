"""Preajustes del usuario, guardados en ~/.photoraw/presets/*.json"""
import copy
import json
import re
from pathlib import Path

from photoraw.engine import DEFAULT_EDITS

PRESET_DIR = Path.home() / ".photoraw" / "presets"

# Cosas que son de UNA foto concreta y jamas viajan en un preajuste: los
# trazos del corrector y los borrados generativos llevan coordenadas y
# mapas pintados sobre esa imagen, no significan nada en otra.
PER_PHOTO_KEYS = ("heal_strokes", "erase_ops")


def _keys(*names_or_prefixes):
    """Claves de DEFAULT_EDITS por nombre exacto o por prefijo ('hsl_')."""
    out = []
    for n in names_or_prefixes:
        if n.endswith("_") or n.endswith("*"):
            pref = n.rstrip("*")
            out += [k for k in DEFAULT_EDITS if k.startswith(pref)]
        elif n in DEFAULT_EDITS:
            out.append(n)
    return tuple(dict.fromkeys(out))


# Bloques que se pueden marcar al guardar un preajuste. El ultimo campo dice
# si viene marcado de serie: recorte y mascaras NO, porque estan pensados
# sobre una foto concreta y casi nunca sirven igual en otra.
GROUPS = [
    ("profile",  "Perfil de color",        _keys("profile"), True),
    ("wb",       "Balance de blancos",     _keys("temperature", "tint",
                                                   "wb_temp", "wb_tint"), True),
    ("tone",     "Luz y tono",
     _keys("exposure", "contrast", "highlights", "shadows", "whites",
           "blacks", "tone_map", "adaptive_contrast"), True),
    ("curves",   "Curvas",
     _keys("p_highlights", "p_lights", "p_darks", "p_shadows", "curve_"), True),
    ("color",    "Color (saturación, HSL, color de punto)",
     _keys("saturation", "vibrance", "hsl_", "pc_"), True),
    ("presence", "Presencia (claridad, textura, neblina)",
     _keys("clarity", "texture", "dehaze"), True),
    ("detail",   "Detalle (enfoque y ruido)", _keys("sharp_", "nr_"), True),
    ("ai",       "Ruido y rostros con IA",  _keys("ai_denoise", "ai_face"), True),
    ("effects",  "Efectos (dramático, virado, mate, brillo…)",
     _keys("dramatic_", "mood_", "tone_hi_", "tone_sh_", "tone_balance",
           "matte_", "mystical_", "glow_"), True),
    ("grain",    "Grano de película",       _keys("grain_"), True),
    ("calib",    "Calibración de cámara",   _keys("cal_"), True),
    ("masks",    "Máscaras (degradados, pincel, IA)", _keys("masks"), False),
    ("crop",     "Recorte, giro y enderezado",
     _keys("crop", "straighten", "rot90", "flip_h", "flip_v"), False),
]

_ASSIGNED = {k for _i, _l, keys, _d in GROUPS for k in keys}
_REST = tuple(k for k in DEFAULT_EDITS
              if k not in _ASSIGNED and k not in PER_PHOTO_KEYS)
if _REST:   # red de seguridad si algun dia se anaden ajustes nuevos
    GROUPS.append(("other", "Otros ajustes", _REST, True))

GROUP_KEYS = {gid: keys for gid, _l, keys, _d in GROUPS}
ALL_GROUPS = [gid for gid, _l, _k, _d in GROUPS]


def changed_keys(edits, group_id):
    """Claves de ese bloque que estan tocadas (distintas de fabrica)."""
    return [k for k in GROUP_KEYS.get(group_id, ())
            if edits.get(k, DEFAULT_EDITS.get(k)) != DEFAULT_EDITS.get(k)]


def _safe_name(name):
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "preset"


def list_presets():
    if not PRESET_DIR.exists():
        return []
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def save_preset(name, edits, groups=None):
    """Guarda los ajustes tocados de los bloques elegidos.

    `groups` es la lista de bloques marcados en la ventana de guardado; sin
    ella se guarda todo (lo que hacia antes) para no romper llamadas viejas.
    """
    if groups is None:
        groups = ALL_GROUPS
    wanted = {k for gid in groups for k in GROUP_KEYS.get(gid, ())}
    clean = {k: v for k, v in edits.items()
             if k in wanted and k not in PER_PHOTO_KEYS
             and v != DEFAULT_EDITS.get(k)}
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    path = PRESET_DIR / f"{_safe_name(name)}.json"
    path.write_text(
        json.dumps({"_v": 2, "_groups": list(groups), "edits": clean},
                   indent=1),
        encoding="utf-8")


def _read(name):
    path = PRESET_DIR / f"{_safe_name(name)}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_preset(name):
    """Solo los valores guardados."""
    data = _read(name)
    return data.get("edits", {}) if data.get("_v") else data


def load_groups(name):
    """Bloques que guarda el preajuste, o None si es de los antiguos (que
    no lo apuntaban y se aplican enteros, como se venia haciendo)."""
    data = _read(name)
    return data.get("_groups") if data.get("_v") else None


def apply_to(edits, saved, groups):
    """Aplica el preajuste sobre unos ajustes existentes.

    Solo se tocan los bloques que el preajuste guarda: dentro de ellos manda
    el preajuste (y lo que no traiga vuelve a fabrica, que es lo que hace
    que el resultado sea siempre el mismo), y todo lo demas de la foto -- su
    recorte, sus mascaras -- se queda exactamente como estaba."""
    out = copy.deepcopy(dict(edits))
    for gid in groups:
        for key in GROUP_KEYS.get(gid, ()):
            out[key] = copy.deepcopy(
                saved[key] if key in saved else DEFAULT_EDITS.get(key))
    return out


def summary(name):
    """Texto corto con los bloques que lleva el preajuste, para la interfaz."""
    groups = load_groups(name)
    if groups is None:
        return "Preajuste antiguo: se aplica entero (recorte incluido)"
    labels = {gid: lab for gid, lab, _k, _d in GROUPS}
    nombres = [labels.get(g, g) for g in groups]
    return "Lleva: " + (" · ".join(nombres) if nombres else "nada")


def delete_preset(name):
    path = PRESET_DIR / f"{_safe_name(name)}.json"
    try:
        path.unlink()
    except Exception:
        pass
