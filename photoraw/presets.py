"""Preajustes del usuario, guardados en ~/.photoraw/presets/*.json"""
import json
import re
from pathlib import Path

from photoraw.engine import DEFAULT_EDITS

PRESET_DIR = Path.home() / ".photoraw" / "presets"


def _safe_name(name):
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "preset"


def list_presets():
    if not PRESET_DIR.exists():
        return []
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def save_preset(name, edits):
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in edits.items()
             if v != DEFAULT_EDITS.get(k) and k != "heal_strokes"}
    path = PRESET_DIR / f"{_safe_name(name)}.json"
    path.write_text(json.dumps(clean, indent=1), encoding="utf-8")


def load_preset(name):
    path = PRESET_DIR / f"{_safe_name(name)}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def delete_preset(name):
    path = PRESET_DIR / f"{_safe_name(name)}.json"
    try:
        path.unlink()
    except Exception:
        pass
