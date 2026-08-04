"""Guardado no destructivo de ajustes: un JSON por carpeta de fotos."""
import json
from pathlib import Path

from photoraw.engine import DEFAULT_EDITS

STORE_NAME = ".photoraw_edits.json"


class EditStore:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.path = self.folder / STORE_NAME
        self.data = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def get(self, photo_path):
        return dict(self.data.get(Path(photo_path).name, {}))

    def set(self, photo_path, edits):
        # Solo guarda valores distintos del predeterminado
        clean = {k: v for k, v in edits.items() if v != DEFAULT_EDITS.get(k)}
        name = Path(photo_path).name
        if clean:
            self.data[name] = clean
        else:
            self.data.pop(name, None)
        self.save()

    def save(self):
        try:
            self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        except Exception:
            pass
