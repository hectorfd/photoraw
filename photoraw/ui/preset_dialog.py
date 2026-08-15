"""Ventana de guardado de preajustes: elegir QUE se lleva a otras fotos.

Antes, guardar un preajuste se llevaba todo lo que estuviera tocado, y eso
incluia el recorte, el giro y las mascaras de esa foto en concreto. Aplicarlo
a otra foto le encajaba el encuadre de la primera. Aqui se ve, bloque por
bloque, lo que hay tocado y se marca lo que de verdad interesa repetir.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QLabel,
                               QLineEdit, QPushButton, QHBoxLayout,
                               QScrollArea, QVBoxLayout, QWidget)

from photoraw import presets


class PresetSaveDialog(QDialog):
    """Al aceptar deja el nombre en `name` y los bloques marcados en `groups`."""

    MAX_NAMES = 4   # cuantos ajustes se nombran antes de resumir con "y N mas"

    def __init__(self, edits, labels, parent=None, existing=()):
        super().__init__(parent)
        self.setWindowTitle("Guardar preajuste")
        self.setMinimumWidth(430)
        self._edits = edits
        self._labels = labels
        self._existing = {e.lower() for e in existing}
        self.name = ""
        self.groups = []
        self._boxes = {}
        self._build()

    # ---------- interfaz ----------

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(10)

        row = QHBoxLayout()
        row.addWidget(QLabel("Nombre"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Ej.: Retrato cálido, B&N contrastado…")
        self.name_edit.textChanged.connect(self._update_ok)
        row.addWidget(self.name_edit, 1)
        root.addLayout(row)

        head = QLabel("Marca lo que quieres poder aplicar luego a otras fotos. "
                      "Lo que dejes sin marcar no se guarda, y al aplicar el "
                      "preajuste esas fotos lo conservarán tal como lo tengan.")
        head.setWordWrap(True)
        head.setStyleSheet("color: #8a8a8a;")
        root.addWidget(head)

        inner = QWidget()
        box = QVBoxLayout(inner)
        box.setContentsMargins(2, 2, 2, 2)
        box.setSpacing(4)
        for gid, label, _keys, default_on in presets.GROUPS:
            changed = presets.changed_keys(self._edits, gid)
            cb = QCheckBox(self._text(label, changed))
            cb.setChecked(bool(changed) and default_on)
            cb.setEnabled(bool(changed))
            if not changed:
                cb.setStyleSheet("color: #6f6f6f;")
            elif gid in ("crop", "masks"):
                cb.setToolTip(
                    "Está pensado sobre ESTA foto: al aplicarlo a otra le "
                    "encajará el mismo encuadre o las mismas zonas, que rara "
                    "vez caen donde toca. Márcalo solo si es lo que buscas.")
            self._boxes[gid] = cb
            box.addWidget(cb)
        box.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(260)
        root.addWidget(scroll, 1)

        marcar = QHBoxLayout()
        for texto, valor in (("Marcar todo", True), ("Desmarcar todo", False)):
            b = QPushButton(texto)
            b.clicked.connect(lambda _c=False, v=valor: self._set_all(v))
            marcar.addWidget(b)
        marcar.addStretch(1)
        root.addLayout(marcar)

        self.warn = QLabel()
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color: #d8a04a;")
        root.addWidget(self.warn)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.ok_button = self.buttons.addButton("Guardar",
                                                QDialogButtonBox.AcceptRole)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        for cb in self._boxes.values():
            cb.toggled.connect(self._update_ok)
        self._update_ok()

    def _text(self, label, changed):
        """'Luz y tono — Exposición, Contraste y 2 más' / '(sin cambios)'."""
        if not changed:
            return f"{label}  —  sin cambios"
        nombres = [self._labels.get(k, k) for k in changed[:self.MAX_NAMES]]
        resto = len(changed) - len(nombres)
        if resto > 0:
            nombres.append(f"y {resto} más")
        return f"{label}  —  " + ", ".join(nombres)

    def _set_all(self, value):
        for cb in self._boxes.values():
            if cb.isEnabled():
                cb.setChecked(value)

    # ---------- estado ----------

    def _marked(self):
        return [gid for gid, cb in self._boxes.items() if cb.isChecked()]

    def _update_ok(self):
        nombre = self.name_edit.text().strip()
        marcados = self._marked()
        self.ok_button.setEnabled(bool(nombre) and bool(marcados))
        if not marcados:
            self.warn.setText("Marca al menos un bloque.")
        elif nombre.lower() in self._existing:
            self.warn.setText(f"Ya existe un preajuste «{nombre}»: se "
                              "reemplazará.")
        else:
            self.warn.setText("")

    def _accept(self):
        self.name = self.name_edit.text().strip()
        self.groups = self._marked()
        if self.name and self.groups:
            self.accept()
