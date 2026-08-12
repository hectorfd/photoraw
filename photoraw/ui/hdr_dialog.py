"""Dialogo de fusion de exposiciones (bracketing HDR).

Ensena como va a quedar el HDR antes de gastar el rato de fusionarlo a
resolucion completa: las tomas se cargan una sola vez en pequeno y, a partir
de ahi, mover un ajuste recalcula la vista previa en un instante.
"""
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QSize, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QVBoxLayout, QWidget)

from photoraw import diskcache, hdr

PREVIEW_SIDE = 1100   # lado max de las tomas para la vista previa
THUMB_SIDE = 96       # miniaturas de las tomas de origen


def _mw():
    """Import diferido de la ventana principal.

    main_window importa este dialogo, asi que importarlo aqui arriba seria
    circular. De aqui salen dos utilidades que ya existen y no merece la pena
    duplicar: np_to_qimage y el deslizador con doble clic para restablecer.
    """
    from photoraw.ui import main_window
    return main_window


class _Signals(QObject):
    shots_ready = Signal(int, object)          # tanda, lista de tomas (o None)
    fused_ready = Signal(int, int, object)     # tanda, generacion, resultado


def _avisar(señal, *args):
    """Emite una señal aguantando que el dialogo ya no exista.

    Cerrar la ventana no cancela la decodificacion de los RAW, que tarda un
    par de segundos: si le das a Cancelar mientras carga, el trabajo termina
    despues de que el dialogo se haya ido y al avisar se encuentra con que ya
    no hay nadie escuchando. No es un problema —ese resultado no le importa a
    nadie—, pero sin esto salta un RuntimeError desde el hilo de trabajo.
    """
    try:
        señal.emit(*args)
    except RuntimeError:
        pass


class _LoadJob(QRunnable):
    """Carga las tomas de una tanda en pequeno. Es la parte lenta: hay que
    decodificar un RAW por toma."""

    def __init__(self, index, paths, signals):
        super().__init__()
        self.index = index
        self.paths = paths
        self.signals = signals

    def run(self):
        try:
            shots = hdr.load_group(self.paths, half_size=True,
                                   max_side=PREVIEW_SIDE)
        except Exception:
            shots = None
        _avisar(self.signals.shots_ready, self.index, shots)


class _FuseJob(QRunnable):
    """Fusiona unas tomas ya cargadas (decimas de segundo en pequeno)."""

    def __init__(self, index, gen, shots, params, exps, signals):
        super().__init__()
        self.index = index
        self.gen = gen
        self.shots = shots
        self.params = params
        self.exps = exps
        self.signals = signals

    def run(self):
        try:
            out = hdr.fuse(self.shots, self.params, exps=self.exps)
        except Exception:
            out = None
        _avisar(self.signals.fused_ready, self.index, self.gen, out)


# Mandos de cada metodo: (clave, etiqueta, minimo, maximo, formato, ayuda).
# Los valores van x100 en el deslizador y el de fabrica sale de hdr.DEFAULT_PARAMS.
SLIDERS = {
    hdr.NATURAL: [
        ("contrast", "Detalle", 0, 200, "pct",
         "Cuanto pesa la textura al elegir de que toma sale cada zona.\n"
         "Subirlo saca mas relieve en la piedra, las nubes y la hoja;\n"
         "pasarse deja el aspecto de HDR artificial."),
        ("saturation", "Color", 0, 200, "pct",
         "Da preferencia a la toma con el color mas vivo en cada zona.\n"
         "Ayuda con los cielos al atardecer."),
        ("exposure", "Equilibrio", 0, 200, "pct",
         "Cuanto tira el resultado hacia el gris medio. Subirlo suaviza\n"
         "los cambios entre tomas y da un resultado mas plano y seguro;\n"
         "bajarlo deja mandar al detalle y al color."),
    ],
    hdr.HDR: [
        ("intensity", "Brillo", -400, 400, "num",
         "Sube o baja toda la foto al comprimir el rango.\n"
         "Es el mando gordo: empieza por aqui."),
        ("gamma", "Gamma", 50, 350, "num",
         "Reparto entre sombras y luces. El mapa de luz que sale de la\n"
         "fusion es lineal, y 2,2 es lo que lo pasa a como lo ve el ojo.\n"
         "Subirlo aclara los medios tonos y abre las sombras; bajarlo los\n"
         "oscurece y da mas cuerpo."),
        ("light", "Contraste local", 0, 100, "pct",
         "Si manda el contraste de cada zona (100 %) o el de la foto\n"
         "entera (0 %). Alto saca mucho detalle local, pero pasarse es\n"
         "lo que produce el aspecto de HDR de calendario."),
        ("color", "Fidelidad de color", 0, 100, "pct",
         "A 0 % cada canal se comprime por su cuenta y los colores salen\n"
         "mas vivos; al subirlo se respeta mejor el color original de la\n"
         "escena, a costa de viveza."),
    ],
}


class HdrDialog(QDialog):
    """Elige la tanda, ajusta la mezcla y mira como queda.

    Al aceptar deja en `groups_to_merge` las tandas marcadas y en `params`
    los ajustes; la fusion de verdad, a resolucion completa, la hace la
    ventana principal.
    """

    def __init__(self, groups, parent=None, auto_detected=False):
        super().__init__(parent)
        self.setWindowTitle("Fusionar HDR")
        self.groups = [list(g) for g in groups]
        self.params = dict(hdr.DEFAULT_PARAMS)
        self.groups_to_merge = list(self.groups)

        self.pool = parent.fast_pool if parent is not None else None
        if self.pool is None:
            from PySide6.QtCore import QThreadPool
            self.pool = QThreadPool.globalInstance()

        self.signals = _Signals()
        self.signals.shots_ready.connect(self._on_shots_ready)
        self.signals.fused_ready.connect(self._on_fused_ready)

        self._shots = {}       # tanda -> tomas cargadas
        self._loading = set()
        self._gen = 0          # para descartar vistas previas que ya no valen
        self._current = 0
        # exposiciones de cada tanda (None si al EXIF le falta el dato): sin
        # ellas no se puede hacer HDR real, solo la fusion natural
        self._exps = {i: hdr.exposures(g) for i, g in enumerate(self.groups)}

        self._build_ui(auto_detected)
        if not any(v is not None for v in self._exps.values()):
            # ninguna tanda lo soporta: no se ofrece y se explica por que
            self.method_box.setCurrentIndex(0)
            self.method_box.model().item(1).setEnabled(False)
            self.method_box.setItemData(
                1, "Estas tomas no traen el tiempo de exposición ni el ISO en "
                   "el EXIF, y sin eso no se puede reconstruir la luz de la "
                   "escena.", Qt.ToolTipRole)
        self._request_shots(0)

    # ---------- interfaz ----------

    def _build_ui(self, auto_detected):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(10)

        if auto_detected:
            head = QLabel(
                f"He encontrado <b>{len(self.groups)}</b> tanda(s) de bracketing "
                "en la carpeta. Desmarca las que no quieras fusionar.")
        else:
            head = QLabel(
                f"Se van a fusionar <b>{len(self.groups[0])}</b> tomas en una "
                "sola foto.")
        head.setWordWrap(True)
        root.addWidget(head)

        middle = QHBoxLayout()
        middle.setSpacing(12)
        root.addLayout(middle, 1)

        # lista de tandas: solo cuando hay varias que elegir
        self.group_list = None
        if len(self.groups) > 1:
            self.group_list = QListWidget()
            self.group_list.setFixedWidth(210)
            for g in self.groups:
                item = QListWidgetItem(hdr.describe_group(g))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.group_list.addItem(item)
            self.group_list.setCurrentRow(0)
            self.group_list.currentRowChanged.connect(self._on_group_changed)
            self.group_list.itemChanged.connect(self._on_check_changed)
            middle.addWidget(self.group_list)

        # vista previa del resultado, con las tomas de origen debajo
        left = QVBoxLayout()
        left.setSpacing(8)
        self.preview = QLabel("Cargando las tomas…")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(QSize(520, 380))
        self.preview.setFrameShape(QFrame.StyledPanel)
        self.preview.setStyleSheet(
            "background: #141414; color: #8a8a8a; border: 1px solid #2a2a2a;")
        left.addWidget(self.preview, 1)

        self.shots_row = QHBoxLayout()
        self.shots_row.setSpacing(6)
        self.shots_row.addStretch(1)
        shots_box = QWidget()
        shots_box.setLayout(self.shots_row)
        shots_box.setFixedHeight(THUMB_SIDE + 8)
        left.addWidget(shots_box)
        middle.addLayout(left, 1)

        # ajustes de la mezcla
        panel = QVBoxLayout()
        panel.setSpacing(6)

        met_fila = QHBoxLayout()
        met_fila.addWidget(QLabel("Método:"))
        self.method_box = _mw().NoWheelCombo()
        self.method_box.addItem("Natural (fusión de exposiciones)", hdr.NATURAL)
        self.method_box.addItem("HDR real (rango completo)", hdr.HDR)
        self.method_box.setToolTip(
            "Natural: mira las tomas zona por zona y se queda con la mejor de\n"
            "cada una. Es el «HDR natural» tipo Lightroom, rápido y creíble.\n\n"
            "HDR real: reconstruye cuánta luz recibió de verdad la escena\n"
            "(necesita el EXIF de las tomas) y luego la comprime para que\n"
            "quepa en la pantalla. Aguanta mejor las luces altas y da más\n"
            "margen, pero pasarse con los mandos deja el aspecto artificial.")
        self.method_box.currentIndexChanged.connect(self._on_method)
        met_fila.addWidget(self.method_box, 1)
        panel.addLayout(met_fila)

        self.method_note = QLabel()
        self.method_note.setWordWrap(True)
        self.method_note.setStyleSheet("color: #d8a13a;")
        self.method_note.hide()
        panel.addWidget(self.method_note)

        # un grupo de mandos por metodo; se ensena el del metodo elegido
        self.sliders = {}
        self.slider_boxes = {}
        for metodo, spec in SLIDERS.items():
            caja = QWidget()
            grid = QGridLayout(caja)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(6)
            for row, (key, label, lo, hi, fmt, tip) in enumerate(spec):
                default = int(round(hdr.DEFAULT_PARAMS[key] * 100))
                name = QLabel(label)
                name.setToolTip(tip)
                slider = _mw().NoWheelSlider(Qt.Horizontal)
                slider.setRange(lo, hi)
                slider.setValue(default)
                slider.default_value = default
                slider.setToolTip(tip + "\n\nDoble clic para restablecer")
                value = QLabel()
                value.setFixedWidth(52)
                value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                value.setText(self._fmt(default, fmt))
                slider.valueChanged.connect(
                    lambda v, k=key, lab=value, f=fmt: self._on_slider(k, v, lab, f))
                grid.addWidget(name, row, 0)
                grid.addWidget(slider, row, 1)
                grid.addWidget(value, row, 2)
                self.sliders[key] = slider
            self.slider_boxes[metodo] = caja
            panel.addWidget(caja)
        self.slider_boxes[hdr.HDR].hide()

        self.align_box = QCheckBox("Alinear las tomas")
        self.align_box.setChecked(bool(hdr.DEFAULT_PARAMS["align"]))
        self.align_box.setToolTip(
            "Corrige el pequeño movimiento entre tomas cuando disparas a\n"
            "pulso. Con tripode puedes desmarcarlo y va algo mas rapido.\n"
            "No arregla lo que se movio dentro de la escena (ramas, agua,\n"
            "gente): ahi el HDR puede dejar contornos dobles.")
        self.align_box.toggled.connect(self._on_align)
        panel.addWidget(self.align_box)

        note = QLabel(
            "El resultado se guarda como TIFF de 16 bits junto a tus RAW y se "
            "abre en la tira para que lo reveles como cualquier otra foto. No "
            "es un RAW: ya lleva el color interpretado, pero con 16 bits "
            "aguanta que estires sombras y luces sin bandas. Sale plano a "
            "propósito, con todo el margen recogido: dale contraste y punto "
            "negro a tu gusto.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #8a8a8a;")
        panel.addWidget(note)
        panel.addStretch(1)

        panel_box = QWidget()
        panel_box.setLayout(panel)
        panel_box.setFixedWidth(300)
        middle.addWidget(panel_box)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.ok_button = self.buttons.addButton("Fusionar",
                                               QDialogButtonBox.AcceptRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        self._update_ok_button()

    # ---------- carga y vista previa ----------

    def _request_shots(self, index):
        """Pide las tomas de una tanda (si no estan ya) y refresca la vista."""
        self._current = index
        self._show_source_thumbs(index)
        self._update_method_note()
        if index in self._shots:
            self._request_preview()
            return
        self.preview.setText("Cargando las tomas…")
        self.preview.setPixmap(QPixmap())
        if index not in self._loading:
            self._loading.add(index)
            self.pool.start(_LoadJob(index, self.groups[index], self.signals))

    def _on_shots_ready(self, index, shots):
        self._loading.discard(index)
        if shots is None:
            if index == self._current:
                self.preview.setText("No se han podido leer estas tomas")
            return
        self._shots[index] = shots
        if index == self._current:
            self._request_preview()

    def _request_preview(self):
        """Refusiona la vista previa. Cada peticion estrena generacion, asi
        que la que llegue de un ajuste ya viejo se tira."""
        shots = self._shots.get(self._current)
        if not shots:
            return
        self._gen += 1
        self.pool.start(_FuseJob(self._current, self._gen, shots,
                                 dict(self.params),
                                 self._exps.get(self._current), self.signals))

    def _on_fused_ready(self, index, gen, out):
        if gen != self._gen or index != self._current:
            return   # llego tarde: ya hay otro ajuste en marcha
        if out is None:
            self.preview.setText("No se ha podido fusionar esta tanda")
            return
        rgb = np.ascontiguousarray((out * 255.0 + 0.5).astype(np.uint8))
        self._preview_pixmap = QPixmap.fromImage(_mw().np_to_qimage(rgb))
        self._paint_preview()

    def _paint_preview(self):
        pm = getattr(self, "_preview_pixmap", None)
        if pm is None or pm.isNull():
            return
        self.preview.setPixmap(pm.scaled(self.preview.size(),
                                         Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation))

    def _show_source_thumbs(self, index):
        """Las tomas de origen debajo de la vista previa, de oscura a clara,
        para ver de un vistazo con que material se esta trabajando."""
        while self.shots_row.count() > 1:
            item = self.shots_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for path in self.groups[index]:
            label = QLabel()
            label.setFixedSize(THUMB_SIDE, THUMB_SIDE)
            label.setAlignment(Qt.AlignCenter)
            label.setToolTip(Path(path).name)
            label.setStyleSheet("border: 1px solid #2a2a2a;")
            data = diskcache.thumb_jpeg(path)
            if data:
                pm = QPixmap()
                if pm.loadFromData(data):
                    label.setPixmap(pm.scaled(THUMB_SIDE, THUMB_SIDE,
                                              Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation))
            self.shots_row.insertWidget(self.shots_row.count() - 1, label)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._paint_preview()

    # ---------- reacciones ----------

    @staticmethod
    def _fmt(value, fmt):
        return f"{value} %" if fmt == "pct" else f"{value / 100.0:.2f}"

    def _on_slider(self, key, value, label, fmt="pct"):
        label.setText(self._fmt(value, fmt))
        self.params[key] = value / 100.0
        self._debounce()

    def _on_method(self, _idx):
        metodo = self.method_box.currentData()
        self.params["method"] = metodo
        for nombre, caja in self.slider_boxes.items():
            caja.setVisible(nombre == metodo)
        self._update_method_note()
        self._request_preview()

    def _update_method_note(self):
        """Avisa si la tanda que se esta viendo no puede hacer HDR real."""
        if (self.params.get("method") == hdr.HDR
                and self._exps.get(self._current) is None):
            self.method_note.setText(
                "Esta tanda no trae los datos de exposición en el EXIF, así "
                "que se fusiona con el método natural.")
            self.method_note.show()
        else:
            self.method_note.hide()

    def _on_align(self, on):
        self.params["align"] = bool(on)
        self._debounce()

    def _debounce(self):
        """No refusionar en cada pixel que se arrastra: espera a que la mano
        se pare un momento."""
        timer = getattr(self, "_timer", None)
        if timer is None:
            timer = self._timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._request_preview)
        timer.start(160)

    def _on_group_changed(self, row):
        if 0 <= row < len(self.groups):
            self._request_shots(row)

    def _on_check_changed(self, _item):
        self._update_ok_button()

    def _checked_groups(self):
        if self.group_list is None:
            return list(self.groups)
        return [self.groups[i] for i in range(self.group_list.count())
                if self.group_list.item(i).checkState() == Qt.Checked]

    def _update_ok_button(self):
        chosen = self._checked_groups()
        if self.group_list is None:
            self.ok_button.setText("Fusionar")
        else:
            self.ok_button.setText(f"Fusionar {len(chosen)} tanda(s)")
        self.ok_button.setEnabled(bool(chosen))

    def accept(self):
        self.groups_to_merge = self._checked_groups()
        if not self.groups_to_merge:
            return
        super().accept()
