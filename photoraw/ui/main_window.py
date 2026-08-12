"""PhotoRAW - editor RAW simple y personalizable."""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (Qt, QObject, QRunnable, QThreadPool, QTimer,
                            Signal, QSize, QPointF, QRectF)
from PySide6.QtGui import QAction, QImage, QPixmap, QIcon, QKeySequence, QColor, QPainter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider, QListWidget,
    QListWidgetItem, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QFileDialog, QInputDialog, QMessageBox, QScrollArea, QSplitter,
    QProgressDialog, QAbstractItemView, QGroupBox, QSizePolicy, QComboBox,
    QFrame, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QGraphicsItem, QProgressBar, QStackedWidget,
    QCheckBox, QMenu, QDialog, QStyledItemDelegate, QStyleOptionViewItem,
    QStyle,
)
from PySide6.QtGui import QPen

import qtawesome as qta

from photoraw import (ai, diskcache, engine, face_parse, faces, generative,
                      hardware, hdr, heal, loader, masks_ai, presets, upscale)
from photoraw.edits import EditStore
from photoraw.ui.curve_widget import CurveWidget, HistogramWidget

APP_NAME = "PhotoRAW"


def icon(name, color="#d8d8d8"):
    """Icono Material Design en el tono claro del tema."""
    return qta.icon(name, color=color, color_active="#ffffff",
                    color_disabled="#5a5a5a")

# (clave, etiqueta, min, max, escala) agrupados en secciones con divisor
SLIDER_SECTIONS = [
    ("Balance de blancos", [
        ("temperature", "Temperatura", -100, 100, 1.0),
        ("tint",        "Matiz",       -100, 100, 1.0),
    ]),
    ("Tono", [
        ("exposure",    "Exposición",  -300, 300, 100.0, True),
        ("contrast",    "Contraste",   -100, 100, 1.0),
        ("highlights",  "Luces",       -100, 100, 1.0),
        ("shadows",     "Sombras",     -100, 100, 1.0),
        ("whites",      "Blancos",     -100, 100, 1.0),
        ("blacks",      "Negros",      -100, 100, 1.0),
    ]),
    ("Presencia", [
        ("clarity",           "Claridad",         -100, 100, 1.0),
        ("texture",           "Textura",          -100, 100, 1.0),
        ("dehaze",            "Borrar neblina",   -100, 100, 1.0),
        ("tone_map",          "Mapeo tonal",        0, 100, 1.0),
        ("adaptive_contrast", "Contraste adaptativo", 0, 100, 1.0),
    ]),
    (None, [
        ("saturation",  "Saturación",  -100, 100, 1.0),
        ("vibrance",    "Vitalidad",   -100, 100, 1.0),
    ]),
]

DETAIL_SECTIONS = [
    ("Enfoque", [
        ("sharp_amount",  "Cantidad",  0, 150, 1.0),
        ("sharp_radius",  "Radio",     5, 30, 10.0),
        ("sharp_detail",  "Detalle",   0, 100, 1.0),
        ("sharp_masking", "Máscara",   0, 100, 1.0),
    ]),
    ("Reducción de ruido", [
        ("nr_luminance", "Luminancia", 0, 100, 1.0),
        ("nr_color",     "Color",      0, 100, 1.0),
    ]),
    ("Reducción de ruido IA", [
        ("ai_denoise", "Intensidad", 0, 100, 1.0),
    ]),
]

FACE_SECTIONS = [
    ("Retoque de rostros IA", [
        ("ai_face", "Intensidad", 0, 100, 1.0),
    ]),
]

EFFECT_SECTIONS = [
    ("Grano de película", [
        ("grain_amount", "Cantidad", 0, 100, 1.0),
        ("grain_size",   "Tamaño",   0, 100, 1.0),
        ("grain_rough",  "Aspereza", 0, 100, 1.0),
    ]),
]

MASK_SLIDERS = [
    ("exposure",    "Exposición",  -300, 300, 100.0),
    ("contrast",    "Contraste",   -100, 100, 1.0),
    ("highlights",  "Luces",       -100, 100, 1.0),
    ("shadows",     "Sombras",     -100, 100, 1.0),
    ("temperature", "Temperatura", -100, 100, 1.0),
    ("tint",        "Matiz",       -100, 100, 1.0),
    ("saturation",  "Saturación",  -100, 100, 1.0),
    ("ai_denoise",  "Ruido IA",       0, 100, 1.0),
    ("ai_face",     "Rostros IA",     0, 100, 1.0),
]

MASK_TYPE_NAMES = {"linear": "Lineal", "radial": "Radial", "brush": "Pincel",
                   "subject": "Sujeto IA", "background": "Fondo IA"}

# Mascaras de retrato (face_parse): zonas que se ofrecen en el menu
FACE_PART_NAMES = [("skin", "Piel"), ("hair", "Cabello"), ("brows", "Cejas"),
                   ("eyes", "Ojos"), ("lips", "Labios"), ("teeth", "Dientes")]
FACE_PART_LABELS = dict(FACE_PART_NAMES)

PROFILE_OPTIONS = [
    ("Estándar",           "standard"),
    ("Vívido",             "vivid"),
    ("Retrato",            "portrait"),
    ("Paisaje",            "landscape"),
    ("Plano (para editar)", "flat"),
    ("Blanco y negro",     "bw"),
]

CAL_SECTIONS = [
    ("Sombras", [
        ("cal_shadow_tint", "Matiz", -100, 100, 1.0),
    ]),
    ("Primario rojo", [
        ("cal_red_hue", "Tono",       -100, 100, 1.0),
        ("cal_red_sat", "Saturación", -100, 100, 1.0),
    ]),
    ("Primario verde", [
        ("cal_green_hue", "Tono",       -100, 100, 1.0),
        ("cal_green_sat", "Saturación", -100, 100, 1.0),
    ]),
    ("Primario azul", [
        ("cal_blue_hue", "Tono",       -100, 100, 1.0),
        ("cal_blue_sat", "Saturación", -100, 100, 1.0),
    ]),
]

SLIDERS = [row for _title, rows in
           SLIDER_SECTIONS + DETAIL_SECTIONS + FACE_SECTIONS + EFFECT_SECTIONS
           + CAL_SECTIONS
           for row in rows]

# Bandas del mezclador HSL: (clave del motor, etiqueta, color de la muestra)
HSL_BAND_INFO = [
    ("red",     "Rojo",       "#d94f4f"),
    ("orange",  "Naranja",    "#dd8a3e"),
    ("yellow",  "Amarillo",   "#d8c04a"),
    ("green",   "Verde",      "#57b45a"),
    ("aqua",    "Aguamarina", "#4cbfc7"),
    ("blue",    "Azul",       "#5079d9"),
    ("purple",  "Púrpura",    "#9a5cd0"),
    ("magenta", "Magenta",    "#cf58a6"),
]

PC_SLIDERS = [
    ("pc_hue",   "Matiz",      -100, 100, 1.0),
    ("pc_sat",   "Saturación", -100, 100, 1.0),
    ("pc_lum",   "Luminancia", -100, 100, 1.0),
    ("pc_range", "Rango",         5, 100, 1.0),
]

CROP_SLIDERS = [
    ("straighten", "Enderezar", -450, 450, 10.0),
]

# (etiqueta, valor, para que sirve, explicacion larga del tooltip)
ASPECT_RATIOS = [
    ("Libre",    None,
     "recorta a ojo",
     "Sin proporción fija: el marco toma la forma que quieras."),
    ("Original", "original",
     "la forma de tu cámara",
     "Mantiene la proporción con la que disparaste la foto."),
    ("1 : 1",    1.0,
     "cuadrado · Instagram",
     "Cuadrado: post clásico de Instagram, avatares y portadas de disco."),
    ("5 : 4",    5 / 4,
     "copia 20×25 cm",
     "Horizontal poco alargado. Papel de 20×25 cm y marcos de pared "
     "clásicos; recorta menos que 3:2 al imprimir."),
    ("4 : 3",    4 / 3,
     "móvil · Micro 4/3 · iPad",
     "Horizontal compacto: cámara del móvil, sensores Micro 4/3, "
     "tablets y diapositivas de presentación."),
    ("3 : 2",    3 / 2,
     "réflex · copia 10×15 cm",
     "La proporción del negativo de 35 mm: réflex y mirrorless. Imprime "
     "sin recortar en 10×15, 20×30 y 30×45 cm."),
    ("16 : 9",   16 / 9,
     "pantalla · vídeo · web",
     "Panorámico de pantalla: TV, monitor, YouTube, portadas y cabeceras "
     "de página web."),
    ("4 : 5",    4 / 5,
     "vertical de Instagram",
     "Vertical del feed de Instagram: es el formato que más espacio ocupa "
     "en el móvil sin ser pantalla completa."),
    ("3 : 4",    3 / 4,
     "retrato vertical · 15×20 cm",
     "Vertical de móvil y de Micro 4/3. Papel de 15×20 cm."),
    ("2 : 3",    2 / 3,
     "vertical de réflex · 10×15 cm",
     "El 3:2 girado: retratos y verticales de réflex, copia de 10×15 cm "
     "en vertical."),
    ("9 : 16",   9 / 16,
     "Stories · Reels · TikTok",
     "Pantalla completa del móvil: historias de Instagram, Reels, "
     "TikTok y Shorts."),
]

PARAM_SLIDERS = [
    ("p_highlights", "Iluminaciones", -100, 100, 1.0),
    ("p_lights",     "Claros",        -100, 100, 1.0),
    ("p_darks",      "Oscuros",       -100, 100, 1.0),
    ("p_shadows",    "Sombras",       -100, 100, 1.0),
]

CURVE_CHANNELS = [
    ("curve_rgb", "RGB",   "#e8e8e8"),
    ("curve_r",   "Rojo",  "#e05555"),
    ("curve_g",   "Verde", "#55c060"),
    ("curve_b",   "Azul",  "#5588e0"),
]

DARK_STYLE = """
QMainWindow, QWidget { background: #232323; color: #d8d8d8; font-size: 13px; }
QToolBar { background: #1b1b1b; border: none; spacing: 6px; padding: 4px; }
QToolButton, QPushButton {
    background: #3a3a3a; color: #e6e6e6; border: 1px solid #4a4a4a;
    border-radius: 4px; padding: 5px 10px;
}
QToolButton:hover, QPushButton:hover { background: #4a4a4a; }
QPushButton:disabled { color: #777; background: #2c2c2c; }
QListWidget { background: #1b1b1b; border: none; }
QListWidget::item { color: #bbb; padding: 2px; }
QListWidget::item:selected { background: #3d6ea5; border-radius: 4px; color: white; }
QSlider::groove:horizontal { height: 4px; background: #4a4a4a; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #d8d8d8; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}
QSlider::handle:horizontal:hover { background: #ffffff; }
QGroupBox {
    border: 1px solid #3a3a3a; border-radius: 6px;
    margin-top: 12px; padding-top: 10px; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; }
QScrollArea { border: none; }
QStatusBar { background: #1b1b1b; color: #999; }
QSplitter::handle { background: #1b1b1b; }
"""


class NoWheelSlider(QSlider):
    """Deslizador de ajuste.

    - La rueda del raton no lo mueve: se deja pasar al scroll del panel.
    - Doble clic = vuelve a su valor de fabrica, como en Lightroom. Subes
      contraste a 23, ves que es demasiado, doble clic y a 0 sin tener que
      afinar el arrastre ni acordarte de cuanto habia.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_value = 0
        self.setToolTip("Doble clic para restablecer")

    def wheelEvent(self, event):
        event.ignore()

    def mouseDoubleClickEvent(self, event):
        self.setValue(self.default_value)   # valueChanged aplica el cambio
        event.accept()


class NoWheelCombo(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class AspectItemDelegate(QStyledItemDelegate):
    """Fila del desplegable: la proporción a la izquierda y para qué sirve,
    en gris, a la derecha."""

    PAD = 10
    GAP = 18

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        name, use = opt.text, index.data(Qt.UserRole + 1) or ""
        opt.text = ""
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)

        rect = opt.rect.adjusted(self.PAD, 0, -self.PAD, 0)
        fm = opt.fontMetrics
        painter.save()
        painter.setPen(QColor("#e8e8e8"))
        painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, name)
        if use:
            free = rect.width() - fm.horizontalAdvance(name) - self.GAP
            if free > 20:
                painter.setPen(QColor("#909090"))
                painter.drawText(rect, Qt.AlignRight | Qt.AlignVCenter,
                                 fm.elidedText(use, Qt.ElideRight, free))
        painter.restore()

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        fm = option.fontMetrics
        use = index.data(Qt.UserRole + 1) or ""
        width = (fm.horizontalAdvance(index.data(Qt.DisplayRole) or "")
                 + fm.horizontalAdvance(use) + self.GAP + 2 * self.PAD)
        return QSize(max(size.width(), width), max(size.height(), 26))


# matiz central (grados) de cada banda del mezclador HSL
BAND_HUES = dict(zip(engine.HSL_BANDS, engine.HSL_CENTERS))


def hue_color(h, s=0.85, v=0.85):
    """Color CSS a partir de un matiz en grados (con vuelta a la rueda)."""
    return QColor.fromHsvF((h % 360.0) / 360.0, s, v).name()


def set_slider_gradient(slider, colors):
    """Pinta la pista del deslizador con un degradado que orienta el ajuste;
    con None vuelve a la pista neutra."""
    if not colors:
        slider.setStyleSheet("")
        return
    n = max(len(colors) - 1, 1)
    stops = ", ".join(f"stop:{i / n:.4f} {c}" for i, c in enumerate(colors))
    slider.setStyleSheet(
        "QSlider::groove:horizontal { height: 6px; border-radius: 3px;"
        " background: qlineargradient(x1:0, y1:0, x2:1, y2:0, " + stops + "); }")


def np_to_qimage(arr):
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, w * 3, QImage.Format_RGB888).copy()


class Signals(QObject):
    thumb_ready = Signal(str, QImage)
    preview_ready = Signal(str, int, QImage, object)
    render_done = Signal()
    base_ready = Signal(str)
    ai_ready = Signal(str)
    ai_status = Signal(str)
    face_ready = Signal(str)
    heal_ready = Signal(str)
    erase_ready = Signal(str)
    mask_ai_ready = Signal(str)
    face_parse_ready = Signal(str)


class BusyChip(QWidget):
    """Pildora flotante sobre el visor con un aro girando y el estado del
    trabajo en curso (con porcentaje cuando se conoce), estilo Lightroom.

    A la derecha lleva una ✕ para detener el trabajo: es el unico sitio de
    la pildora que responde al raton, el resto de los clics siguen su camino
    hacia la foto."""

    cancelled = Signal()

    X_SIZE = 22   # lado del cuadro sensible de la ✕
    X_PAD = 8     # separacion con el borde derecho

    def __init__(self, parent):
        super().__init__(parent)
        self.setMouseTracking(True)   # para iluminar la ✕ al pasar por encima
        self.setToolTip("Detener el trabajo de IA (Esc)")
        self._angle = 0
        self._text = ""
        self._hover_x = False
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def _tick(self):
        self._angle = (self._angle + 10) % 360
        self._reposition()
        self.update()

    def show_text(self, text):
        self._text = text
        w = self.fontMetrics().horizontalAdvance(text) + 56 + self.X_SIZE + self.X_PAD
        self.setFixedSize(max(w, 160), 34)
        self._reposition()
        if not self.isVisible():
            self.show()
            self._timer.start()
        self.raise_()
        self.update()

    def hide_chip(self):
        self._timer.stop()
        self._hover_x = False
        self.unsetCursor()
        self.hide()

    def _reposition(self):
        p = self.parentWidget()
        if p is not None:
            self.move((p.width() - self.width()) // 2, 14)

    def _x_rect(self):
        return QRectF(self.width() - self.X_SIZE - self.X_PAD,
                      (34 - self.X_SIZE) / 2, self.X_SIZE, self.X_SIZE)

    def mouseMoveEvent(self, event):
        over = self._x_rect().contains(event.position())
        if over != self._hover_x:
            self._hover_x = over
            if over:
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.unsetCursor()
            self.update()
        event.ignore()

    def leaveEvent(self, _event):
        if self._hover_x:
            self._hover_x = False
            self.unsetCursor()
            self.update()

    def mousePressEvent(self, event):
        if self._x_rect().contains(event.position()):
            self.cancelled.emit()
            event.accept()
        else:
            event.ignore()   # clic en la pildora: que lo reciba la foto

    def paintEvent(self, _event):
        pt = QPainter(self)
        pt.setRenderHint(QPainter.Antialiasing)
        pt.setPen(Qt.NoPen)
        pt.setBrush(QColor(24, 24, 24, 215))
        pt.drawRoundedRect(self.rect(), 17, 17)
        pen = QPen(QColor(255, 255, 255, 235), 3)
        pen.setCapStyle(Qt.RoundCap)
        pt.setPen(pen)
        pt.drawArc(QRectF(12, 8, 18, 18), -self._angle * 16, 110 * 16)
        pt.setPen(QColor(235, 235, 235))
        pt.drawText(self.rect().adjusted(40, 0, -(self.X_SIZE + self.X_PAD), 0),
                    Qt.AlignVCenter | Qt.AlignLeft, self._text)
        # boton de parada: circulo tenue (mas marcado al pasar por encima)
        xr = self._x_rect()
        pt.setPen(Qt.NoPen)
        pt.setBrush(QColor(255, 255, 255, 46 if self._hover_x else 24))
        pt.drawEllipse(xr)
        pen = QPen(QColor(255, 255, 255, 255 if self._hover_x else 190), 2)
        pen.setCapStyle(Qt.RoundCap)
        pt.setPen(pen)
        c = xr.adjusted(6.5, 6.5, -6.5, -6.5)
        pt.drawLine(c.topLeft(), c.bottomRight())
        pt.drawLine(c.topRight(), c.bottomLeft())
        pt.end()


class ThumbJob(QRunnable):
    """Genera miniaturas de todas las fotos de la carpeta en segundo plano."""

    def __init__(self, files, signals, cancel_flag):
        super().__init__()
        self.files = files
        self.signals = signals
        self.cancel_flag = cancel_flag

    def run(self):
        for path in self.files:
            if self.cancel_flag["stop"]:
                return
            img = QImage()
            data = diskcache.thumb_jpeg(path)  # con cache en disco
            if data:
                img = QImage.fromData(data)
                if not img.isNull():
                    img = img.scaled(320, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.signals.thumb_ready.emit(str(path), img)


class DecodeJob(QRunnable):
    """Decodifica el RAW a una vista previa editable."""

    def __init__(self, path, window):
        super().__init__()
        self.path = path
        self.window = window

    def run(self):
        try:
            base = diskcache.load_preview(self.path)
        except Exception:
            base = None
        self.window.on_base_decoded(str(self.path), base)


class RenderJob(QRunnable):
    """Aplica los ajustes a la vista previa.

    Con `cache_tag` el revelado terminado se guarda en el cache de disco y se
    reutiliza: volver a una foto ya revelada pasa de segundos a un pestaneo.
    El tag lo pone la ventana y lleva TODO lo que cambia el resultado (los
    ajustes y que ingredientes de IA entran); la huella del `base` se calcula
    aqui, en el hilo de trabajo, porque son unas decimas sobre 37 MB.
    """

    def __init__(self, path, base, edits, gen, signals, denoised=None,
                 faced=None, ai_masks=None, cache_tag=None):
        super().__init__()
        self.path = path
        self.base = base
        self.edits = edits
        self.gen = gen
        self.signals = signals
        self.denoised = denoised
        self.faced = faced
        self.ai_masks = ai_masks
        self.cache_tag = cache_tag

    def run(self):
        try:
            key = None
            if self.cache_tag:
                key = "rend1-" + diskcache.result_key(self.base, self.cache_tag)
                hit = diskcache.load_result(self.path, key)
                if hit is not None:
                    out = np.ascontiguousarray(hit[0])
                    self.signals.preview_ready.emit(
                        self.path, self.gen, np_to_qimage(out), out)
                    return
            base = self.base
            amount = self.edits.get("ai_denoise", 0.0) / 100.0
            if amount > 0 and self.denoised is not None and self.denoised.shape == base.shape:
                base = base * (1.0 - amount) + self.denoised * amount
            # Rostros: se suma solo la diferencia, para no deshacer el denoise
            f_amount = self.edits.get("ai_face", 0.0) / 100.0
            if f_amount > 0 and self.faced is not None and self.faced.shape == base.shape:
                base = np.clip(base + (self.faced - self.base) * f_amount, 0.0, 1.0)
            out = engine.apply_edits(base, self.edits, ai_masks=self.ai_masks,
                                     denoised=self.denoised, faced=self.faced,
                                     is_raw=loader.is_raw(self.path))
            if key:
                diskcache.save_result(self.path, key, out)
            self.signals.preview_ready.emit(self.path, self.gen, np_to_qimage(out), out)
        finally:
            # avisa siempre (incluso si algo falla) para que la fila india
            # de renders no se quede atascada
            self.signals.render_done.emit()


class AIJob(QRunnable):
    """Trabajo de IA que se puede detener desde la interfaz.

    Al crearse comparte con la ventana un "testigo" de parada. Cuando el
    usuario pulsa Detener, la ventana lo marca y estrena uno nuevo: los
    trabajos en curso se enteran y abortan en su proximo punto de control
    (cada paso de difusion, cada mosaico, cada zona), y los que se lancen
    despues nacen con el testigo limpio.

    OJO: el corte solo puede ocurrir ENTRE pasos. Una llamada al modelo ya
    en marcha en la GPU no se puede interrumpir desde Python, asi que el
    trabajo puede tardar unos segundos mas en soltarla; la interfaz no
    espera a eso (ver MainWindow.cancel_ai)."""

    def __init__(self, path, base, window):
        super().__init__()
        self.path = path
        self.base = base
        self.window = window
        self.token = window.ai_cancel

    @property
    def cancelled(self):
        return self.token["stop"]

    def status(self, msg):
        self.window.signals.ai_status.emit(msg)

    def progress(self, label, scale=None):
        """Devuelve el progress_cb que pasar al modulo de IA: informa del
        porcentaje y, de paso, aborta el trabajo si se pidio parar."""
        def cb(p):
            if self.token["stop"]:
                raise ai.Cancelled()
            if scale is not None:
                p = scale(p)
            self.status(f"{label}… {min(int(p * 100), 99)} %")
        return cb


class MaskAIJob(AIJob):
    """Segmenta el sujeto de la foto para las mascaras IA."""

    def run(self):
        if self.cancelled:
            self.window.on_mask_ai_done(str(self.path), None)
            return
        key = "subj-" + diskcache.result_key(self.base)
        hit = diskcache.load_result(self.path, key)
        if hit is not None:
            self.window.on_mask_ai_done(str(self.path), hit[0])
            return
        try:
            self.status("Máscara IA: analizando la foto…")
            result = masks_ai.subject_mask(self.base)
            diskcache.save_result(self.path, key, result)
        except ai.Cancelled:
            result = None
            self.status("Máscara IA: detenida")
        except Exception as exc:
            result = None
            self.status(f"Máscara IA: error — {exc}")
        self.window.on_mask_ai_done(str(self.path), result)


class FaceParseJob(AIJob):
    """Divide las caras de la foto en zonas (piel, pelo, labios...) para
    las mascaras de retrato."""

    def run(self):
        if self.cancelled:
            self.window.on_face_parse_done(str(self.path), None)
            return
        key = "fpl-" + diskcache.result_key(self.base)
        hit = diskcache.load_result(self.path, key)
        if hit is not None:
            self.window.on_face_parse_done(str(self.path), hit[0])
            return
        try:
            self.status("Máscara de retrato: analizando las caras…")
            result = face_parse.parse_labels(self.base)
            diskcache.save_result(self.path, key, result)
        except ai.Cancelled:
            result = None
            self.status("Máscara de retrato: detenida")
        except Exception as exc:
            result = None
            self.status(f"Máscara de retrato: error — {exc}")
        self.window.on_face_parse_done(str(self.path), result)


class AIDenoiseJob(AIJob):
    """Pasa la vista previa por el modelo de IA en la GPU."""

    def run(self):
        if self.cancelled:
            self.window.on_ai_denoised(str(self.path), None)
            return
        key = "den-" + diskcache.result_key(self.base)
        hit = diskcache.load_result(self.path, key)
        if hit is not None:
            self.window.on_ai_denoised(str(self.path), hit[0])
            return
        try:
            self.status("IA: procesando…")
            result = ai.denoise(self.base, progress_cb=self.progress("Ruido IA"))
            diskcache.save_result(self.path, key, result)
        except ai.Cancelled:
            result = None
            self.status("Ruido IA: detenido")
        except Exception as exc:
            result = None
            self.status(f"IA: error — {exc}")
        self.window.on_ai_denoised(str(self.path), result)


class HealJob(AIJob):
    """Rellena con LaMa las zonas pintadas con el pincel corrector."""

    def __init__(self, path, base, strokes, window):
        super().__init__(path, base, window)
        self.strokes = strokes

    def run(self):
        if self.cancelled:
            self.window.on_healed(str(self.path), None, self.strokes)
            return
        # "heal2": desde que el corrector usa LaMa tambien en manchas
        # pequenas, los resultados del relleno clasico guardados no valen
        key = "heal2-" + diskcache.result_key(self.base, self.strokes)
        hit = diskcache.load_result(self.path, key)
        if hit is not None:
            self.window.on_healed(str(self.path), hit[0], self.strokes)
            return
        try:
            self.status("Corrector: borrando…")
            h, w = self.base.shape[:2]
            mask = heal.rasterize_strokes(self.strokes, h, w)
            result = heal.inpaint(self.base, mask,
                                  progress_cb=self.progress("Corrector"))
            # redondeado a 8 bits (= como se guarda): asi la huella de los
            # pasos que parten de la foto corregida no cambia entre sesiones
            result = (np.clip(result, 0.0, 1.0) * 255.0 + 0.5).astype(
                np.uint8).astype(np.float32) / 255.0
            diskcache.save_result(self.path, key, result)
        except ai.Cancelled:
            result = None
            self.status("Corrector: detenido")
            # los trazos no llegaron a aplicarse: que no queden guardados
            self.window.heal_undo_request = (str(self.path), self.strokes)
        except Exception as exc:
            result = None
            self.status(f"Corrector: error — {exc}")
        self.window.on_healed(str(self.path), result, self.strokes)


class EraseJob(AIJob):
    """Borrado generativo: reconstruye el fondo con difusion en las zonas
    marcadas. `ops` es la lista de operaciones a aplicar en orden.

    `drop_on_cancel` distingue los dos casos: un borrado recien pedido que
    se detiene debe desaparecer del historial (no llego a hacerse), mientras
    que al re-aplicar borrados ya guardados solo se suspende el calculo, la
    edicion sigue ahi."""

    def __init__(self, path, base, ops, window, drop_on_cancel=False):
        super().__init__(path, base, window)
        self.ops = ops
        self.drop_on_cancel = drop_on_cancel

    def run(self):
        if self.cancelled:
            self._give_up()
            return
        # "erase2": desde que las zonas grandes se difunden a 768 px, los
        # rellenos a 512 guardados quedan invalidados a proposito
        key = "erase2-" + diskcache.result_key(
            self.base, json.dumps(self.ops, sort_keys=True, default=str))
        hit = diskcache.load_result(self.path, key)
        result = self.base
        union = None
        try:
            n = len(self.ops)
            for j, op in enumerate(self.ops):
                h, w = result.shape[:2]
                wmap = generative.decode_map(op["map"], h, w)
                if wmap is None:
                    continue
                union = wmap if union is None else np.maximum(union, wmap)
                if hit is not None:
                    continue  # resultado ya en cache; solo falta la union
                result = generative.erase(
                    result, wmap, seed=int(op.get("seed", 0)),
                    progress_cb=self.progress("Borrado generativo",
                                              lambda p, j=j, n=n: (j + p) / n))
            if hit is not None:
                result = hit[0]
            elif result is not self.base:
                # redondeado a 8 bits (= como se guarda): huella estable
                # para los pasos que parten de la foto borrada
                result = (np.clip(result, 0.0, 1.0) * 255.0 + 0.5).astype(
                    np.uint8).astype(np.float32) / 255.0
                diskcache.save_result(self.path, key, result)
        except ai.Cancelled:
            self._give_up()
            return
        except Exception as exc:
            result = None
            self.window.signals.ai_status.emit(
                f"Borrado generativo: error — {exc}")
        self.window.on_erased(str(self.path), result, union)

    def _give_up(self):
        """Abandona el borrado: suelta los ~2 GB de VRAM del modelo de
        difusion y, si el borrado era nuevo, pide a la ventana que lo quite
        del historial (lo hara en el hilo de la interfaz)."""
        generative.release_sessions()
        self.status("Borrado generativo: detenido")
        if self.drop_on_cancel:
            self.window.erase_undo_request = (str(self.path), self.ops)
        self.window.on_erased(str(self.path), None, None)


class AIFaceJob(AIJob):
    """Detecta y restaura los rostros de la vista previa en la GPU."""

    def run(self):
        if self.cancelled:
            self.window.on_ai_faces_done(str(self.path), None, 0)
            return
        key = f"fac-{faces.current_model()}-" + diskcache.result_key(self.base)
        hit = diskcache.load_result(self.path, key)
        if hit is not None:
            arr, n = hit
            # meta=0 es el "aqui no hay caras" guardado (ver mas abajo)
            self.window.on_ai_faces_done(str(self.path),
                                         arr if n else None, n)
            return
        try:
            self.status("IA rostros: procesando…")
            result, n = faces.enhance_faces(self.base)
            if n == 0:
                result = None
                self.status("IA rostros: no se detectaron caras")
                # Guardar tambien que NO hay caras, con un array de mentira y
                # meta=0. Si no, una foto sin gente que tenga el ajuste de
                # rostros guardado vuelve a buscarlas en cada apertura y no
                # apunta nunca el resultado.
                diskcache.save_result(self.path, key,
                                      np.zeros((1, 1, 3), np.uint8), meta=0)
            else:
                diskcache.save_result(self.path, key, result, meta=n)
        except ai.Cancelled:
            result, n = None, 0
            self.status("IA rostros: detenido")
        except Exception as exc:
            result, n = None, 0
            self.status(f"IA rostros: error — {exc}")
        self.window.on_ai_faces_done(str(self.path), result, n)


class AIModelDownloadJob(QRunnable):
    """Descarga el modelo de IA si no esta en el disco."""

    def __init__(self, window):
        super().__init__()
        self.window = window

    def run(self):
        try:
            self.window.signals.ai_status.emit("IA: descargando modelo…")
            ai.download_model(lambda p: self.window.signals.ai_status.emit(
                f"IA: descargando modelo… {p * 100:.0f} %"))
            self.window.signals.ai_status.emit("IA: modelo listo")
        except Exception as exc:
            self.window.signals.ai_status.emit(f"IA: fallo la descarga — {exc}")
        self.window.on_ai_model_downloaded()


class PhotoView(QGraphicsView):
    """Visor con zoom (rueda del ratón), arrastre para desplazarse,
    doble clic para alternar ajuste/100 % y modo pincel corrector."""

    zoomChanged = Signal(float)
    strokesChanged = Signal()
    colorPicked = Signal(QColor)
    cropChanged = Signal(list)
    maskDrawn = Signal(str, dict)
    maskDragging = Signal(str, dict)  # mientras dibujas: mascara de mentira
    maskEdited = Signal(bool)  # True mientras se arrastra, False al soltar

    _HANDLES = ["tl", "t", "tr", "l", "r", "bl", "b", "br"]
    _CURSORS = {"tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
                "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
                "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
                "move": Qt.SizeAllCursor}

    def __init__(self):
        super().__init__()
        self._picker_mode = False
        self._mask_draw = None        # "linear" | "radial" mientras se dibuja
        self._mask_drag = None        # punto inicial del arrastre
        self._mask_edit = None        # mascara seleccionada con tiradores
        self._mask_edit_drag = None   # tirador que se esta arrastrando
        self._crop_mode = False
        self._crop_aspect = None      # None = libre, float = ancho/alto
        self._crop_rect = None        # QRectF en coords de la foto
        self._crop_drag = None        # (tipo, punto inicial, rect original)
        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._item)
        self._overlay_item = QGraphicsPixmapItem()
        self._overlay_item.setZValue(1)
        self._scene.addItem(self._overlay_item)
        # tinte rojo que muestra la mascara seleccionada
        self._mask_overlay_item = QGraphicsPixmapItem()
        self._mask_overlay_item.setZValue(1.5)
        self._scene.addItem(self._mask_overlay_item)
        # tiradores para editar la mascara (mover / estirar)
        guide_pen = QPen(QColor(255, 255, 255, 200))
        guide_pen.setCosmetic(True)
        guide_pen.setStyle(Qt.DashLine)
        self._mask_line = self._scene.addLine(0, 0, 0, 0, guide_pen)
        self._mask_line.setZValue(2.5)
        self._mask_line.hide()
        self._mask_ellipse = self._scene.addEllipse(QRectF(), guide_pen)
        self._mask_ellipse.setZValue(2.5)
        self._mask_ellipse.hide()
        mh_pen = QPen(QColor(20, 20, 20))
        mh_pen.setWidthF(2.0)
        self._mask_handle_items = {}
        for name in ("p0", "p1", "c", "l", "r", "t", "b"):
            it = self._scene.addEllipse(QRectF(-7, -7, 14, 14), mh_pen,
                                        QColor(255, 255, 255))
            it.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            it.setZValue(2.6)
            it.hide()
            self._mask_handle_items[name] = it
        self._overlay = None          # QImage ARGB con los trazos pintados
        self._brush_mode = False
        self._brush_radius = 30.0     # radio en pixeles de la foto
        self._strokes = []            # trazos pendientes, normalizados
        self._current = None
        # circulo que muestra el tamano del pincel bajo el cursor
        cursor_pen = QPen(QColor(255, 255, 255, 230))
        cursor_pen.setWidthF(1.5)
        cursor_pen.setCosmetic(True)  # grosor constante en pantalla
        self._cursor_item = QGraphicsEllipseItem()
        self._cursor_item.setPen(cursor_pen)
        self._cursor_item.setZValue(2)
        self._cursor_item.hide()
        self._scene.addItem(self._cursor_item)
        self.viewport().setMouseTracking(True)
        # elementos del marco de recorte
        shade = QColor(0, 0, 0, 150)
        self._shade_items = []
        for _ in range(4):
            it = self._scene.addRect(QRectF(), QPen(Qt.NoPen), shade)
            it.setZValue(3)
            it.hide()
            self._shade_items.append(it)
        border_pen = QPen(QColor(255, 255, 255, 230))
        border_pen.setCosmetic(True)
        border_pen.setWidthF(1.5)
        self._crop_border = self._scene.addRect(QRectF(), border_pen)
        self._crop_border.setZValue(4)
        self._crop_border.hide()
        grid_pen = QPen(QColor(255, 255, 255, 70))
        grid_pen.setCosmetic(True)
        self._grid_items = []
        for _ in range(4):
            it = self._scene.addLine(0, 0, 0, 0, grid_pen)
            it.setZValue(4)
            it.hide()
            self._grid_items.append(it)
        handle_pen = QPen(QColor(255, 255, 255))
        self._handle_items = []
        for _ in range(8):
            it = self._scene.addRect(QRectF(-4, -4, 8, 8), handle_pen,
                                     QColor(255, 255, 255))
            it.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            it.setZValue(5)
            it.hide()
            self._handle_items.append(it)
        self._text = self._scene.addText("Abre una carpeta con fotos para empezar")
        self._text.setDefaultTextColor(QColor("#888888"))
        self.setScene(self._scene)
        self.setBackgroundBrush(QColor("#1e1e1e"))
        self.setFrameStyle(0)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        # con elementos ItemIgnoresTransformations (tiradores del marco), el
        # modo de actualizacion parcial deja residuos de fotogramas viejos
        # (bandas con la foto "a medio editar"); repintado completo siempre
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._fit = True
        self._has_photo = False

    def set_photo(self, pixmap, fit=False):
        self._text.setVisible(False)
        self._item.setVisible(True)
        old_w = self._item.pixmap().width() if self._has_photo else 0
        size_changed = self._item.pixmap().size() != pixmap.size()
        # con zoom activo y resolucion distinta (borrador <-> calidad
        # completa) hay que conservar el punto mirado: capturarlo ANTES de
        # tocar el pixmap, porque cambiar la escena ya desplaza el scroll
        keep_view = (not fit and not self._fit and old_w
                     and pixmap.width() and pixmap.width() != old_w)
        if keep_view:
            center = self.mapToScene(self.viewport().rect().center())
        self._item.setPixmap(pixmap)
        self._scene.setSceneRect(self._item.boundingRect())
        self._has_photo = True
        if size_changed:
            self._reset_overlay()
        if fit:
            self._fit = True
        if self._fit:
            self._refit()
        elif keep_view:
            # misma escala visual y mismo centro (anclado al centro del
            # visor, nunca al raton, que puede estar sobre los paneles)
            ratio = pixmap.width() / old_w
            old_anchor = self.transformationAnchor()
            self.setTransformationAnchor(QGraphicsView.NoAnchor)
            self.scale(1.0 / ratio, 1.0 / ratio)
            self.centerOn(center.x() * ratio, center.y() * ratio)
            self.setTransformationAnchor(old_anchor)
            self.zoomChanged.emit(self.transform().m11())

    # ---- pincel corrector ----

    def _reset_overlay(self):
        self._strokes = []
        self._current = None
        size = self._item.pixmap().size()
        if size.isEmpty():
            self._overlay = None
            self._overlay_item.setPixmap(QPixmap())
        else:
            self._overlay = QImage(size, QImage.Format_ARGB32_Premultiplied)
            self._overlay.fill(Qt.transparent)
            self._overlay_item.setPixmap(QPixmap.fromImage(self._overlay))

    def set_brush_mode(self, on):
        self._brush_mode = on
        self.setDragMode(QGraphicsView.NoDrag if on else QGraphicsView.ScrollHandDrag)
        self.setCursor(Qt.BlankCursor if on else Qt.ArrowCursor)
        if not on:
            self._cursor_item.hide()

    # ---- cuentagotas del color de punto ----

    def set_picker_mode(self, on):
        self._picker_mode = on
        if on:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        elif not self._brush_mode:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.ArrowCursor)

    # ---- dibujo de mascaras (lineal / radial) ----

    def set_mask_draw_mode(self, kind):
        self._mask_draw = kind
        self._mask_drag = None
        if kind:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        elif not (self._brush_mode or self._picker_mode or self._crop_mode):
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.ArrowCursor)

    def set_mask_overlay(self, image, scale=1.0):
        """El velo se calcula a resolucion reducida y la GPU lo escala."""
        if image is None or image.isNull():
            self._mask_overlay_item.setPixmap(QPixmap())
        else:
            self._mask_overlay_item.setPixmap(QPixmap.fromImage(image))
            self._mask_overlay_item.setScale(scale)

    def set_mask_handles(self, mask):
        """Muestra tiradores para editar la mascara (solo lineal/radial).
        Con tiradores activos se desactiva el paneo con la mano, para que
        el arrastre siempre agarre la mascara y no mueva la foto."""
        new = mask if (mask and mask.get("type")
                       in ("linear", "radial")) else None
        if new is not self._mask_edit:
            # solo se suelta el arrastre si la mascara cambio de verdad;
            # el refresco del velo NO debe cortar el gesto a mitad de camino
            self._mask_edit_drag = None
        self._mask_edit = new
        if self._mask_edit is not None:
            self.setDragMode(QGraphicsView.NoDrag)
            if self._mask_edit_drag is None:
                self.setCursor(Qt.ArrowCursor)
        elif not (self._brush_mode or self._picker_mode or self._crop_mode
                  or self._mask_draw):
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.ArrowCursor)
        self._update_mask_handles()

    def _update_mask_handles(self):
        m = self._mask_edit
        for it in self._mask_handle_items.values():
            it.hide()
        self._mask_line.hide()
        self._mask_ellipse.hide()
        if m is None or not self._has_photo:
            return
        w = self._item.pixmap().width()
        h = self._item.pixmap().height()
        if not w or not h:
            return
        if m["type"] == "linear":
            x0, y0 = m["x0"] * w, m["y0"] * h
            x1, y1 = m["x1"] * w, m["y1"] * h
            self._mask_line.setLine(x0, y0, x1, y1)
            self._mask_line.show()
            self._mask_handle_items["p0"].setPos(x0, y0)
            self._mask_handle_items["p1"].setPos(x1, y1)
            self._mask_handle_items["p0"].show()
            self._mask_handle_items["p1"].show()
        else:
            cx, cy = m["cx"] * w, m["cy"] * h
            rx, ry = m["rx"] * w, m["ry"] * h
            self._mask_ellipse.setRect(QRectF(cx - rx, cy - ry, 2 * rx, 2 * ry))
            self._mask_ellipse.show()
            for name, (px, py) in (("c", (cx, cy)), ("l", (cx - rx, cy)),
                                   ("r", (cx + rx, cy)), ("t", (cx, cy - ry)),
                                   ("b", (cx, cy + ry))):
                self._mask_handle_items[name].setPos(px, py)
                self._mask_handle_items[name].show()

    def _mask_handle_hit(self, view_pos):
        if self._mask_edit is None:
            return None
        for name, it in self._mask_handle_items.items():
            if not it.isVisible():
                continue
            vp = self.mapFromScene(it.pos())
            if (vp - view_pos).manhattanLength() <= 16:
                return name
        return None

    def _drag_mask_handle(self, pos):
        m = self._mask_edit
        w = self._item.pixmap().width()
        h = self._item.pixmap().height()
        if m is None or not w or not h:
            return
        cl = lambda v: min(max(v, 0.0), 1.0)
        k = self._mask_edit_drag
        if m["type"] == "linear":
            if k == "p0":
                m["x0"], m["y0"] = cl(pos.x() / w), cl(pos.y() / h)
            elif k == "p1":
                m["x1"], m["y1"] = cl(pos.x() / w), cl(pos.y() / h)
        else:
            if k == "c":
                m["cx"], m["cy"] = cl(pos.x() / w), cl(pos.y() / h)
            elif k in ("l", "r"):
                m["rx"] = max(abs(pos.x() / w - m["cx"]), 0.02)
            elif k in ("t", "b"):
                m["ry"] = max(abs(pos.y() / h - m["cy"]), 0.02)
        self._update_mask_handles()
        self.maskEdited.emit(True)

    def _mask_drag_params(self, scene_pos):
        """Parametros de la mascara que estas dibujando ahora mismo, con el
        raton todavia apretado. None si la foto aun no tiene tamano."""
        p0 = self._mask_drag
        w = self._item.pixmap().width()
        h = self._item.pixmap().height()
        if p0 is None or not w or not h:
            return None
        cl = lambda v: min(max(v, 0.0), 1.0)
        if self._mask_draw == "linear":
            return {"x0": cl(p0.x() / w), "y0": cl(p0.y() / h),
                    "x1": cl(scene_pos.x() / w), "y1": cl(scene_pos.y() / h)}
        # radial: del centro hacia afuera
        return {"cx": cl(p0.x() / w), "cy": cl(p0.y() / h),
                "rx": max(abs(scene_pos.x() - p0.x()) / w, 0.02),
                "ry": max(abs(scene_pos.y() - p0.y()) / h, 0.02),
                "feather": 50.0}

    def _preview_mask_drag(self, scene_pos):
        """Ensena la mascara MIENTRAS la arrastras: la linea (o la elipse) con
        sus tiradores y, por encima, el velo azul de la zona afectada. Antes
        no se veia absolutamente nada hasta soltar el boton, asi que dibujabas
        a ciegas y solo al final descubrias donde habia quedado."""
        params = self._mask_drag_params(scene_pos)
        if params is None:
            return
        self._mask_edit = {"type": self._mask_draw, **params}
        self._update_mask_handles()
        self.maskDragging.emit(self._mask_draw, self._mask_edit)

    def _finish_mask_drag(self, scene_pos):
        params = self._mask_drag_params(scene_pos)
        kind = self._mask_draw
        self._mask_drag = None
        # se suelta la mascara de mentira del arrastre: la de verdad llega
        # enseguida por maskDrawn -> _create_mask -> set_mask_handles
        self._mask_edit = None
        if params is None:
            return
        self.maskDrawn.emit(kind, params)

    # ---- marco de recorte ----

    def _crop_items(self):
        return (self._shade_items + [self._crop_border]
                + self._grid_items + self._handle_items)

    def set_crop_mode(self, on):
        self._crop_mode = on
        self._crop_drag = None
        if on:
            self.setDragMode(QGraphicsView.NoDrag)
            if self._crop_rect is None and self._has_photo:
                self._crop_rect = QRectF(self._item.boundingRect())
            self._update_crop_items()
        else:
            self.setCursor(Qt.ArrowCursor)
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            for it in self._crop_items():
                it.hide()
        self.viewport().update()

    def set_crop_rect(self, norm):
        """norm: [x0, y0, x1, y1] en 0..1, o None para toda la foto."""
        if not self._has_photo:
            return
        b = self._item.boundingRect()
        if norm:
            self._crop_rect = QRectF(
                norm[0] * b.width(), norm[1] * b.height(),
                (norm[2] - norm[0]) * b.width(), (norm[3] - norm[1]) * b.height())
        else:
            self._crop_rect = QRectF(b)
        if self._crop_mode:
            self._update_crop_items()

    def set_crop_aspect(self, aspect, reshape=True):
        """aspect: None (libre), "original" o ancho/alto como numero."""
        if aspect == "original":
            b = self._item.boundingRect()
            aspect = b.width() / max(b.height(), 1.0)
        self._crop_aspect = aspect
        if (aspect and reshape and self._crop_mode
                and self._has_photo and self._crop_rect is not None):
            self._reshape_to_aspect()
            self._update_crop_items()
            self._emit_crop()

    def _reshape_to_aspect(self):
        """Reencaja el marco actual a la proporcion elegida, centrado."""
        b = QRectF(self._item.boundingRect())
        r = self._crop_rect
        a = self._crop_aspect
        w, h = r.width(), r.height()
        if w / max(h, 1e-6) > a:
            w = h * a
        else:
            h = w / a
        out = QRectF(0, 0, w, h)
        out.moveCenter(r.center())
        out.translate(
            max(0.0, b.left() - out.left()) + min(0.0, b.right() - out.right()),
            max(0.0, b.top() - out.top()) + min(0.0, b.bottom() - out.bottom()))
        self._crop_rect = out.intersected(b)

    def _handle_points(self):
        r = self._crop_rect
        return [(r.left(), r.top()), (r.center().x(), r.top()),
                (r.right(), r.top()), (r.left(), r.center().y()),
                (r.right(), r.center().y()), (r.left(), r.bottom()),
                (r.center().x(), r.bottom()), (r.right(), r.bottom())]

    def _update_crop_items(self):
        if not (self._crop_mode and self._has_photo and self._crop_rect):
            return
        b = QRectF(self._item.boundingRect())
        r = self._crop_rect
        self._shade_items[0].setRect(QRectF(b.left(), b.top(),
                                            b.width(), r.top() - b.top()))
        self._shade_items[1].setRect(QRectF(b.left(), r.bottom(),
                                            b.width(), b.bottom() - r.bottom()))
        self._shade_items[2].setRect(QRectF(b.left(), r.top(),
                                            r.left() - b.left(), r.height()))
        self._shade_items[3].setRect(QRectF(r.right(), r.top(),
                                            b.right() - r.right(), r.height()))
        self._crop_border.setRect(r)
        for i in (0, 1):
            x = r.left() + r.width() * (i + 1) / 3.0
            y = r.top() + r.height() * (i + 1) / 3.0
            self._grid_items[i].setLine(x, r.top(), x, r.bottom())
            self._grid_items[i + 2].setLine(r.left(), y, r.right(), y)
        for it, (px, py) in zip(self._handle_items, self._handle_points()):
            it.setPos(px, py)
        for it in self._crop_items():
            it.show()

    def _crop_hit(self, view_pos):
        """Que parte del marco hay bajo el cursor (esquinas primero)."""
        if self._crop_rect is None:
            return None
        points = dict(zip(self._HANDLES, self._handle_points()))
        for name in ("tl", "tr", "bl", "br", "t", "b", "l", "r"):
            px, py = points[name]
            vp = self.mapFromScene(QPointF(px, py))
            if (vp - view_pos).manhattanLength() <= 16:
                return name
        if self._crop_rect.contains(self.mapToScene(view_pos)):
            return "move"
        return None

    def _drag_crop(self, scene_pos):
        kind, start, orig = self._crop_drag
        b = QRectF(self._item.boundingRect())
        min_sz = max(b.width(), b.height()) * 0.03
        if kind == "move":
            d = scene_pos - start
            dx = min(max(d.x(), b.left() - orig.left()), b.right() - orig.right())
            dy = min(max(d.y(), b.top() - orig.top()), b.bottom() - orig.bottom())
            self._crop_rect = orig.translated(dx, dy)
            self._update_crop_items()
            return
        x = min(max(scene_pos.x(), b.left()), b.right())
        y = min(max(scene_pos.y(), b.top()), b.bottom())
        left, top = orig.left(), orig.top()
        right, bottom = orig.right(), orig.bottom()
        if kind in ("tl", "l", "bl"):
            left = min(x, right - min_sz)
        if kind in ("tr", "r", "br"):
            right = max(x, left + min_sz)
        if kind in ("tl", "t", "tr"):
            top = min(y, bottom - min_sz)
        if kind in ("bl", "b", "br"):
            bottom = max(y, top + min_sz)
        r = QRectF(QPointF(left, top), QPointF(right, bottom))
        a = self._crop_aspect
        if a:
            if kind in ("l", "r"):
                cy = orig.center().y()
                half = min(r.width() / a / 2.0, cy - b.top(), b.bottom() - cy)
                w = half * 2.0 * a
                x0 = orig.left() if kind == "r" else orig.right() - w
                r = QRectF(x0, cy - half, w, half * 2.0)
            elif kind in ("t", "b"):
                cx = orig.center().x()
                half = min(r.height() * a / 2.0, cx - b.left(), b.right() - cx)
                h = half * 2.0 / a
                y0 = orig.top() if kind == "b" else orig.bottom() - h
                r = QRectF(cx - half, y0, half * 2.0, h)
            else:
                ax = orig.right() if kind in ("tl", "bl") else orig.left()
                ay = orig.bottom() if kind in ("tl", "tr") else orig.top()
                sx = -1.0 if kind in ("tl", "bl") else 1.0
                sy = -1.0 if kind in ("tl", "tr") else 1.0
                dx = max((x - ax) * sx, min_sz)
                dy = max((y - ay) * sy, min_sz)
                w = max(dx, dy * a)
                max_w = (b.right() - ax) if sx > 0 else (ax - b.left())
                max_h = (b.bottom() - ay) if sy > 0 else (ay - b.top())
                w = min(w, max_w, max_h * a)
                h = w / a
                r = QRectF(min(ax, ax + sx * w), min(ay, ay + sy * h), w, h)
        self._crop_rect = r.intersected(b)
        self._update_crop_items()

    def _emit_crop(self):
        b = self._item.boundingRect()
        r = self._crop_rect
        if b.width() <= 0 or b.height() <= 0 or r is None:
            return
        norm = [min(max(r.left() / b.width(), 0.0), 1.0),
                min(max(r.top() / b.height(), 0.0), 1.0),
                min(max(r.right() / b.width(), 0.0), 1.0),
                min(max(r.bottom() / b.height(), 0.0), 1.0)]
        # marco completo = sin recorte
        if norm[0] < 1e-3 and norm[1] < 1e-3 and norm[2] > 0.999 and norm[3] > 0.999:
            norm = []
        self.cropChanged.emit(norm)

    def set_brush_radius(self, radius):
        self._brush_radius = float(radius)
        if self._cursor_item.isVisible():
            c = self._cursor_item.rect().center()
            r = self._brush_radius
            self._cursor_item.setRect(c.x() - r, c.y() - r, 2 * r, 2 * r)

    def _move_cursor_circle(self, view_pos):
        if not (self._brush_mode and self._has_photo):
            self._cursor_item.hide()
            return
        pos = self.mapToScene(view_pos)
        r = self._brush_radius
        self._cursor_item.setRect(pos.x() - r, pos.y() - r, 2 * r, 2 * r)
        self._cursor_item.show()

    def leaveEvent(self, event):
        self._cursor_item.hide()
        super().leaveEvent(event)

    def has_strokes(self):
        return bool(self._strokes)

    def peek_strokes(self):
        return list(self._strokes)

    def clear_strokes(self):
        self._reset_overlay()
        self.strokesChanged.emit()

    def take_strokes(self):
        strokes = self._strokes
        self._reset_overlay()
        return strokes

    def _draw_segment(self, p1, p2):
        if self._overlay is None:
            return
        painter = QPainter(self._overlay)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = painter.pen()
        pen.setColor(QColor(255, 70, 70, 130))
        pen.setWidthF(self._brush_radius * 2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        if p1 == p2:
            painter.drawPoint(p1)
        else:
            painter.drawLine(p1, p2)
        painter.end()
        self._overlay_item.setPixmap(QPixmap.fromImage(self._overlay))

    def clear_photo(self, text=""):
        self._has_photo = False
        self._item.setVisible(False)
        self.resetTransform()
        self._text.setPlainText(text)
        self._text.setVisible(bool(text))
        if text:
            self._scene.setSceneRect(self._text.boundingRect())

    def _refit(self):
        if self._has_photo:
            self.fitInView(self._item, Qt.KeepAspectRatio)
            self.zoomChanged.emit(self.transform().m11())

    def wheelEvent(self, event):
        if not self._has_photo:
            return
        delta = event.angleDelta().y()
        if not delta:
            return
        factor = 1.25 if delta > 0 else 0.8
        current = self.transform().m11()
        target = min(max(current * factor, 0.02), 8.0)
        self._fit = False
        self.scale(target / current, target / current)
        self.zoomChanged.emit(target)

    def mousePressEvent(self, event):
        if self._mask_draw and self._has_photo and event.button() == Qt.LeftButton:
            self._mask_drag = self.mapToScene(event.position().toPoint())
            return
        if (self._mask_edit is not None and self._has_photo
                and not self._brush_mode and event.button() == Qt.LeftButton):
            hit = self._mask_handle_hit(event.position().toPoint())
            if hit:
                self._mask_edit_drag = hit
                return
            # sin tirador debajo: sigue el paneo normal
        if self._crop_mode and self._has_photo and event.button() == Qt.LeftButton:
            hit = self._crop_hit(event.position().toPoint())
            if hit:
                self._crop_drag = (hit, self.mapToScene(event.position().toPoint()),
                                   QRectF(self._crop_rect))
            return
        if self._picker_mode and self._has_photo and event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            if self._item.boundingRect().contains(pos):
                img = self._item.pixmap().toImage()
                x = min(int(pos.x()), img.width() - 1)
                y = min(int(pos.y()), img.height() - 1)
                self.colorPicked.emit(img.pixelColor(x, y))
            return
        if self._brush_mode and self._has_photo and event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            if self._item.boundingRect().contains(pos):
                w = self._item.pixmap().width()
                h = self._item.pixmap().height()
                norm_r = self._brush_radius / max(w, h)
                self._current = [norm_r, [[pos.x() / w, pos.y() / h]]]
                self._draw_segment(pos, pos)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._mask_draw and self._mask_drag is not None:
            self._preview_mask_drag(self.mapToScene(event.position().toPoint()))
            return
        if self._mask_edit_drag is not None:
            self._drag_mask_handle(self.mapToScene(event.position().toPoint()))
            return
        if (self._mask_edit is not None and self._has_photo
                and not self._brush_mode and self._mask_draw is None):
            hit = self._mask_handle_hit(event.position().toPoint())
            self.setCursor(Qt.SizeAllCursor if hit else Qt.ArrowCursor)
        if self._crop_mode and self._has_photo:
            if self._crop_drag is not None:
                self._drag_crop(self.mapToScene(event.position().toPoint()))
            else:
                hit = self._crop_hit(event.position().toPoint())
                self.setCursor(self._CURSORS.get(hit, Qt.ArrowCursor))
            return
        self._move_cursor_circle(event.position().toPoint())
        if self._brush_mode and self._current is not None:
            pos = self.mapToScene(event.position().toPoint())
            w = self._item.pixmap().width()
            h = self._item.pixmap().height()
            last = self._current[1][-1]
            self._draw_segment(QPointF(last[0] * w, last[1] * h), pos)
            self._current[1].append([pos.x() / w, pos.y() / h])
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._mask_draw and self._mask_drag is not None:
            self._finish_mask_drag(self.mapToScene(event.position().toPoint()))
            return
        if self._mask_edit_drag is not None:
            self._mask_edit_drag = None
            self.maskEdited.emit(False)
            return
        if self._crop_mode and self._crop_drag is not None:
            self._crop_drag = None
            self._emit_crop()
            return
        if self._brush_mode and self._current is not None:
            self._strokes.append(self._current)
            self._current = None
            self.strokesChanged.emit()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._brush_mode or self._crop_mode or not self._has_photo:
            return
        if self._fit:
            # a 100 % centrado donde esta el cursor
            pos = self.mapToScene(event.position().toPoint())
            self._fit = False
            self.resetTransform()
            self.centerOn(pos)
            self.zoomChanged.emit(1.0)
        else:
            self._fit = True
            self._refit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit:
            self._refit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 850)

        self.pool = QThreadPool.globalInstance()
        # carril propio para decodificar y renderizar: la foto en pantalla
        # nunca hace cola detras de los trabajos IA largos
        self.fast_pool = QThreadPool(self)
        self.fast_pool.setMaxThreadCount(2)
        self.signals = Signals()
        self.signals.thumb_ready.connect(self.on_thumb_ready)
        self.signals.preview_ready.connect(self.on_preview_ready)
        self.signals.render_done.connect(self._on_render_done)
        self.signals.base_ready.connect(self.on_base_ready)
        self.signals.ai_ready.connect(self.on_ai_ready)
        self.signals.ai_status.connect(self._on_ai_status)
        self.signals.face_ready.connect(self.on_face_ready)
        self.signals.heal_ready.connect(self.on_heal_ready)
        self.signals.erase_ready.connect(self.on_erase_ready)
        self.signals.mask_ai_ready.connect(self.on_mask_ai_ready)
        self.signals.face_parse_ready.connect(self.on_face_parse_ready)

        self.folder = None
        self.store = None
        self.current_path = None
        self.current_edits = engine.full_edits()
        self.copied_edits = None
        self.show_mask = False            # vista de la mascara de enfoque (Alt)
        self.crop_mode = False            # pestana Recorte activa
        self.fit_next = True              # ajustar zoom al cambiar de foto
        # path -> miniatura del revelado, para las fotos que tienen edicion.
        # La de la camara vive en thumb_pixmaps y no se pisa nunca
        self.edited_thumbs = {}
        self.base_cache = OrderedDict()   # path -> float32 array
        # path -> revelado ya terminado (huella, ingredientes, QImage, array):
        # volver a una foto que no ha cambiado se pinta sin calcular nada
        self.render_cache = OrderedDict()
        self._cacheable_fp = None         # ingredientes del render en marcha
        self._cacheable_gen = None        # su `gen`, si es guardable
        self.ai_cache = OrderedDict()     # path -> vista previa sin ruido (IA)
        self.ai_running = set()
        self.ai_pending_path = None       # foto esperando a que baje el modelo
        self.face_cache = OrderedDict()   # path -> vista previa con rostros IA
        self.face_running = set()
        self.no_faces = set()             # fotos donde ya se busco y no hay caras
        self.healed = set()               # fotos con trazos ya aplicados a su base
        self.heal_running = set()
        self.heal_undo = {}   # path -> [(n trazos, n borrados IA, base previa f16)]
        self.erased = set()               # fotos con borrado generativo aplicado
        self.erase_running = set()
        self.decoding = set()
        self.gen = 0
        self.thumb_cancel = {"stop": False}
        # testigo de "Detener": los trabajos de IA se quedan con el que haya
        # al crearse y lo consultan en cada punto de control (ver AIJob)
        self.ai_cancel = {"stop": False}
        self.ai_paused = False    # tras Detener, la IA guardada no se relanza
                                  # sola hasta que vuelvas a pedirla
        # trabajos detenidos a medias que hay que borrar del historial
        # (los rellena el hilo de trabajo, los atiende el de la interfaz)
        self.erase_undo_request = None
        self.heal_undo_request = None
        # borrados generativos con el hilo aun vivo (incluye los que se
        # detuvieron y todavia no han soltado la GPU): dos a la vez no
        # caben en 8 GB de VRAM
        self.erase_alive = 0
        self.thumb_pixmaps = {}           # path -> miniatura sin insignia
        self.mask_ai_cache = {}           # path -> {"subject": mapa 0..1}
        self.mask_ai_running = set()
        self._pending_ai_mask = None      # tipo de mascara esperando su mapa
        self.face_parse_running = set()
        self._pending_face_part = None    # zona de retrato esperando su mapa
        self.mask_brush_on = False
        self.mask_refine_sign = 0         # +1 añade, -1 resta, 0 apagado
        self._ai_from_mask = set()        # (foto, clave): IA pedida por una
                                          # mascara — no encender el global
        self._mask_ai_prompted = set()    # descargas ya preguntadas (sesion)

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(80)
        self.render_timer.timeout.connect(self.request_render)

        # fila india de renders: nunca mas de uno a la vez; si llegan mas
        # mientras uno trabaja, solo se guarda el ultimo (los intermedios
        # ya estan obsoletos y solo quemarian procesador)
        self._render_busy = False
        self._render_next = None

        # mientras se arrastra un ajuste se renderiza un borrador reducido;
        # este temporizador dispara el render a calidad completa al soltar
        self.small_cache = {}   # (path, tipo) -> (referencia, copia reducida)
        self.final_timer = QTimer(self)
        self.final_timer.setSingleShot(True)
        self.final_timer.setInterval(450)
        self.final_timer.timeout.connect(lambda: self.request_render(final=True))

        # los modelos de IA se quedan cargados en la GPU para no recargarlos
        # a cada uso; si pasas un buen rato sin IA, se devuelven solos (en
        # una tarjeta de 8 GB compartida con otras apps se nota mucho)
        self.idle_free_timer = QTimer(self)
        self.idle_free_timer.setSingleShot(True)
        self.idle_free_timer.setInterval(self.IDLE_FREE_MIN * 60_000)
        self.idle_free_timer.timeout.connect(self.free_gpu)

        self._build_ui()
        self._build_toolbar()
        self.refresh_presets()
        self._aplicar_perfil()

    # ---------- perfil segun la tarjeta grafica ----------

    def _aplicar_perfil(self):
        """Enciende o apaga las herramientas IA segun el equipo.

        En un PC sin CUDA, las IAs pesadas tardarian de minutos a horas por
        CPU. Dejarlas pulsables seria una trampa: parecen disponibles y te
        cuelgan el programa. Se apagan y se explica por que en el globo de
        ayuda. El revelado entero (exposicion, curvas, HSL, recorte,
        corrector, mascaras de degradado...) va por CPU y funciona igual."""
        info = hardware.detectar()
        if info["perfil"] == hardware.COMPLETO:
            self.statusBar().showMessage(hardware.resumen())
            return
        porque = ("\n\nDESACTIVADO: necesita una tarjeta NVIDIA con CUDA.\n"
                  f"En este equipo: {info['motivo']}.\n"
                  "Por CPU tardaría de minutos a horas por foto.")
        for widget in (self.ai_btn, self.face_btn, self.face_model_combo,
                       *self._solo_gpu):
            widget.setEnabled(False)
            widget.setToolTip(widget.toolTip() + porque)
        self.statusBar().showMessage(hardware.resumen())

    # ---------- interfaz ----------

    def _build_ui(self):
        self.film = QListWidget()
        self.film.setViewMode(QListWidget.IconMode)
        self.film.setIconSize(QSize(150, 150))
        self.film.setGridSize(QSize(170, 185))
        self.film.setResizeMode(QListWidget.Adjust)
        self.film.setMovement(QListWidget.Static)
        self.film.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.film.setWordWrap(True)
        self.film.currentItemChanged.connect(self.on_photo_selected)
        self.film.setContextMenuPolicy(Qt.CustomContextMenu)
        self.film.customContextMenuRequested.connect(self._film_context_menu)

        self.preview = PhotoView()
        self.status_chip = BusyChip(self.preview)
        self.status_chip.cancelled.connect(self.cancel_ai)
        self._last_status = ""
        self.preview.zoomChanged.connect(
            lambda s: self.statusBar().showMessage(f"Zoom: {s * 100:.0f} %"))
        self.preview.strokesChanged.connect(self.on_strokes_changed)
        self.preview.colorPicked.connect(self.on_color_picked)
        self.preview.cropChanged.connect(self.on_crop_changed)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 6, 10, 6)

        self.sliders = {}
        self.value_labels = {}
        self._solo_gpu = []   # controles que se apagan sin CUDA
        self._auto_exp_btn = None
        self._auto_exp_active = False

        adjust_box = self._build_section_box("Ajustes", SLIDER_SECTIONS)
        # perfil: la interpretacion base del RAW, encima de todo (como en LR)
        profile_row = QHBoxLayout()
        profile_label = QLabel("Perfil")
        profile_label.setMinimumWidth(92)
        self.profile_combo = NoWheelCombo()
        for label, _key in PROFILE_OPTIONS:
            self.profile_combo.addItem(label)
        self.profile_combo.setToolTip(
            "Punto de partida del revelado, antes de tus ajustes:\n"
            "Vívido satura, Retrato cuida la piel, Paisaje realza verdes y\n"
            "cielos, Plano deja la foto lavada para editarla a tu gusto.")
        self.profile_combo.currentIndexChanged.connect(self.on_profile_changed)
        profile_row.addWidget(profile_label)
        profile_row.addWidget(self.profile_combo, 1)
        adjust_box.layout().insertLayout(0, profile_row)
        # pistas coloreadas del balance de blancos, como en Lightroom
        set_slider_gradient(self.sliders["temperature"],
                            ["#3d6fd0", "#93a0b0", "#e8b23a"])
        set_slider_gradient(self.sliders["tint"],
                            ["#4fb054", "#a8a2a8", "#c855c0"])
        # Tooltips de los algoritmos profesionales
        self.sliders["tone_map"].setToolTip(
            "Mapeo tonal adaptativo (Reinhard):\n"
            "Comprime el rango dinámico del RAW de forma inteligente,\n"
            "preservando contraste local. Como Lightroom/Capture One.\n"
            "0 = desactivado, 50-70 = recomendado para RAW oscuros.")
        self.sliders["adaptive_contrast"].setToolTip(
            "Contraste adaptativo local (CLAHE):\n"
            "Mejora el contraste por zonas sin lavar la imagen.\n"
            "Revela detalle en sombras e iluminaciones.\n"
            "0 = desactivado, 30-50 = efecto natural.")
        reset_btn = QPushButton(icon("mdi6.restore"), "Restablecer ajustes")
        reset_btn.clicked.connect(self.reset_edits)
        adjust_box.layout().addWidget(reset_btn)

        detail_box = self._build_section_box("Detalle", DETAIL_SECTIONS)
        self.ai_btn = QPushButton(icon("mdi6.auto-fix"), "Procesar foto con IA")
        self.ai_btn.setToolTip(
            "Analiza la foto con el modelo SCUNet en tu GPU y elimina el ruido.\n"
            "Después regula la mezcla con el deslizador Intensidad.")
        self.ai_btn.clicked.connect(self.run_ai_denoise)
        detail_box.layout().addWidget(self.ai_btn)

        # Alt + arrastrar el deslizador Mascara muestra la mascara en B/N
        mask_slider = self.sliders["sharp_masking"]
        mask_slider.setToolTip("Mantén Alt mientras arrastras para ver la máscara:\n"
                               "blanco = se enfoca, negro = protegido")
        mask_slider.sliderReleased.connect(self._end_mask_view)

        face_box = self._build_section_box("Retoque IA", FACE_SECTIONS)
        self.face_model_combo = NoWheelCombo()
        self.face_model_combo.addItem("CodeFormer — fiel (gafas, oclusiones)")
        self.face_model_combo.addItem("GFPGAN — embellece más (puede alucinar)")
        self.face_model_combo.currentIndexChanged.connect(self.on_face_model_changed)
        face_box.layout().addWidget(self.face_model_combo)
        faces.set_model("CodeFormer")
        self.face_btn = QPushButton(icon("mdi6.face-recognition"),
                                    "Retocar rostros con IA")
        self.face_btn.setToolTip(
            "Detecta las caras y las restaura con GFPGAN en tu GPU:\n"
            "piel, ojos y detalles. Regula la mezcla con Intensidad.")
        self.face_btn.clicked.connect(self.run_ai_faces)
        face_box.layout().addWidget(self.face_btn)

        effects_box = self._build_section_box("Efectos", EFFECT_SECTIONS)
        color_box = self._build_color_box()

        cal_box = self._build_section_box("Calibración", CAL_SECTIONS)
        cal_reset = QPushButton(icon("mdi6.restore"), "Restablecer calibración")
        cal_reset.clicked.connect(self.reset_calibration)
        cal_box.layout().addWidget(cal_reset)
        # pistas coloreadas: matiz de sombras verde-magenta y cada primario
        # con su rotacion de tono y saturacion (como Lightroom)
        set_slider_gradient(self.sliders["cal_shadow_tint"],
                            ["#4fb054", "#a8a2a8", "#c855c0"])
        for base_hue, key in ((0, "red"), (120, "green"), (240, "blue")):
            set_slider_gradient(
                self.sliders[f"cal_{key}_hue"],
                [hue_color(base_hue - 30), hue_color(base_hue),
                 hue_color(base_hue + 30)])
            set_slider_gradient(
                self.sliders[f"cal_{key}_sat"],
                [hue_color(base_hue, s=0.04, v=0.60),
                 hue_color(base_hue, s=1.0)])

        # --- histograma ---
        hist_box = QGroupBox("Histograma")
        hist_layout = QVBoxLayout(hist_box)
        hist_layout.setContentsMargins(4, 4, 4, 4)
        self.histogram = HistogramWidget()
        hist_layout.addWidget(self.histogram)
        right_layout.addWidget(hist_box)

        # --- curva de tonos ---
        curve_box = QGroupBox("Curva de tonos")
        cv_layout = QVBoxLayout(curve_box)

        self.channel_combo = NoWheelCombo()
        for _key, label, _color in CURVE_CHANNELS:
            self.channel_combo.addItem(label)
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        cv_layout.addWidget(self.channel_combo)

        self.curve_widget = CurveWidget()
        self.curve_widget.curveChanged.connect(self.on_curve_changed)
        cv_layout.addWidget(self.curve_widget)

        curve_reset = QPushButton(icon("mdi6.restore"), "Restablecer curva del canal")
        curve_reset.clicked.connect(self.reset_current_curve)
        cv_layout.addWidget(curve_reset)

        param_grid = QGridLayout()
        param_grid.setVerticalSpacing(8)
        self._add_slider_rows(param_grid, PARAM_SLIDERS, 0)
        cv_layout.addLayout(param_grid)

        preset_box = QGroupBox("Preajustes")
        pv = QVBoxLayout(preset_box)
        self.preset_list = QListWidget()
        self.preset_list.setMaximumHeight(150)
        self.preset_list.itemDoubleClicked.connect(lambda _: self.apply_preset())
        pv.addWidget(self.preset_list)
        row1 = QHBoxLayout()
        b_save = QPushButton("Guardar")
        b_save.clicked.connect(self.save_preset)
        b_apply = QPushButton("Aplicar")
        b_apply.clicked.connect(self.apply_preset)
        b_del = QPushButton("Eliminar")
        b_del.clicked.connect(self.delete_preset)
        row1.addWidget(b_save)
        row1.addWidget(b_apply)
        row1.addWidget(b_del)
        pv.addLayout(row1)

        right_layout.addWidget(adjust_box)
        right_layout.addWidget(curve_box)
        right_layout.addWidget(color_box)
        right_layout.addWidget(detail_box)
        right_layout.addWidget(face_box)
        right_layout.addWidget(effects_box)
        right_layout.addWidget(cal_box)
        right_layout.addWidget(preset_box)
        right_layout.addStretch()

        right_scroll = QScrollArea()
        right_scroll.setWidget(right)
        right_scroll.setWidgetResizable(True)

        # panel lateral con pestanas: Revelar / Recorte
        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(10, 0, 10, 0)
        tab_row.setSpacing(6)
        self.tab_buttons = {}
        for key, label, ic in (("revelar", "Revelar", "mdi6.tune-variant"),
                               ("recorte", "Recorte", "mdi6.crop"),
                               ("mascaras", "Máscaras", "mdi6.circle-half-full")):
            btn = QPushButton(icon(ic), label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c=False, k=key: self.switch_panel(k))
            tab_row.addWidget(btn)
            self.tab_buttons[key] = btn
        self.tab_buttons["revelar"].setChecked(True)

        self.panel_stack = QStackedWidget()
        self.panel_stack.addWidget(right_scroll)
        self.panel_stack.addWidget(self._build_crop_panel())
        self.panel_stack.addWidget(self._build_mask_panel())
        self.preview.maskDrawn.connect(self.on_mask_drawn)
        self.preview.maskDragging.connect(self.on_mask_dragging)
        self.preview.maskEdited.connect(self.on_mask_edited)
        # el velo rojo se recalcula con un pequeno retraso durante arrastres
        self.overlay_timer = QTimer(self)
        self.overlay_timer.setSingleShot(True)
        self.overlay_timer.setInterval(60)
        self.overlay_timer.timeout.connect(self._update_mask_overlay)
        self._overlay_suspended = False
        # guardar a disco se difiere durante los arrastres de mascara
        self.mask_save_timer = QTimer(self)
        self.mask_save_timer.setSingleShot(True)
        self.mask_save_timer.setInterval(400)
        self.mask_save_timer.timeout.connect(self._save_masks)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 6, 0, 0)
        side_layout.setSpacing(6)
        side_layout.addLayout(tab_row)
        side_layout.addWidget(self.panel_stack)
        side.setMinimumWidth(340)
        side.setMaximumWidth(430)

        self._sync_sliders()

        film_container = QWidget()
        film_layout = QVBoxLayout(film_container)
        film_layout.setContentsMargins(0, 0, 0, 4)
        film_layout.setSpacing(4)
        film_layout.addWidget(self.film)
        self.delete_btn = QPushButton(icon("mdi6.delete-outline"), "Eliminar foto")
        self.delete_btn.setToolTip(
            "Elimina la foto seleccionada del disco o quítala del proyecto")
        self.delete_btn.clicked.connect(self.delete_photos)
        film_layout.addWidget(self.delete_btn)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(film_container)
        splitter.addWidget(self.preview)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([200, 840, 380])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Listo")
        # barra animada que indica trabajo de IA/corrector en curso
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setFixedWidth(120)
        self.busy_bar.setTextVisible(False)
        self.statusBar().addPermanentWidget(self.busy_bar)
        self.busy_bar.hide()

    def _build_crop_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 6, 10, 6)

        box = QGroupBox("Recorte y transformación")
        v = QVBoxLayout(box)
        v.setSpacing(8)

        ar_row = QHBoxLayout()
        ar_label = QLabel("Proporción")
        ar_label.setMinimumWidth(92)
        self.aspect_combo = NoWheelCombo()
        for label, _val, use, tip in ASPECT_RATIOS:
            self.aspect_combo.addItem(label)
            i = self.aspect_combo.count() - 1
            self.aspect_combo.setItemData(i, use, Qt.UserRole + 1)
            self.aspect_combo.setItemData(i, tip, Qt.ToolTipRole)
        self.aspect_combo.setItemDelegate(AspectItemDelegate(self.aspect_combo))
        # el desplegable se ensancha para que quepa el "para qué sirve"
        self.aspect_combo.view().setMinimumWidth(
            min(460, self.aspect_combo.view().sizeHintForColumn(0) + 24))
        self.aspect_combo.currentIndexChanged.connect(self.on_aspect_changed)
        ar_row.addWidget(ar_label)
        ar_row.addWidget(self.aspect_combo, 1)
        v.addLayout(ar_row)

        self.aspect_hint = QLabel()
        self.aspect_hint.setStyleSheet("color: #8a8a8a;")
        self.aspect_hint.setWordWrap(True)
        v.addWidget(self.aspect_hint)
        self._update_aspect_hint(self.aspect_combo.currentIndex())

        grid = QGridLayout()
        grid.setVerticalSpacing(8)
        self._add_slider_rows(grid, CROP_SLIDERS, 0)
        v.addLayout(grid)

        btn_row = QHBoxLayout()
        for ic, tip, fn in (
                ("mdi6.rotate-left", "Girar 90° a la izquierda",
                 lambda: self.rotate90(False)),
                ("mdi6.rotate-right", "Girar 90° a la derecha",
                 lambda: self.rotate90(True)),
                ("mdi6.flip-horizontal", "Voltear horizontal (espejo)",
                 lambda: self.flip_photo(True)),
                ("mdi6.flip-vertical", "Voltear vertical",
                 lambda: self.flip_photo(False))):
            b = QPushButton(icon(ic), "")
            b.setToolTip(tip)
            b.clicked.connect(fn)
            btn_row.addWidget(b)
        v.addLayout(btn_row)

        crop_reset = QPushButton(icon("mdi6.restore"), "Restablecer recorte")
        crop_reset.clicked.connect(self.reset_crop)
        v.addWidget(crop_reset)

        hint = QLabel("Arrastra las esquinas o los bordes del marco sobre la "
                      "foto; arrastra desde el centro para moverlo. El recorte "
                      "se aplica al volver a la pestaña Revelar.")
        hint.setStyleSheet("color: #8a8a8a;")
        hint.setWordWrap(True)
        v.addWidget(hint)

        layout.addWidget(box)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        return scroll

    def _build_mask_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 6, 10, 6)

        box = QGroupBox("Máscaras")
        v = QVBoxLayout(box)
        v.setSpacing(8)

        new_label = QLabel("AÑADIR MÁSCARA NUEVA")
        new_label.setStyleSheet("color: #8a8a8a; font-size: 11px; font-weight: bold;")
        v.addWidget(new_label)

        row1 = QHBoxLayout()
        b_lin = QPushButton(icon("mdi6.gradient-horizontal"), "Lineal")
        b_lin.setToolTip("Degradado lineal: arrastra sobre la foto desde donde\n"
                         "empieza el efecto hacia donde termina")
        b_lin.clicked.connect(lambda: self.start_mask_draw("linear"))
        b_rad = QPushButton(icon("mdi6.circle-outline"), "Radial")
        b_rad.setToolTip("Degradado radial: arrastra del centro hacia afuera")
        b_rad.clicked.connect(lambda: self.start_mask_draw("radial"))
        self.mask_brush_btn = QPushButton(icon("mdi6.brush"), "Pincel")
        self.mask_brush_btn.setCheckable(True)
        self.mask_brush_btn.setToolTip("Pinta a mano la zona a ajustar")
        self.mask_brush_btn.toggled.connect(self.toggle_mask_brush)
        row1.addWidget(b_lin)
        row1.addWidget(b_rad)
        row1.addWidget(self.mask_brush_btn)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        b_subj = QPushButton(icon("mdi6.account-outline"), "Sujeto (IA)")
        b_subj.setToolTip("La IA detecta a las personas u objeto principal")
        b_subj.clicked.connect(lambda: self.add_ai_mask("subject"))
        b_back = QPushButton(icon("mdi6.image-outline"), "Fondo (IA)")
        b_back.setToolTip("Todo lo que NO es el sujeto principal")
        b_back.clicked.connect(lambda: self.add_ai_mask("background"))
        b_face = QPushButton(icon("mdi6.face-recognition"), "Retrato (IA)")
        b_face.setToolTip("La IA divide las caras en zonas: piel, cabello,\n"
                          "cejas, ojos, labios o dientes")
        face_menu = QMenu(b_face)
        for part, label in FACE_PART_NAMES:
            face_menu.addAction(label,
                                lambda p=part: self.add_face_mask(p))
        b_face.setMenu(face_menu)
        row2.addWidget(b_subj)
        row2.addWidget(b_back)
        row2.addWidget(b_face)
        v.addLayout(row2)

        size_row = QHBoxLayout()
        size_label = QLabel("Tamaño del pincel")
        size_label.setMinimumWidth(92)
        mask_brush_size = NoWheelSlider(Qt.Horizontal)
        mask_brush_size.setRange(5, 120)
        mask_brush_size.setValue(30)
        mask_brush_size.default_value = 30
        mask_brush_size.valueChanged.connect(
            lambda val: self.preview.set_brush_radius(val))
        self.mask_brush_erase = QPushButton(icon("mdi6.eraser"), "Quitar")
        self.mask_brush_erase.setCheckable(True)
        self.mask_brush_erase.setToolTip(
            "Activado: el pincel BORRA de la máscara en vez de añadir")
        size_row.addWidget(size_label)
        size_row.addWidget(mask_brush_size)
        size_row.addWidget(self.mask_brush_erase)
        v.addLayout(size_row)

        self.mask_list = QListWidget()
        self.mask_list.setMaximumHeight(110)
        self.mask_list.currentRowChanged.connect(self.on_mask_selected)
        v.addWidget(self.mask_list)

        refine_row = QHBoxLayout()
        refine_label = QLabel("Refinar a mano")
        refine_label.setMinimumWidth(92)
        self.mask_refine_add = QPushButton(icon("mdi6.plus"), "Añadir")
        self.mask_refine_add.setCheckable(True)
        self.mask_refine_add.setToolTip(
            "Pinta sobre la foto para AÑADIR zonas a la máscara seleccionada")
        self.mask_refine_add.toggled.connect(
            lambda on: self.toggle_mask_refine(1, on))
        self.mask_refine_sub = QPushButton(icon("mdi6.minus"), "Restar")
        self.mask_refine_sub.setCheckable(True)
        self.mask_refine_sub.setToolTip(
            "Pinta sobre la foto para QUITAR zonas de la máscara seleccionada")
        self.mask_refine_sub.toggled.connect(
            lambda on: self.toggle_mask_refine(-1, on))
        refine_row.addWidget(refine_label)
        refine_row.addWidget(self.mask_refine_add)
        refine_row.addWidget(self.mask_refine_sub)
        v.addLayout(refine_row)

        opts_row = QHBoxLayout()
        self.mask_invert = QCheckBox("Invertir")
        self.mask_invert.toggled.connect(self.on_mask_invert)
        self.mask_show_overlay = QCheckBox("Ver máscara en rojo")
        self.mask_show_overlay.setChecked(True)
        self.mask_show_overlay.toggled.connect(
            lambda _on: self._update_mask_overlay())
        opts_row.addWidget(self.mask_invert)
        opts_row.addWidget(self.mask_show_overlay)
        v.addLayout(opts_row)

        feather_row = QHBoxLayout()
        feather_label = QLabel("Desvanecido")
        feather_label.setMinimumWidth(92)
        self.mask_feather = NoWheelSlider(Qt.Horizontal)
        self.mask_feather.setRange(2, 100)
        self.mask_feather.setValue(50)
        self.mask_feather.default_value = 50
        self.mask_feather.setToolTip(
            "Suavidad del borde del degradado radial")
        self.mask_feather.valueChanged.connect(self.on_mask_feather)
        feather_row.addWidget(feather_label)
        feather_row.addWidget(self.mask_feather)
        v.addLayout(feather_row)

        grid = QGridLayout()
        grid.setVerticalSpacing(8)
        self.mask_sliders = {}
        self.mask_value_labels = {}
        for row, (key, label, lo, hi, scale) in enumerate(MASK_SLIDERS):
            lab = QLabel(label)
            lab.setMinimumWidth(92)
            slider = NoWheelSlider(Qt.Horizontal)
            slider.setRange(lo, hi)
            val = QLabel("0")
            val.setFixedWidth(42)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            slider.valueChanged.connect(
                lambda value, k=key, s=scale, wdg=val:
                self.on_mask_slider(k, value, s, wdg))
            # mientras se arrastra un ajuste, el velo rojo se aparta solo
            slider.sliderPressed.connect(self._suspend_mask_overlay)
            slider.sliderReleased.connect(self._resume_mask_overlay)
            grid.addWidget(lab, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(val, row, 2)
            self.mask_sliders[key] = slider
            self.mask_value_labels[key] = val
        v.addLayout(grid)
        set_slider_gradient(self.mask_sliders["temperature"],
                            ["#3d6fd0", "#93a0b0", "#e8b23a"])
        set_slider_gradient(self.mask_sliders["tint"],
                            ["#4fb054", "#a8a2a8", "#c855c0"])
        self.mask_sliders["ai_denoise"].setToolTip(
            "Reducción de ruido IA solo en la zona de la máscara.\n"
            "La primera vez en cada foto la IA tarda unos segundos.")
        self.mask_sliders["ai_face"].setToolTip(
            "Retoque de rostros IA solo en la zona de la máscara.\n"
            "La primera vez en cada foto la IA tarda unos segundos.")

        b_erase = QPushButton(icon("mdi6.creation"), "Borrar con IA (rellenar fondo)")
        b_erase.setToolTip(
            "Hace desaparecer lo que cubre la máscara seleccionada y\n"
            "reconstruye el fondo con IA generativa (difusión).\n"
            "Ideal para quitar personas u objetos grandes.")
        b_erase.clicked.connect(self.run_generative_erase)
        v.addWidget(b_erase)
        # el generativo es lo unico impensable sin GPU: en CPU son horas
        self._solo_gpu.append(b_erase)

        b_del = QPushButton(icon("mdi6.delete-outline"), "Eliminar máscara")
        b_del.clicked.connect(self.delete_mask)
        v.addWidget(b_del)

        hint = QLabel("La zona en rojo es donde actúa la máscara seleccionada. "
                      "Los ajustes de arriba solo afectan a esa zona.")
        hint.setStyleSheet("color: #8a8a8a;")
        hint.setWordWrap(True)
        v.addWidget(hint)

        layout.addWidget(box)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        return scroll

    def _build_color_box(self):
        box = QGroupBox("Mezclador de color")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        # fila de muestras: elige la banda de color a ajustar
        band_row = QHBoxLayout()
        band_row.setSpacing(4)
        self.band_buttons = {}
        for key, label, color in HSL_BAND_INFO:
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setFixedSize(26, 20)
            btn.setToolTip(label)
            btn.setStyleSheet(
                f"QPushButton {{ background: {color}; border: 1px solid #222;"
                f" border-radius: 3px; padding: 0; }}"
                f"QPushButton:checked {{ border: 2px solid #ffffff; }}")
            btn.clicked.connect(lambda _c=False, k=key: self.on_band_selected(k))
            band_row.addWidget(btn)
            self.band_buttons[key] = btn
        band_row.addStretch()
        layout.addLayout(band_row)

        self.current_band = "red"
        self.band_buttons["red"].setChecked(True)

        grid = QGridLayout()
        grid.setVerticalSpacing(8)
        self.hsl_sliders = {}
        self.hsl_value_labels = {}
        for row, (comp, label) in enumerate(
                [("h", "Matiz"), ("s", "Saturación"), ("l", "Luminancia")]):
            lab = QLabel(label)
            lab.setMinimumWidth(92)
            slider = NoWheelSlider(Qt.Horizontal)
            slider.setRange(-100, 100)
            val = QLabel("0")
            val.setFixedWidth(42)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            slider.valueChanged.connect(lambda v, c=comp: self.on_hsl_slider(c, v))
            grid.addWidget(lab, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(val, row, 2)
            self.hsl_sliders[comp] = slider
            self.hsl_value_labels[comp] = val
        layout.addLayout(grid)
        self._update_hsl_gradients()

        mix_reset = QPushButton(icon("mdi6.restore"), "Restablecer mezclador")
        mix_reset.clicked.connect(self.reset_hsl)
        layout.addWidget(mix_reset)

        # --- color de punto ---
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #3a3a3a; background: #3a3a3a; max-height: 1px;")
        layout.addWidget(line)
        title = QLabel("COLOR DE PUNTO")
        title.setStyleSheet("color: #8a8a8a; font-size: 11px; font-weight: bold;")
        layout.addWidget(title)

        pick_row = QHBoxLayout()
        self.pick_btn = QPushButton(icon("mdi6.eyedropper-variant"), "Tomar muestra")
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip(
            "Actívalo y haz clic en la foto sobre el color que quieras cambiar.\n"
            "Después mueve Matiz / Saturación / Luminancia: solo afecta a ese color.\n"
            "Rango controla cuántos tonos parecidos se incluyen.")
        self.pick_btn.toggled.connect(self.toggle_picker)
        pick_row.addWidget(self.pick_btn)
        self.pc_swatch = QLabel()
        self.pc_swatch.setFixedSize(34, 24)
        pick_row.addWidget(self.pc_swatch)
        pick_row.addStretch()
        layout.addLayout(pick_row)

        pc_grid = QGridLayout()
        pc_grid.setVerticalSpacing(8)
        self._add_slider_rows(pc_grid, PC_SLIDERS, 0)
        layout.addLayout(pc_grid)

        pc_reset = QPushButton(icon("mdi6.restore"), "Restablecer color de punto")
        pc_reset.clicked.connect(self.reset_point_color)
        layout.addWidget(pc_reset)

        self._update_pc_swatch()
        return box

    def _build_section_box(self, box_title, sections):
        box = QGroupBox(box_title)
        layout = QVBoxLayout(box)
        layout.setSpacing(6)
        for i, (title, rows) in enumerate(sections):
            if i:
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setStyleSheet("color: #3a3a3a; background: #3a3a3a; max-height: 1px;")
                layout.addWidget(line)
            if title:
                sec = QLabel(title.upper())
                sec.setStyleSheet("color: #8a8a8a; font-size: 11px; font-weight: bold;")
                layout.addWidget(sec)
            grid = QGridLayout()
            grid.setVerticalSpacing(8)
            self._add_slider_rows(grid, rows, 0)
            layout.addLayout(grid)
        return box

    def _add_slider_rows(self, grid, spec, start_row):
        for row, item in enumerate(spec, start=start_row):
            if len(item) == 6:
                key, label, lo, hi, scale, auto = item
            else:
                key, label, lo, hi, scale = item
                auto = False
            lab = QLabel(label)
            lab.setMinimumWidth(92)  # alinea los deslizadores entre secciones
            slider = NoWheelSlider(Qt.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(0)
            # el doble clic devuelve el ajuste a fabrica, que no siempre es
            # cero (enfoque: radio 1,0 y detalle 25)
            slider.default_value = int(round(
                float(engine.DEFAULT_EDITS.get(key, 0) or 0) * scale))
            val = QLabel("0")
            val.setFixedWidth(42)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            slider.valueChanged.connect(
                lambda v, k=key, s=scale, w=val: self.on_slider(k, v, s, w))
            grid.addWidget(lab, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(val, row, 2)
            if auto:
                btn = QPushButton("A")
                btn.setFixedSize(26, 22)
                btn.setToolTip(
                    "Exposición automática (click = activar)\n"
                    "Calcula el EV óptimo analizando la imagen.\n"
                    "Se desactiva si mueves el slider manualmente.")
                self._update_auto_btn_style(btn, False)
                btn.clicked.connect(lambda _, k=key: self._auto_exposure(k))
                grid.addWidget(btn, row, 3)
                self._auto_exp_btn = btn
            self.sliders[key] = slider
            self.value_labels[key] = val

    def _build_toolbar(self):
        tb = self.addToolBar("Principal")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        a_open = QAction(icon("mdi6.folder-open-outline"), "Abrir carpeta", self)
        a_open.setShortcut(QKeySequence("Ctrl+O"))
        a_open.triggered.connect(self.open_folder)
        tb.addAction(a_open)

        a_export = QAction(icon("mdi6.export-variant"), "Exportar JPEG", self)
        a_export.setShortcut(QKeySequence("Ctrl+E"))
        a_export.triggered.connect(self.export_selected)
        tb.addAction(a_export)

        a_hdr = QAction(icon("mdi6.hdr"), "Fusionar HDR", self)
        a_hdr.setShortcut(QKeySequence("Ctrl+H"))
        a_hdr.setToolTip(
            "Junta las tomas de un bracketing (la oscura, la normal y la\n"
            "clara) en una sola foto con detalle en todo: cielo sin quemar\n"
            "y sombras abiertas.\n\n"
            "Selecciona las tomas en la tira y pulsa aquí. Si no seleccionas\n"
            "nada, PhotoRAW busca solo las tandas de la carpeta.\n"
            "Atajo: Ctrl+H.")
        a_hdr.triggered.connect(self.merge_hdr)
        tb.addAction(a_hdr)

        tb.addSeparator()

        a_copy = QAction(icon("mdi6.content-copy"), "Copiar ajustes", self)
        a_copy.setShortcut(QKeySequence("Ctrl+Shift+C"))
        a_copy.triggered.connect(self.copy_edits)
        tb.addAction(a_copy)

        a_paste = QAction(icon("mdi6.content-paste"), "Pegar ajustes", self)
        a_paste.setShortcut(QKeySequence("Ctrl+Shift+V"))
        a_paste.triggered.connect(self.paste_edits)
        tb.addAction(a_paste)

        tb.addSeparator()

        self.a_original = QAction(icon("mdi6.compare"), "Original", self)
        self.a_original.setCheckable(True)
        self.a_original.setShortcut(QKeySequence("O"))
        self.a_original.setToolTip(
            "Antes / después: muestra la foto sin edición para comparar.\n"
            "Atajo: O. Vuelve a pulsarlo (o mueve un ajuste) para ver tu edición.")
        self.a_original.toggled.connect(self.toggle_original)
        tb.addAction(self.a_original)

        a_restore = QAction(icon("mdi6.backup-restore"), "Restaurar foto", self)
        a_restore.setToolTip(
            "Quita TODA la edición de esta foto (ajustes, recorte, corrector)\n"
            "y vuelve al original. Pide confirmación.")
        a_restore.triggered.connect(self.restore_photo)
        tb.addAction(a_restore)

        tb.addSeparator()

        self.a_brush = QAction(icon("mdi6.bandage"), "Corrector", self)
        self.a_brush.setCheckable(True)
        self.a_brush.setShortcut(QKeySequence("B"))
        self.a_brush.setToolTip("Pincel corrector: pinta sobre lo que quieras "
                                "borrar (brillos, manchas, objetos) y pulsa Borrar")
        self.a_brush.toggled.connect(self.toggle_brush)
        tb.addAction(self.a_brush)

        brush_label = QLabel(" Pincel: ")
        self._brush_widgets = [tb.addWidget(brush_label)]
        brush_slider = QSlider(Qt.Horizontal)
        brush_slider.setRange(5, 120)
        brush_slider.setValue(30)
        brush_slider.setFixedWidth(110)
        brush_slider.valueChanged.connect(
            lambda v: self.preview.set_brush_radius(v))
        self._brush_widgets.append(tb.addWidget(brush_slider))
        hint = QLabel(" pinta y suelta para borrar · Ctrl+Z deshace ")
        hint.setStyleSheet("color: #8a8a8a;")
        self._brush_widgets.append(tb.addWidget(hint))
        for w in self._brush_widgets:
            w.setVisible(False)

        undo = QAction("Deshacer borrado", self)
        undo.setShortcut(QKeySequence.Undo)
        undo.triggered.connect(self.undo_heal)
        self.addAction(undo)

        tb.addSeparator()

        # Detener la IA: apagado mientras no hay nada calculando, y en rojo
        # cuando si lo hay (asi de un vistazo sabes si trabaja en segundo
        # plano). Lo enciende y apaga _update_busy.
        self._stop_icon_idle = icon("mdi6.stop-circle-outline")
        self._stop_icon_busy = icon("mdi6.stop-circle", color="#e0554e")
        self.a_stop_ai = QAction(self._stop_icon_idle, "Detener IA", self)
        self.a_stop_ai.setShortcut(QKeySequence("Esc"))
        self.a_stop_ai.setToolTip(
            "Detiene el trabajo de IA en marcha (ruido, rostros, corrector,\n"
            "borrado generativo, máscaras). Atajo: Esc.\n"
            "La tarjeta gráfica puede tardar unos segundos en soltar el\n"
            "último paso; lo calculado a medias se descarta.")
        self.a_stop_ai.triggered.connect(self.cancel_ai)
        tb.addAction(self.a_stop_ai)

        a_models = QAction(icon("mdi6.brain"), "Modelos de IA", self)
        a_models.setToolTip(
            "Qué IAs tiene PhotoRAW, cuáles están descargadas y cuánto\n"
            "ocupan. Desde ahí puedes bajar las que falten o borrar las\n"
            "que no uses.")
        a_models.triggered.connect(self.show_models)
        tb.addAction(a_models)

    # ---------- carpeta y miniaturas ----------

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Elige la carpeta de fotos")
        if not folder:
            return
        self.load_folder(folder)

    def load_folder(self, folder):
        self.thumb_cancel["stop"] = True
        self.thumb_cancel = {"stop": False}
        self.folder = Path(folder)
        self.store = EditStore(self.folder)
        self.current_path = None
        self.base_cache.clear()
        self.render_cache.clear()
        self.no_faces.clear()
        self.decoding.clear()
        self.thumb_pixmaps.clear()
        self.edited_thumbs.clear()
        self.film.clear()
        self.preview.clear_photo("Cargando miniaturas…")

        files = loader.list_photos(self.folder)
        if not files:
            self.preview.clear_photo("No hay fotos compatibles en esta carpeta")
            self.statusBar().showMessage("Carpeta sin fotos compatibles")
            return

        placeholder = QPixmap(150, 150)
        placeholder.fill(Qt.darkGray)
        for path in files:
            item = QListWidgetItem(QIcon(placeholder), path.name)
            item.setData(Qt.UserRole, str(path))
            item.setSizeHint(QSize(165, 180))
            self.film.addItem(item)

        self.setWindowTitle(f"{APP_NAME} — {self.folder.name} ({len(files)} fotos)")
        self.statusBar().showMessage(f"{len(files)} fotos encontradas")
        self.pool.start(ThumbJob(files, self.signals, self.thumb_cancel))
        self.film.setCurrentRow(0)

    def delete_photos(self):
        items = self.film.selectedItems()
        if not items:
            QMessageBox.information(self, APP_NAME,
                                    "Selecciona una o varias fotos de la tira primero.")
            return
        n = len(items)
        msg = QMessageBox(self)
        msg.setWindowTitle("Eliminar foto")
        msg.setText(f"¿Qué quieres hacer con las {n} foto(s) seleccionada(s)?")
        btn_disk = msg.addButton("Eliminar del disco", QMessageBox.DestructiveRole)
        btn_project = msg.addButton("Quitar del proyecto", QMessageBox.ActionRole)
        btn_cancel = msg.addButton("Cancelar", QMessageBox.RejectRole)
        msg.setDefaultButton(btn_cancel)
        msg.setInformativeText(
            '"Eliminar del disco" borra el archivo permanentemente.\n'
            '"Quitar del proyecto" solo la oculta de la tira.')
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_cancel:
            return
        delete_from_disk = (clicked == btn_disk)
        removed_paths = []
        for item in list(items):
            path = item.data(Qt.UserRole)
            removed_paths.append(path)
            row = self.film.row(item)
            self.film.takeItem(row)
            self.thumb_pixmaps.pop(path, None)
            self.edited_thumbs.pop(path, None)
            self.base_cache.pop(path, None)
            self.render_cache.pop(path, None)
            self.no_faces.discard(path)
            self.ai_cache.pop(path, None)
            self.face_cache.pop(path, None)
            self.small_cache = {k: v for k, v in self.small_cache.items()
                                if k[0] != path}
            self.healed.discard(path)
            self.erased.discard(path)
            if delete_from_disk:
                try:
                    Path(path).unlink()
                except Exception as exc:
                    QMessageBox.warning(
                        self, APP_NAME, f"No se pudo eliminar {Path(path).name}: {exc}")
                self.store.data.pop(Path(path).name, None)
        remaining = self.film.count()
        self.setWindowTitle(
            f"{APP_NAME} — {self.folder.name} ({remaining} {'foto' if remaining == 1 else 'fotos'})")
        self.statusBar().showMessage(
            f"{'Eliminadas' if delete_from_disk else 'Quitadas'} {n} foto(s)")
        if self.current_path in removed_paths:
            self.current_path = None
            if self.film.count():
                self.film.setCurrentRow(0)
            else:
                self.preview.clear_photo("No hay fotos en esta carpeta")

    def _film_context_menu(self, pos):
        item = self.film.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        n = len(self.film.selectedItems())
        hdr_action = menu.addAction(
            icon("mdi6.hdr"),
            f"Fusionar HDR ({n} tomas)" if n > 1 else "Fusionar HDR…")
        hdr_action.triggered.connect(self.merge_hdr)
        menu.addSeparator()
        del_action = menu.addAction(icon("mdi6.delete-outline"), "Eliminar")
        del_action.triggered.connect(self.delete_photos)
        menu.exec(self.film.viewport().mapToGlobal(pos))

    def on_thumb_ready(self, path, image):
        self._update_film_icon(path, image)

    def _film_item(self, path):
        for i in range(self.film.count()):
            item = self.film.item(i)
            if item.data(Qt.UserRole) == path:
                return item
        return None

    def _update_film_icon(self, path, image=None, rendered=False):
        """Actualiza la miniatura de la tira; si la foto tiene edicion
        guardada le pone la insignia del lapiz (como Lightroom).

        Se guardan por separado la foto **tal cual la tomo la camara** (su
        miniatura incrustada) y el revelado, y se ensena una u otra segun la
        foto tenga edicion o no. Asi una tanda de bracketing se ve como es
        —una oscura, una clara, una normal— en vez de igualarse en cuanto
        abres cada toma: el revelado corrige la exposicion de cada foto por
        su cuenta y borraba justo lo que las distingue. Guardarlas separadas
        (y no sobreescribir) es lo que permite que al quitarle la edicion a
        una foto su miniatura vuelva sola a la de la camara.
        """
        if image is not None and not image.isNull():
            pm = QPixmap.fromImage(image).scaled(
                150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if rendered:
                self.edited_thumbs[path] = pm
            else:
                self.thumb_pixmaps[path] = pm
        editada = bool(self.store and self.store.get(path))
        pm = self.edited_thumbs.get(path) if editada else None
        if pm is None:
            pm = self.thumb_pixmaps.get(path)
        item = self._film_item(path)
        if item is None or pm is None:
            return
        if self.store and self.store.get(path):
            pm = pm.copy()
            painter = QPainter(pm)
            painter.setRenderHint(QPainter.Antialiasing)
            r = 11
            cx, cy = pm.width() - r - 4, pm.height() - r - 4
            painter.setBrush(QColor(20, 20, 20, 200))
            painter.setPen(QPen(QColor(255, 255, 255, 90)))
            painter.drawEllipse(QPointF(cx, cy), r, r)
            icon("mdi6.pencil").paint(painter, cx - 7, cy - 7, 14, 14)
            painter.end()
        item.setIcon(QIcon(pm))

    # ---------- seleccion y renderizado ----------

    def on_photo_selected(self, item, _prev=None):
        if item is None:
            return
        # guardado de mascaras pendiente: al disco antes de cambiar de foto
        if self.mask_save_timer.isActive():
            self.mask_save_timer.stop()
            self._save_masks()
        path = item.data(Qt.UserRole)
        self.current_path = path
        self.ai_paused = False   # foto nueva, la pausa del Detener no aplica
        if self.a_original.isChecked():
            self.a_original.blockSignals(True)
            self.a_original.setChecked(False)
            self.a_original.blockSignals(False)
        # el historial de deshacer solo se conserva para la foto activa
        self.heal_undo = {path: self.heal_undo.get(path, [])}
        # las copias reducidas del borrador tambien
        self.small_cache = {k: v for k, v in self.small_cache.items()
                            if k[0] == path}
        self.fit_next = True
        self.current_edits = engine.full_edits(self.store.get(path))
        self._sync_sliders()
        self.statusBar().showMessage(Path(path).name)

        self._refresh_mask_list()
        if path in self.base_cache:
            # Al abrir una foto se va directo a la calidad final: el borrador
            # no se guarda, asi que pedirlo primero costaria medio segundo y
            # otro tanto de espera antes de mirar el revelado ya guardado
            self.request_render(final=True)
        elif path not in self.decoding:
            self.decoding.add(path)
            self.preview.clear_photo("Cargando foto…")
            self.fast_pool.start(DecodeJob(path, self))
        else:
            self.preview.clear_photo("Cargando foto…")
        self._update_busy()

    def on_base_decoded(self, path, base):
        # Llamado desde el hilo de trabajo: solo guarda y avisa por señal.
        self.decoding.discard(path)
        if base is not None:
            self.base_cache[path] = base
            while len(self.base_cache) > 6:
                self.base_cache.popitem(last=False)
        self.signals.base_ready.emit(path)

    def on_base_ready(self, path):
        self._update_busy()
        if path == self.current_path:
            if path in self.base_cache:
                self.request_render(final=True)   # foto recien abierta
                # Trazos del corrector pendientes: primero se aplican, y al
                # terminar se relanza la IA guardada; si no hay, IA directa
                strokes = self.current_edits.get("heal_strokes")
                hb, wb = self.base_cache[path].shape[:2]
                if (strokes and path not in self.healed
                        and path not in self.heal_running
                        and (heal.model_available()
                             or not heal.needs_model(strokes, hb, wb))):
                    self.heal_running.add(path)
                    self.pool.start(HealJob(path, self.base_cache[path],
                                            strokes, self))
                elif not self._autorun_erase(path):
                    self._autorun_ai(path)
                self._update_busy()
            else:
                self.preview.clear_photo("No se pudo abrir esta foto")

    DRAFT_LONG = 1000  # lado largo del borrador de edicion rapida
    # ajustes de detalle: si se esta moviendo uno de estos, el borrador
    # no puede saltarselos
    DETAIL_KEYS = {"nr_luminance", "nr_color", "texture", "clarity", "sharp_amount",
                   "sharp_radius", "sharp_detail", "sharp_masking",
                   "grain_amount", "grain_size", "grain_rough", "ai_denoise",
                   "ai_face"}

    def _small(self, path, kind, arr):
        """Copia reducida de `arr` cacheada; se invalida sola si `arr` cambia."""
        key = (path, kind)
        hit = self.small_cache.get(key)
        if hit is not None and hit[0] is arr:
            return hit[1]
        h, w = arr.shape[:2]
        scale = self.DRAFT_LONG / max(h, w)
        if scale >= 1.0:
            small = arr
        else:
            import cv2
            small = cv2.resize(arr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        self.small_cache[key] = (arr, small)
        return small

    # ---------- cache del revelado terminado ----------

    RENDER_CACHE_MAX = 6      # fotos con su revelado listo en memoria

    def _render_tag(self, edits, den, fac, masks):
        """Todo lo que cambia un revelado, menos la foto de partida.

        Los ingredientes de IA no hace falta resumirlos por su contenido: son
        deterministas a partir del `base` (que ya entra en la clave por su
        propia huella) y del modelo, asi que basta con apuntar cuales entran y
        con que modelo de rostros.
        """
        return json.dumps({
            "edits": edits,
            "den": den is not None,
            "fac": faces.current_model() if fac is not None else None,
            "masks": sorted(masks) if masks else None,
        }, sort_keys=True, default=str)

    def _render_from_memory(self, path, tag, base, den, fac, masks):
        """Revelado ya hecho de esta foto, si sigue valiendo.

        Los arrays se comparan por identidad, como en `_small`: si el objeto
        es el mismo, su contenido tambien. Guardar la referencia los mantiene
        vivos, asi que no hay riesgo de acertar con uno que ya no existe.
        """
        hit = self.render_cache.get(path)
        if hit is None:
            return None
        h_tag, h_base, h_den, h_fac, h_masks, image, arr = hit
        if (h_tag == tag and h_base is base and h_den is den
                and h_fac is fac and h_masks is masks):
            self.render_cache.move_to_end(path)
            return image, arr
        return None

    def _store_render(self, path, image, arr):
        """Guarda en memoria el revelado que se acaba de pintar."""
        fp = getattr(self, "_cacheable_fp", None)
        if not fp or fp[0] != path:
            return
        _p, tag, base, den, fac, masks = fp
        self.render_cache[path] = (tag, base, den, fac, masks, image, arr)
        self.render_cache.move_to_end(path)
        while len(self.render_cache) > self.RENDER_CACHE_MAX:
            self.render_cache.popitem(last=False)

    def request_render(self, final=False):
        path = self.current_path
        if not path or path not in self.base_cache:
            return
        self.gen += 1
        self._cacheable_fp = None
        self._cacheable_gen = None
        if self.a_original.isChecked():
            # vista "antes": la foto tal cual, sin ajustes ni IA ni recorte
            self._start_render(RenderJob(path, self.base_cache[path], {},
                                         self.gen, self.signals))
            return
        edits = dict(self.current_edits)
        if self.show_mask:
            edits["_show_mask"] = True  # solo para esta pasada, no se guarda
        if self.crop_mode:
            edits["_skip_crop"] = True  # en la pestana Recorte se ve todo el marco
        if self.a_brush.isChecked():
            # el corrector pinta sobre la foto original, sin girar ni recortar;
            # y siempre a resolucion completa para que los trazos no se pierdan
            edits["_skip_geometry"] = True
            final = True
        base = self.base_cache[path]
        den = self.ai_cache.get(path)
        fac = self.face_cache.get(path)
        if not final:
            # borrador rapido a resolucion reducida; la calidad completa
            # llega sola cuando dejas de mover los ajustes
            base = self._small(path, "base", base)
            den = self._small(path, "den", den) if den is not None else None
            fac = self._small(path, "face", fac) if fac is not None else None
            if getattr(self, "_last_edit_key", None) not in self.DETAIL_KEYS:
                edits["_draft_skip_detail"] = True
            self.final_timer.start()
        else:
            self.final_timer.stop()

        # Solo se guarda el revelado de calidad y "normal": los borradores no
        # valen para volver a la foto, y las pasadas especiales (ver la
        # mascara, el marco de recorte, el corrector) llevan banderas `_...`
        # que cambian la imagen y no son lo que se quiere conservar.
        masks = self.mask_ai_cache.get(path)
        cacheable = final and not any(k.startswith("_") for k in edits)
        tag = None
        if cacheable:
            tag = self._render_tag(edits, den, fac, masks)
            hit = self._render_from_memory(path, tag, base, den, fac, masks)
            if hit is not None:
                # ya revelada y nada ha cambiado: se pinta y no se calcula nada
                self.on_preview_ready(path, self.gen, hit[0], hit[1])
                return
            self._cacheable_fp = (path, tag, base, den, fac, masks)
            self._cacheable_gen = self.gen

        self._start_render(RenderJob(path, base, edits, self.gen, self.signals,
                                     denoised=den, faced=fac, ai_masks=masks,
                                     cache_tag=tag))

    def _start_render(self, job):
        """Fila india: si ya hay un render en marcha, este espera su turno.
        Solo se guarda el ultimo pedido; los intermedios se descartan porque
        de todos modos llegarian obsoletos."""
        if self._render_busy:
            self._render_next = job
            return
        self._render_busy = True
        self.fast_pool.start(job)

    def _on_render_done(self):
        self._render_busy = False
        if self._render_next is not None:
            job, self._render_next = self._render_next, None
            self._render_busy = True
            self.fast_pool.start(job)

    # ---------- reduccion de ruido IA ----------

    def run_ai_denoise(self, _checked=False, path=None):
        if path is None:
            self.ai_paused = False   # lo pides tu: se acabo la pausa
        path = path or self.current_path
        if not path or path not in self.base_cache or path in self.ai_running:
            return
        if not ai.model_available():
            answer = QMessageBox.question(
                self, APP_NAME,
                "Hace falta descargar el modelo de IA (~91 MB) una sola vez.\n¿Descargar ahora?")
            if answer == QMessageBox.Yes:
                self.ai_pending_path = path
                self.ai_btn.setEnabled(False)
                self.pool.start(AIModelDownloadJob(self))
            return
        self.ai_running.add(path)
        self.ai_btn.setEnabled(False)
        self.pool.start(AIDenoiseJob(path, self.base_cache[path], self))
        self._update_busy()

    def on_ai_model_downloaded(self):
        # Llamado desde el hilo de descarga
        pending, self.ai_pending_path = self.ai_pending_path, None
        if ai.model_available() and pending:
            self.ai_running.add(pending)
            if pending in self.base_cache:
                self.pool.start(AIDenoiseJob(pending, self.base_cache[pending], self))
                return
            self.ai_running.discard(pending)
        self.signals.ai_ready.emit("")

    def on_ai_denoised(self, path, result):
        # Llamado desde el hilo de IA: guarda y avisa por señal.
        self.ai_running.discard(path)
        if result is not None:
            self.ai_cache[path] = result
            while len(self.ai_cache) > 4:
                self.ai_cache.popitem(last=False)
        self.signals.ai_ready.emit(path)

    def _on_ai_status(self, msg):
        """Estado de los trabajos en segundo plano: barra inferior + pildora."""
        self.statusBar().showMessage(msg)
        self._last_status = msg
        if self.status_chip.isVisible():
            self.status_chip.show_text(msg)

    def _ai_busy(self):
        """Trabajos de IA en marcha (el decodificado no cuenta: es corto y
        no se puede detener)."""
        return (self.ai_running or self.face_running or self.heal_running
                or self.erase_running or self.mask_ai_running
                or self.face_parse_running)

    def cancel_ai(self):
        """Detiene la IA en marcha: ✕ de la pildora o tecla Esc.

        Marca el testigo de parada (cada trabajo aborta en su proximo punto
        de control) y libera la interfaz al momento, sin esperarlos: el aro
        deja de girar y la app vuelve a ser usable aunque la GPU tarde unos
        segundos en soltar el paso que ya tenia entre manos. Lo calculado a
        medias se descarta.

        Ademas deja la IA "en pausa" para esta foto: si no, los procesos
        guardados (ruido, rostros, mascaras) se relanzarian solos al llegar
        el resultado vacio y volveriamos a empezar."""
        if not self._ai_busy():
            # nada que parar: al menos devolvemos la tarjeta grafica, que es
            # lo otro que se suele querer al pulsar aqui
            self.free_gpu(quiet=True)
            self.statusBar().showMessage(
                "No hay ningún trabajo de IA en marcha — modelos "
                "descargados de la tarjeta gráfica")
            return
        self.ai_cancel["stop"] = True
        self.ai_cancel = {"stop": False}   # los proximos trabajos, limpios
        self.ai_paused = True
        for running in (self.ai_running, self.face_running, self.heal_running,
                        self.erase_running, self.mask_ai_running,
                        self.face_parse_running):
            running.clear()
        self._pending_ai_mask = None
        self._pending_face_part = None
        self.ai_btn.setEnabled(True)
        self.face_btn.setEnabled(True)
        self._last_status = ""
        self._update_busy()
        # si pides parar es porque quieres tu tarjeta de vuelta: ademas de
        # cortar el trabajo, se sueltan los modelos que tenia residentes
        self.free_gpu(quiet=True)
        self.statusBar().showMessage(
            "IA detenida y tarjeta gráfica liberada — puede tardar unos "
            "segundos en soltar el último paso")

    IDLE_FREE_MIN = 5   # minutos sin IA antes de devolver la GPU sola

    def free_gpu(self, quiet=False):
        """Descarga los modelos de IA de la tarjeta grafica."""
        self.idle_free_timer.stop()
        ai.release_all_sessions()
        if not quiet:
            self.statusBar().showMessage(
                f"Sin usar la IA {self.IDLE_FREE_MIN} min: modelos "
                "descargados de la tarjeta gráfica (se recargan solos "
                "cuando los vuelvas a usar)")

    def warm_up_heal(self):
        """Deja el corrector listo ANTES de que pintes.

        Crear la sesion de LaMa tarda ~11 s (medido) y, mientras lo hace,
        bloquea el interprete: la ventana se queda congelada aunque el
        trabajo corra en segundo plano. Antes eso pasaba justo despues de
        soltar el primer trazo — pintabas y la app se moria un rato, que es
        lo mas desconcertante posible. Ahora el peaje se paga al ELEGIR la
        herramienta, avisando de lo que va a ocurrir, y el primer trazo ya
        sale en ~1 s. Una sola vez por sesion."""
        if heal._session is not None or not heal.model_available():
            return
        aviso = QProgressDialog("Preparando el corrector…\n\n"
                                "Solo la primera vez: la ventana no "
                                "responderá unos segundos.",
                                None, 0, 0, self)
        aviso.setWindowTitle(APP_NAME)
        aviso.setWindowModality(Qt.WindowModal)
        aviso.setMinimumDuration(0)
        aviso.show()
        QApplication.processEvents()   # que el aviso se pinte ANTES del bloqueo
        try:
            heal._get_session()
        except Exception as exc:
            self.statusBar().showMessage(f"Corrector: no se pudo preparar — {exc}")
        aviso.close()
        self.statusBar().showMessage("Corrector listo: pinta y suelta para borrar")

    def show_models(self):
        """Ventana con todas las IAs del proyecto: cuales estan descargadas,
        cuanto ocupan, y botones para bajarlas o borrarlas."""
        from photoraw.ui.models_dialog import ModelsDialog
        if getattr(self, "_models_dialog", None) is None:
            self._models_dialog = ModelsDialog(self)
        self._models_dialog.refrescar()
        self._models_dialog.show()
        self._models_dialog.raise_()
        self._models_dialog.activateWindow()

    def _touch_ai_idle(self):
        """Reinicia la cuenta atras para devolver la GPU. Se llama cada vez
        que termina un trabajo: mientras encadenes ediciones con IA los
        modelos siguen calientes, y solo se sueltan si de verdad los dejas
        de usar."""
        if self._ai_busy():
            self.idle_free_timer.stop()
        else:
            self.idle_free_timer.start()

    def _update_busy(self):
        working = bool(self.ai_running or self.face_running
                       or self.heal_running or self.decoding
                       or self.mask_ai_running or self.face_parse_running
                       or self.erase_running)
        self.busy_bar.setVisible(working)
        # el boton de la barra solo se puede pulsar (y se pone rojo) cuando
        # hay IA que detener; el decodificado no cuenta, no se puede parar
        # el boton se pone rojo cuando hay IA que detener, pero se puede
        # pulsar siempre: en reposo te dice que no hay nada corriendo y te
        # devuelve la tarjeta grafica (antes parecia que "no hacia nada")
        stoppable = bool(self._ai_busy())
        if getattr(self, "_stop_red", None) != stoppable:
            self._stop_red = stoppable
            self.a_stop_ai.setIcon(self._stop_icon_busy if stoppable
                                   else self._stop_icon_idle)
        if working:
            self.status_chip.show_text(self._last_status or "Procesando…")
        else:
            self._last_status = ""
            self.status_chip.hide_chip()
        self._touch_ai_idle()

    # ---------- pincel corrector ----------

    def toggle_brush(self, on):
        if on and self.pick_btn.isChecked():
            self.pick_btn.setChecked(False)
        if on and self.mask_brush_btn.isChecked():
            self.mask_brush_btn.setChecked(False)
        if on and self.panel_stack.currentIndex() != 0:
            self.switch_panel("revelar")
        self.preview.set_brush_mode(on)
        for w in self._brush_widgets:
            w.setVisible(on)
        if on:
            self.warm_up_heal()   # el peaje de LaMa, antes de que pintes
        if not on:
            self.preview.clear_strokes()
        # El corrector pinta sobre la foto SIN girar ni recortar. Solo hace
        # falta reencuadrar si esa geometria cambia la imagen; si la foto no
        # esta recortada ni girada es la misma de siempre y se conserva el
        # zoom donde lo tuvieras. Antes saltaba a pantalla completa siempre,
        # justo cuando te habias acercado a la mota que ibas a borrar.
        self.fit_next = self._geometry_changes_image()
        self.request_render()

    def _geometry_changes_image(self):
        """¿El recorte/giro hacen que la foto revelada no coincida con la
        original? (las claves son las de engine._apply_geometry)"""
        e = self.current_edits
        return bool(e.get("crop") or int(e.get("rot90", 0) or 0)
                    or e.get("flip_h") or e.get("flip_v")
                    or float(e.get("straighten") or 0.0))

    def on_strokes_changed(self):
        """Al soltar el raton: trazo de mascara o borrado del corrector."""
        if not self.preview.has_strokes():
            return
        self.ai_paused = False   # lo pides tu: se acabo la pausa
        if self.mask_refine_sign:
            strokes = [[r, pts, self.mask_refine_sign]
                       for r, pts in self.preview.take_strokes()]
            m = self._current_mask()
            if m is None:
                self.statusBar().showMessage(
                    "Crea o selecciona primero una máscara")
                return
            m["refine"] = list(m.get("refine") or []) + strokes
            self._save_masks()
            self._update_mask_overlay()
            self.request_render()
            return
        if self.mask_brush_on:
            sign = -1 if self.mask_brush_erase.isChecked() else 1
            strokes = [[r, pts, sign] for r, pts in self.preview.take_strokes()]
            m = self._current_mask()
            if m is None or m.get("type") != "brush":
                self._create_mask({"type": "brush", "strokes": []})
                m = self._current_mask()
            m["strokes"] = list(m.get("strokes") or []) + strokes
            self._save_masks()
            self._update_mask_overlay()
            self.request_render()
            return
        path = self.current_path
        if not path or path not in self.base_cache:
            return
        if path in self.heal_running:
            return  # se aplicara al terminar el borrado en curso
        # los trazos pequenos se rellenan al instante sin modelo; LaMa solo
        # hace falta para borrar cosas grandes
        h, w = self.base_cache[path].shape[:2]
        if (not heal.model_available()
                and heal.needs_model(self.preview.peek_strokes(), h, w)):
            answer = QMessageBox.question(
                self, APP_NAME,
                "Ese trazo es grande: hace falta descargar el modelo de borrado\n"
                "(~208 MB) una sola vez. ¿Descargar ahora?")
            if answer != QMessageBox.Yes:
                self.preview.clear_strokes()
                return
            try:
                heal.download_model()
            except Exception as exc:
                QMessageBox.warning(self, APP_NAME, f"Falló la descarga: {exc}")
                self.preview.clear_strokes()
                return
        strokes = self.preview.take_strokes()
        # instantanea para Ctrl+Z: base actual y cuantos trazos habia
        stack = self.heal_undo.setdefault(path, [])
        stack.append((len(self.current_edits.get("heal_strokes", [])),
                      len(self.current_edits.get("erase_ops", [])),
                      self.base_cache[path].astype(np.float16)))
        if len(stack) > 10:
            stack.pop(0)
        self.current_edits["heal_strokes"] = (
            list(self.current_edits.get("heal_strokes", [])) + strokes)
        self.store.set(path, self.current_edits)
        self.heal_running.add(path)
        self.pool.start(HealJob(path, self.base_cache[path], strokes, self))
        self._update_busy()

    def undo_heal(self):
        path = self.current_path
        stack = self.heal_undo.get(path)
        if not stack:
            self.statusBar().showMessage("Nada que deshacer")
            return
        if path in self.heal_running or path in self.erase_running:
            self.statusBar().showMessage("Espera a que termine el borrado en curso")
            return
        n_strokes, n_erase, snapshot = stack.pop()
        self.base_cache[path] = snapshot.astype(np.float32)
        self.current_edits["heal_strokes"] = \
            list(self.current_edits.get("heal_strokes", []))[:n_strokes]
        self.current_edits["erase_ops"] = \
            list(self.current_edits.get("erase_ops", []))[:n_erase]
        self.store.set(path, self.current_edits)
        self.ai_cache.pop(path, None)
        self.face_cache.pop(path, None)
        if not self.current_edits["heal_strokes"]:
            self.healed.discard(path)
        if not self.current_edits["erase_ops"]:
            self.erased.discard(path)
        self.statusBar().showMessage("Borrado deshecho")
        self.request_render()
        self._autorun_ai(path)

    # ---------- borrado generativo ----------

    def run_generative_erase(self):
        self.ai_paused = False   # lo pides tu: se acabo la pausa
        if self.erase_alive:
            # tipico tras Detener: el hilo anterior sigue en la GPU. Lanzar
            # otro cargaria dos veces el modelo (~2 GB cada uno) y la 4070
            # de 8 GB se desbordaria, que es justo lo que la vuelve lentisima
            self.statusBar().showMessage(
                "Espera unos segundos: el borrado anterior aún está "
                "soltando la tarjeta gráfica")
            return
        path = self.current_path
        m = self._current_mask()
        if m is None:
            self.statusBar().showMessage(
                "Selecciona la máscara que cubre lo que quieres borrar")
            return
        if not path or path not in self.base_cache:
            return
        if path in self.erase_running:
            self.statusBar().showMessage("Ya hay un borrado en marcha…")
            return
        if not generative.model_available():
            if QMessageBox.question(
                    self, APP_NAME,
                    "Hace falta descargar el modelo generativo (~2 GB) una "
                    "sola vez.\nPuede tardar bastante. ¿Descargar ahora?") \
                    != QMessageBox.Yes:
                return
            dl = QProgressDialog("Descargando modelo generativo…",
                                 None, 0, 100, self)
            dl.setWindowModality(Qt.WindowModal)
            dl.setMinimumDuration(0)
            try:
                generative.download_model(
                    lambda p: (dl.setValue(int(p * 100)),
                               QApplication.processEvents()))
            except Exception as exc:
                dl.close()
                QMessageBox.warning(self, APP_NAME, f"Falló la descarga: {exc}")
                return
            dl.close()
        base = self.base_cache[path]
        h, w = base.shape[:2]
        e = engine.full_edits(self.current_edits)
        wmap = engine.mask_weight_base(m, e, h, w,
                                       self.mask_ai_cache.get(path))
        if wmap is None:
            self.statusBar().showMessage(
                "La máscara IA aún se está calculando — prueba en un momento")
            return
        if float(wmap.max()) <= 0.5:
            self.statusBar().showMessage("La máscara no cubre nada que borrar")
            return
        # instantanea para Ctrl+Z, como el corrector
        stack = self.heal_undo.setdefault(path, [])
        stack.append((len(self.current_edits.get("heal_strokes", [])),
                      len(self.current_edits.get("erase_ops", [])),
                      base.astype(np.float16)))
        if len(stack) > 10:
            stack.pop(0)
        op = {"map": generative.encode_map(wmap),
              "seed": int(np.random.default_rng().integers(1, 2**31))}
        self.current_edits["erase_ops"] = (
            list(self.current_edits.get("erase_ops", [])) + [op])
        self.store.set(path, self.current_edits)
        self.erase_running.add(path)
        self.erase_alive += 1
        # la base ya lleva los borrados anteriores: solo se aplica el nuevo.
        # drop_on_cancel: si lo detienes, este borrado no debe quedar guardado
        self.pool.start(EraseJob(path, base, [op], self, drop_on_cancel=True))
        self._update_busy()
        self.statusBar().showMessage(
            "Borrado generativo en marcha… (unos segundos · Esc o ✕ para parar)")

    def _autorun_erase(self, path):
        """Re-aplica los borrados generativos guardados al reabrir la foto.
        Devuelve True si lanzo el trabajo (encadena la demas IA al acabar)."""
        if self.ai_paused or self.erase_alive:
            return False   # detenida a proposito (o el hilo anterior sigue
                           # soltando la GPU): el borrado guardado espera
        ops = self.current_edits.get("erase_ops")
        if (ops and path not in self.erased
                and path not in self.erase_running
                and path in self.base_cache
                and generative.model_available()):
            self.erase_running.add(path)
            self.erase_alive += 1
            self.pool.start(EraseJob(path, self.base_cache[path],
                                     list(ops), self))
            self._update_busy()
            return True
        return False

    def on_erased(self, path, result, union_map):
        # Llamado desde el hilo de trabajo: guarda y avisa por señal.
        # Aqui acaba el trabajo de verdad (tambien si se detuvo): a partir
        # de ahora la GPU esta libre para otro borrado.
        self.erase_alive = max(self.erase_alive - 1, 0)
        self.erase_running.discard(path)
        if result is not None:
            self.base_cache[path] = result
            self.erased.add(path)
            # igual que el corrector: la zona borrada se copia a las caches
            # IA en vez de recalcularlas enteras
            if union_map is not None:
                blend = cv2.GaussianBlur(union_map, (0, 0), 4)[..., None]
                for cache in (self.ai_cache, self.face_cache):
                    arr = cache.get(path)
                    if arr is not None and arr.shape == result.shape:
                        cache[path] = arr * (1.0 - blend) + result * blend
            self.small_cache = {k: v for k, v in self.small_cache.items()
                                if k[0] != path}
        self.signals.erase_ready.emit(path)

    def _drop_heal_strokes(self, path, strokes):
        """Igual que _drop_erase_ops, para los trazos del corrector que se
        quedaron sin aplicar al detener el trabajo."""
        if path != self.current_path or not self.store:
            return
        self.current_edits["heal_strokes"] = [
            s for s in self.current_edits.get("heal_strokes", [])
            if not any(s is dropped for dropped in strokes)]
        self.store.set(path, self.current_edits)
        stack = self.heal_undo.get(path)
        if stack:
            stack.pop()

    def _drop_erase_ops(self, path, ops):
        """Quita del historial un borrado generativo que se detuvo a medias:
        no llego a aplicarse, asi que no debe quedar guardado ni reintentarse
        al reabrir la foto. Tambien sobra su instantanea de Ctrl+Z."""
        if path != self.current_path or not self.store:
            return
        self.current_edits["erase_ops"] = [
            o for o in self.current_edits.get("erase_ops", [])
            if not any(o is dropped for dropped in ops)]
        self.store.set(path, self.current_edits)
        stack = self.heal_undo.get(path)
        if stack:
            stack.pop()

    def on_erase_ready(self, path):
        req, self.erase_undo_request = self.erase_undo_request, None
        if req is not None and req[0] == path:
            self._drop_erase_ops(path, req[1])
        self._update_busy()
        if path != self.current_path or self.ai_paused:
            return   # detenido: ni mensaje de "listo" ni encadenar mas IA
        self.statusBar().showMessage("Borrado generativo: listo")
        self.request_render()
        self._autorun_ai(path)

    def on_healed(self, path, result, strokes):
        # Llamado desde el hilo de trabajo: guarda y avisa por señal.
        self.heal_running.discard(path)
        if result is not None:
            self.base_cache[path] = result
            self.healed.add(path)
            # en vez de recalcular la IA de ruido/rostros (varios segundos),
            # se copia la zona corregida a sus caches: el relleno ya sale
            # limpio y el resto de la foto no cambio
            h, w = result.shape[:2]
            mask = heal.rasterize_strokes(strokes, h, w)
            blend = cv2.GaussianBlur((mask > 0).astype(np.float32),
                                     (0, 0), 4)[..., None]
            for cache in (self.ai_cache, self.face_cache):
                arr = cache.get(path)
                if arr is not None and arr.shape == result.shape:
                    cache[path] = arr * (1.0 - blend) + result * blend
            # las copias reducidas del borrador quedan obsoletas
            self.small_cache = {k: v for k, v in self.small_cache.items()
                                if k[0] != path}
        self.signals.heal_ready.emit(path)

    def on_heal_ready(self, path):
        req, self.heal_undo_request = self.heal_undo_request, None
        if req is not None and req[0] == path:
            self._drop_heal_strokes(path, req[1])
        self._update_busy()
        if path != self.current_path or self.ai_paused:
            return   # detenido: ni mensaje de "listo" ni encadenar mas IA
        self.statusBar().showMessage("Corrector: listo")
        self.request_render()
        if not self._autorun_erase(path):
            self._autorun_ai(path)
        # si el usuario pinto mas trazos mientras se procesaba, van ahora
        if self.preview.has_strokes():
            self.on_strokes_changed()

    def _autorun_ai(self, path):
        """Relanza los procesos IA guardados de la foto si aun no hay cache."""
        if self.ai_paused:
            return   # acabas de pulsar Detener: no volver a arrancarla sola
        masks = self.current_edits.get("masks") or []
        adj = [m.get("adjust") or {} for m in masks]
        need_den = any(a.get("ai_denoise") for a in adj)
        need_fac = any(a.get("ai_face") for a in adj)
        if ((self.current_edits.get("ai_denoise") or need_den)
                and path not in self.ai_cache
                and path not in self.ai_running
                and ai.model_available()):
            if not self.current_edits.get("ai_denoise"):
                self._ai_from_mask.add((path, "ai_denoise"))
            self.run_ai_denoise(path=path)
        if ((self.current_edits.get("ai_face") or need_fac)
                and path not in self.face_cache
                and path not in self.no_faces
                and path not in self.face_running
                and faces.models_available()):
            if not self.current_edits.get("ai_face"):
                self._ai_from_mask.add((path, "ai_face"))
            self.run_ai_faces(path=path)
        # mapas de las mascaras IA guardadas (sujeto/fondo y retrato)
        cache = self.mask_ai_cache.get(path) or {}
        if (any(m.get("type") in ("subject", "background") for m in masks)
                and cache.get("subject") is None
                and path not in self.mask_ai_running
                and masks_ai.model_available()):
            self.mask_ai_running.add(path)
            self.pool.start(MaskAIJob(path, self.base_cache[path], self))
        if (any(m.get("type") == "face_part" for m in masks)
                and cache.get("face_labels") is None
                and path not in self.face_parse_running
                and face_parse.model_available()):
            self.face_parse_running.add(path)
            self.pool.start(FaceParseJob(path, self.base_cache[path], self))

    # ---------- retoque de rostros IA ----------

    def on_face_model_changed(self, index):
        faces.set_model("CodeFormer" if index == 0 else "GFPGAN")
        self.face_cache.clear()  # los resultados del otro modelo ya no valen
        self.statusBar().showMessage(
            f"Modelo de rostros: {faces.current_model()} — vuelve a pulsar Retocar")

    def run_ai_faces(self, _checked=False, path=None):
        if path is None:
            self.ai_paused = False   # lo pides tu: se acabo la pausa
        path = path or self.current_path
        if not path or path not in self.base_cache or path in self.face_running:
            return
        if not faces.models_available():
            answer = QMessageBox.question(
                self, APP_NAME,
                "Hace falta descargar los modelos de rostros (~90 MB) una sola vez.\n"
                "¿Descargar ahora?")
            if answer != QMessageBox.Yes:
                return
            try:
                faces.download_models()
            except Exception as exc:
                QMessageBox.warning(self, APP_NAME, f"Falló la descarga: {exc}")
                return
        self.face_running.add(path)
        self.face_btn.setEnabled(False)
        self.pool.start(AIFaceJob(path, self.base_cache[path], self))
        self._update_busy()

    def on_ai_faces_done(self, path, result, n_faces):
        # Llamado desde el hilo de IA: guarda y avisa por señal.
        self.face_running.discard(path)
        if result is not None:
            self.face_cache[path] = result
            while len(self.face_cache) > 4:
                self.face_cache.popitem(last=False)
        elif n_faces == 0:
            # sin caras: se anota para no volver a buscarlas cada vez que se
            # abre esta foto (el ajuste de rostros sigue guardado en ella)
            self.no_faces.add(path)
        self._last_face_count = n_faces
        self.signals.face_ready.emit(path)

    def on_face_ready(self, path):
        self.face_btn.setEnabled(True)
        self._update_busy()
        if not path or path != self.current_path:
            return
        if path in self.face_cache:
            from_mask = (path, "ai_face") in self._ai_from_mask
            self._ai_from_mask.discard((path, "ai_face"))
            if not self.current_edits.get("ai_face") and not from_mask:
                self.current_edits["ai_face"] = 100.0
                self.store.set(path, self.current_edits)
                self._sync_sliders()
            n = getattr(self, "_last_face_count", 0)
            self.statusBar().showMessage(f"IA rostros: {n} cara(s) retocada(s)")
            self.request_render()

    def on_ai_ready(self, path):
        self.ai_btn.setEnabled(True)
        self._update_busy()
        if not path or path != self.current_path:
            return
        if path in self.ai_cache:
            from_mask = (path, "ai_denoise") in self._ai_from_mask
            self._ai_from_mask.discard((path, "ai_denoise"))
            if not self.current_edits.get("ai_denoise") and not from_mask:
                self.current_edits["ai_denoise"] = 100.0
                self.store.set(path, self.current_edits)
                self._sync_sliders()
            device = "GPU" if ai.gpu_in_use() else "CPU"
            self.statusBar().showMessage(f"IA: listo ({device})")
            self.request_render()

    def on_preview_ready(self, path, gen, image, arr=None):
        if gen == self.gen and path == self.current_path:
            # Mismo `gen` = nada ha cambiado desde que se pidio, asi que este
            # revelado sigue siendo el que toca y se puede guardar
            if self._cacheable_gen == gen and arr is not None:
                self._store_render(path, image, arr)
            self.preview.set_photo(QPixmap.fromImage(image), fit=self.fit_next)
            self.fit_next = False
            if arr is not None:
                self.histogram.set_image(arr)
            if self.crop_mode:
                self.preview.set_crop_rect(self.current_edits.get("crop") or None)
            elif not (self.show_mask or self.a_original.isChecked()
                      or self.a_brush.isChecked()):
                # se guarda como miniatura del revelado; _update_film_icon
                # decide si toca ensenarla o dejar la de la camara
                self._update_film_icon(path, image, rendered=True)
            if self.panel_stack.currentIndex() == 2:
                self._throttle_overlay()

    # ---------- ajustes ----------

    _STYLE_BTN_OFF = (
        "QPushButton { background: #3a3a3a; color: #888; border: 1px solid #555;"
        " border-radius: 3px; font-weight: bold; font-size: 11px; }"
        "QPushButton:hover { background: #4a6a8a; color: #fff; }")
    _STYLE_BTN_ON = (
        "QPushButton { background: #2a6aa8; color: #fff; border: 1px solid #4a9ae8;"
        " border-radius: 3px; font-weight: bold; font-size: 11px; }"
        "QPushButton:hover { background: #3a7ab8; }")

    def _update_auto_btn_style(self, btn, active):
        btn.setStyleSheet(self._STYLE_BTN_ON if active else self._STYLE_BTN_OFF)

    def _auto_exposure(self, key):
        """Calcula la exposicion optima y aplica al slider."""
        if key != "exposure" or not self.current_path:
            return
        base = self.base_cache.get(self.current_path)
        if base is None:
            return
        btn = self._auto_exp_btn
        # Toggle: si ya esta activo, desactivar (volver a 0)
        if btn and getattr(self, '_auto_exp_active', False):
            self.current_edits["exposure"] = 0.0
            slider = self.sliders.get("exposure")
            if slider:
                slider.blockSignals(True)
                slider.setValue(0)
                slider.blockSignals(False)
            vl = self.value_labels.get("exposure")
            if vl:
                vl.setText("0")
            self._auto_exp_active = False
            if btn:
                self._update_auto_btn_style(btn, False)
            if self.current_path:
                self.store.set(self.current_path, self.current_edits)
            self._throttle_render()
            return
        
        # Calcular EV optimo sobre la foto YA renderizada con tus ajustes
        # actuales (perfil, correccion automatica del RAW al abrir, etc.)
        # con la propia exposicion en 0: mide cuanto le falta o le sobra
        # DESPUES de esas correcciones, para no aplicarlas dos veces.
        probe_edits = {**self.current_edits, "exposure": 0.0,
                       "_draft_skip_detail": True}
        is_raw = loader.is_raw(self.current_path)
        rendered = engine.apply_edits(base, probe_edits, is_raw=is_raw)
        ev = engine.auto_exposure(rendered.astype(np.float32) / 255.0)
        
        # Aplicar al slider de exposicion
        self.current_edits["exposure"] = ev
        slider = self.sliders.get("exposure")
        if slider:
            slider.blockSignals(True)
            slider.setValue(int(round(ev * 100.0)))
            slider.blockSignals(False)
        vl = self.value_labels.get("exposure")
        if vl:
            vl.setText(f"{ev:.2f}".rstrip("0").rstrip("."))
        
        # Activar boton
        self._auto_exp_active = True
        if btn:
            self._update_auto_btn_style(btn, True)
        if self.current_path:
            self.store.set(self.current_path, self.current_edits)
        self._throttle_render()

    def on_slider(self, key, value, scale, val_label):
        if self.a_original.isChecked():
            self.a_original.setChecked(False)  # mover un ajuste vuelve a la edición
        self._last_edit_key = key
        real = value / scale
        self.current_edits[key] = real
        val_label.setText(f"{real:.2f}".rstrip("0").rstrip(".") if scale != 1.0 else str(int(real)))
        if key == "exposure" and self._auto_exp_btn and getattr(self, '_auto_exp_active', False):
            self._auto_exp_active = False
            self._update_auto_btn_style(self._auto_exp_btn, False)
        if key == "sharp_masking":
            self.show_mask = bool(QApplication.keyboardModifiers() & Qt.AltModifier)
        if self.current_path:
            self.store.set(self.current_path, self.current_edits)
        self._throttle_render()

    def _end_mask_view(self):
        if self.show_mask:
            self.show_mask = False
            self.request_render()

    def _sync_sliders(self):
        for item in (SLIDERS + PARAM_SLIDERS + PC_SLIDERS + CROP_SLIDERS):
            key = item[0]
            scale = item[4]
            slider = self.sliders[key]
            slider.blockSignals(True)
            slider.setValue(int(round(self.current_edits.get(key, 0.0) * scale)))
            slider.blockSignals(False)
            v = self.current_edits.get(key, 0.0)
            self.value_labels[key].setText(
                f"{v:.2f}".rstrip("0").rstrip(".") if scale != 1.0 else str(int(v)))
        self._sync_curve_widget()
        if hasattr(self, "hsl_sliders"):
            self._sync_hsl_sliders()
            self._update_pc_swatch()
        if hasattr(self, "profile_combo"):
            keys = [k for _l, k in PROFILE_OPTIONS]
            idx = keys.index(self.current_edits.get("profile", "standard")) \
                if self.current_edits.get("profile", "standard") in keys else 0
            self.profile_combo.blockSignals(True)
            self.profile_combo.setCurrentIndex(idx)
            self.profile_combo.blockSignals(False)

    def _sync_curve_widget(self):
        key, _label, color = CURVE_CHANNELS[self.channel_combo.currentIndex()]
        self.curve_widget.set_curve(self.current_edits.get(key, engine.DEFAULT_CURVE), color)

    def on_channel_changed(self, _index):
        self._sync_curve_widget()

    def on_curve_changed(self, points):
        key = CURVE_CHANNELS[self.channel_combo.currentIndex()][0]
        self.current_edits[key] = points
        if self.current_path:
            self.store.set(self.current_path, self.current_edits)
        self._throttle_render()

    # ---------- recorte y transformacion ----------

    def switch_panel(self, key):
        for k, btn in self.tab_buttons.items():
            btn.setChecked(k == key)
        self.panel_stack.setCurrentIndex(
            {"revelar": 0, "recorte": 1, "mascaras": 2}[key])
        on = key == "recorte"
        if key != "revelar":
            if self.a_brush.isChecked():
                self.a_brush.setChecked(False)
            if self.pick_btn.isChecked():
                self.pick_btn.setChecked(False)
        if key != "mascaras":
            if self.mask_brush_btn.isChecked():
                self.mask_brush_btn.setChecked(False)
            self._uncheck_refine()
            self.preview.set_mask_draw_mode(None)
        self.crop_mode = on
        self.preview.set_crop_mode(on)
        if on:
            self.preview.set_crop_rect(self.current_edits.get("crop") or None)
            self.preview.set_crop_aspect(
                ASPECT_RATIOS[self.aspect_combo.currentIndex()][1], reshape=False)
            self.statusBar().showMessage(
                "Recorte: arrastra el marco sobre la foto")
        elif key == "revelar" and self.current_edits.get("crop"):
            self.statusBar().showMessage("Recorte aplicado")
        self._update_mask_overlay()
        # reencuadrar siempre: las dimensiones cambian entre pestañas
        self.fit_next = True
        path = self.current_path
        if path and path not in self.base_cache and path not in self.decoding:
            # la foto salio de la cache: decodificar de nuevo para renderizar
            self.decoding.add(path)
            self.fast_pool.start(DecodeJob(path, self))
            self._update_busy()
        else:
            self.request_render()

    def _update_aspect_hint(self, index):
        """Texto de ayuda con el uso típico de la proporción elegida."""
        self.aspect_hint.setText(ASPECT_RATIOS[index][3])
        self.aspect_combo.setToolTip(ASPECT_RATIOS[index][3])

    def on_aspect_changed(self, index):
        self._update_aspect_hint(index)
        self.preview.set_crop_aspect(ASPECT_RATIOS[index][1])

    def on_crop_changed(self, norm):
        self.current_edits["crop"] = list(norm)
        if self.current_path:
            self.store.set(self.current_path, self.current_edits)

    def rotate90(self, clockwise):
        e = self.current_edits
        e["rot90"] = (int(e.get("rot90", 0)) + (1 if clockwise else 3)) % 4
        crop = e.get("crop")
        if crop:
            x0, y0, x1, y1 = crop
            # el marco gira con la foto
            e["crop"] = ([1 - y1, x0, 1 - y0, x1] if clockwise
                         else [y0, 1 - x1, y1, 1 - x0])
        self._after_geometry_change()

    def flip_photo(self, horizontal):
        e = self.current_edits
        key = "flip_h" if horizontal else "flip_v"
        e[key] = 0 if e.get(key) else 1
        crop = e.get("crop")
        if crop:
            x0, y0, x1, y1 = crop
            e["crop"] = ([1 - x1, y0, 1 - x0, y1] if horizontal
                         else [x0, 1 - y1, x1, 1 - y0])
        self._after_geometry_change()

    def reset_crop(self):
        for key, value in (("crop", []), ("straighten", 0.0),
                           ("rot90", 0), ("flip_h", 0), ("flip_v", 0)):
            self.current_edits[key] = value
        self._sync_sliders()
        self.preview.set_crop_rect(None)
        self._after_geometry_change()

    def _after_geometry_change(self):
        if self.current_path:
            self.store.set(self.current_path, self.current_edits)
        self.fit_next = True  # las dimensiones cambian: reencuadrar la vista
        self.request_render()

    # ---------- mascaras con ajustes locales ----------

    def _masks(self):
        return self.current_edits.setdefault("masks", [])

    def _current_mask(self):
        masks = self._masks()
        i = self.mask_list.currentRow()
        return masks[i] if 0 <= i < len(masks) else None

    def _save_masks(self):
        if self.current_path:
            self.store.set(self.current_path, self.current_edits)

    def _refresh_mask_list(self, select=None):
        self.mask_list.blockSignals(True)
        self.mask_list.clear()
        for i, m in enumerate(self._masks()):
            if m.get("type") == "face_part":
                name = FACE_PART_LABELS.get(m.get("part"), "Retrato")
            else:
                name = MASK_TYPE_NAMES.get(m.get("type"), "Máscara")
            self.mask_list.addItem(f"{i + 1}. {name}")
        count = self.mask_list.count()
        if count:
            row = count - 1 if select is None else min(max(select, 0), count - 1)
            self.mask_list.setCurrentRow(row)
        self.mask_list.blockSignals(False)
        self._sync_mask_controls()

    def _sync_mask_controls(self):
        m = self._current_mask()
        adjust = (m or {}).get("adjust") or {}
        for key, _label, _lo, _hi, scale in MASK_SLIDERS:
            slider = self.mask_sliders[key]
            slider.blockSignals(True)
            v = adjust.get(key, 0.0)
            slider.setValue(int(round(v * scale)))
            slider.blockSignals(False)
            self.mask_value_labels[key].setText(
                f"{v:.2f}".rstrip("0").rstrip(".") if scale != 1.0 else str(int(v)))
        self.mask_invert.blockSignals(True)
        self.mask_invert.setChecked(bool((m or {}).get("invert")))
        self.mask_invert.blockSignals(False)
        is_radial = bool(m) and m.get("type") == "radial"
        self.mask_feather.setEnabled(is_radial)
        self.mask_feather.blockSignals(True)
        self.mask_feather.setValue(int((m or {}).get("feather", 50.0))
                                   if is_radial else 50)
        self.mask_feather.blockSignals(False)

    def on_mask_selected(self, _row):
        self._sync_mask_controls()
        self._update_mask_overlay()

    def on_mask_slider(self, key, value, scale, val_label):
        m = self._current_mask()
        if m is None:
            self.statusBar().showMessage("Crea primero una máscara")
            return
        real = value / scale
        m.setdefault("adjust", {})[key] = real
        val_label.setText(
            f"{real:.2f}".rstrip("0").rstrip(".") if scale != 1.0 else str(int(real)))
        self._last_edit_key = "mask"
        if key in engine.MASK_AI_KEYS and real > 0:
            self._ensure_mask_ai(key)
        self.mask_save_timer.start()
        self._throttle_render()

    def _ensure_mask_ai(self, key):
        """Un deslizador IA de mascara necesita el analisis IA de la foto;
        si aun no esta calculado ni en marcha, se lanza una sola vez."""
        self.ai_paused = False   # mover el deslizador ya es pedirla
        path = self.current_path
        if not path or path not in self.base_cache:
            return
        if key == "ai_denoise":
            have = path in self.ai_cache
            running = path in self.ai_running or bool(self.ai_pending_path)
            available = ai.model_available()
        else:
            have = path in self.face_cache
            running = path in self.face_running
            available = faces.models_available()
        if have or running:
            return
        if not available and key in self._mask_ai_prompted:
            return  # la descarga ya se pregunto en esta sesion
        self._mask_ai_prompted.add(key)
        self._ai_from_mask.add((path, key))
        if key == "ai_denoise":
            self.run_ai_denoise(path=path)
        else:
            self.run_ai_faces(path=path)
        self.statusBar().showMessage(
            "IA calculando… el efecto aparecerá en la zona de la máscara al terminar")

    def on_mask_invert(self, on):
        m = self._current_mask()
        if m is None:
            return
        m["invert"] = 1 if on else 0
        self._save_masks()
        self._update_mask_overlay()
        self.request_render()

    def delete_mask(self):
        i = self.mask_list.currentRow()
        masks = self._masks()
        if 0 <= i < len(masks):
            masks.pop(i)
            self._save_masks()
            self._refresh_mask_list(select=i - 1)
            self._update_mask_overlay()
            self.request_render()

    def start_mask_draw(self, kind):
        if not self.current_path:
            return
        self.mask_brush_btn.setChecked(False)
        self._uncheck_refine()
        self.preview.set_mask_draw_mode(kind)
        self.statusBar().showMessage(
            "Degradado lineal: arrastra desde donde empieza el efecto hacia "
            "donde termina" if kind == "linear"
            else "Degradado radial: arrastra del centro hacia afuera")

    def on_mask_drawn(self, kind, params):
        self.preview.set_mask_draw_mode(None)
        self._create_mask({"type": kind, **params})

    def _create_mask(self, mask):
        mask.setdefault("invert", 0)
        mask.setdefault("adjust", {k: 0.0 for k in engine.MASK_ADJUST_KEYS})
        self._masks().append(mask)
        self._save_masks()
        self._refresh_mask_list(select=len(self._masks()) - 1)
        self._update_mask_overlay()
        self.request_render()
        self.statusBar().showMessage(
            "Máscara creada — mueve sus ajustes para ver el efecto en la zona roja")

    def toggle_mask_brush(self, on):
        if on:
            self.preview.set_mask_draw_mode(None)
            if self.a_brush.isChecked():
                self.a_brush.setChecked(False)
            self._uncheck_refine()
        self.mask_brush_on = on
        self.preview.set_brush_mode(on or self.mask_refine_sign != 0)
        self._update_mask_overlay()
        if on:
            self.statusBar().showMessage(
                "Pinta la zona a ajustar; con «Quitar» activado el pincel borra")

    def _uncheck_refine(self):
        for btn in (self.mask_refine_add, self.mask_refine_sub):
            if btn.isChecked():
                btn.setChecked(False)

    def toggle_mask_refine(self, sign, on):
        """Pincel de refinado: anade (+1) o quita (-1) zonas pintando sobre
        la mascara seleccionada, sea del tipo que sea."""
        btn = self.mask_refine_add if sign > 0 else self.mask_refine_sub
        if on and self._current_mask() is None:
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
            self.statusBar().showMessage("Crea o selecciona primero una máscara")
            return
        if on:
            other = self.mask_refine_sub if sign > 0 else self.mask_refine_add
            if other.isChecked():
                other.setChecked(False)
            if self.mask_brush_btn.isChecked():
                self.mask_brush_btn.setChecked(False)
            if self.a_brush.isChecked():
                self.a_brush.setChecked(False)
            self.preview.set_mask_draw_mode(None)
            self.mask_refine_sign = sign
        else:
            self.mask_refine_sign = (1 if self.mask_refine_add.isChecked()
                                     else -1 if self.mask_refine_sub.isChecked()
                                     else 0)
        self.preview.set_brush_mode(self.mask_refine_sign != 0
                                    or self.mask_brush_on)
        self._update_mask_overlay()
        if on:
            self.statusBar().showMessage(
                "Pinta sobre la foto para AÑADIR esa zona a la máscara"
                if sign > 0 else
                "Pinta sobre la foto para QUITAR esa zona de la máscara")

    def add_ai_mask(self, kind):
        self.ai_paused = False   # lo pides tu: se acabo la pausa
        path = self.current_path
        if not path or path not in self.base_cache:
            return
        if not masks_ai.model_available():
            if QMessageBox.question(
                    self, APP_NAME,
                    "Hace falta descargar el modelo de segmentación (~168 MB) "
                    "una sola vez.\n¿Descargar ahora?") != QMessageBox.Yes:
                return
            dl = QProgressDialog("Descargando modelo…", None, 0, 100, self)
            dl.setWindowModality(Qt.WindowModal)
            dl.setMinimumDuration(0)
            try:
                masks_ai.download_model(
                    lambda p: (dl.setValue(int(p * 100)),
                               QApplication.processEvents()))
            except Exception as exc:
                dl.close()
                QMessageBox.warning(self, APP_NAME, f"Falló la descarga: {exc}")
                return
            dl.close()
        cache = self.mask_ai_cache.get(path) or {}
        if cache.get("subject") is not None:
            self._create_mask({"type": kind})
            return
        if path in self.mask_ai_running:
            self._pending_ai_mask = kind
            return
        self._pending_ai_mask = kind
        self.mask_ai_running.add(path)
        self.pool.start(MaskAIJob(path, self.base_cache[path], self))
        self._update_busy()

    def _store_mask_ai(self, path, key, result):
        """Guarda un mapa IA en la cache por foto sin pisar los demas
        (sujeto y retrato conviven en la misma entrada)."""
        entry = self.mask_ai_cache.pop(path, None) or {}
        entry[key] = result
        self.mask_ai_cache[path] = entry
        while len(self.mask_ai_cache) > 4:
            self.mask_ai_cache.pop(next(iter(self.mask_ai_cache)))

    def add_face_mask(self, part):
        self.ai_paused = False   # lo pides tu: se acabo la pausa
        path = self.current_path
        if not path or path not in self.base_cache:
            return
        if not face_parse.model_available():
            if QMessageBox.question(
                    self, APP_NAME,
                    "Hace falta descargar el modelo de retrato (~90 MB) "
                    "una sola vez.\n¿Descargar ahora?") != QMessageBox.Yes:
                return
            dl = QProgressDialog("Descargando modelo…", None, 0, 100, self)
            dl.setWindowModality(Qt.WindowModal)
            dl.setMinimumDuration(0)
            try:
                face_parse.download_model(
                    lambda p: (dl.setValue(int(p * 100)),
                               QApplication.processEvents()))
            except Exception as exc:
                dl.close()
                QMessageBox.warning(self, APP_NAME, f"Falló la descarga: {exc}")
                return
            dl.close()
        labels = (self.mask_ai_cache.get(path) or {}).get("face_labels")
        if labels is not None:
            if labels.any():
                self._create_mask({"type": "face_part", "part": part})
            else:
                self.statusBar().showMessage(
                    "No se encontró ninguna cara en la foto")
            return
        self._pending_face_part = part
        if path in self.face_parse_running:
            return
        self.face_parse_running.add(path)
        self.pool.start(FaceParseJob(path, self.base_cache[path], self))
        self._update_busy()

    def on_mask_ai_done(self, path, result):
        # Llamado desde el hilo de trabajo: guarda y avisa por señal.
        self.mask_ai_running.discard(path)
        if result is not None:
            self._store_mask_ai(path, "subject", result)
        self.signals.mask_ai_ready.emit(path)

    def on_face_parse_done(self, path, result):
        # Llamado desde el hilo de trabajo: guarda y avisa por señal.
        self.face_parse_running.discard(path)
        if result is not None:
            self._store_mask_ai(path, "face_labels", result)
        self.signals.face_parse_ready.emit(path)

    def on_face_parse_ready(self, path):
        self._update_busy()
        if path != self.current_path:
            return
        pending, self._pending_face_part = self._pending_face_part, None
        labels = (self.mask_ai_cache.get(path) or {}).get("face_labels")
        if labels is None:
            return
        if not labels.any():
            self.statusBar().showMessage(
                "No se encontró ninguna cara en la foto")
            return
        if pending:
            self._create_mask({"type": "face_part", "part": pending})
            self.statusBar().showMessage("Máscara de retrato lista")
        else:
            # mascara guardada de otra sesion: aplicar el mapa recien llegado
            self.request_render()
            self._throttle_overlay()

    def on_mask_ai_ready(self, path):
        self._update_busy()
        if path != self.current_path:
            return
        if (self.mask_ai_cache.get(path) or {}).get("subject") is None:
            return
        pending, self._pending_ai_mask = self._pending_ai_mask, None
        if pending:
            self._create_mask({"type": pending})
            self.statusBar().showMessage("Máscara IA lista")
        else:
            # mascara guardada de otra sesion: aplicar el mapa recien llegado
            self.request_render()
            self._throttle_overlay()

    def on_mask_feather(self, value):
        m = self._current_mask()
        if m is None or m.get("type") != "radial":
            return
        m["feather"] = float(value)
        self.mask_save_timer.start()
        self._throttle_overlay()
        self._throttle_render()

    def _throttle_render(self):
        """Renderiza a ritmo fijo MIENTRAS te mueves (estilo videojuego),
        en vez de esperar a que sueltes."""
        if not self.render_timer.isActive():
            self.render_timer.start()

    def _throttle_overlay(self):
        if not self.overlay_timer.isActive():
            self.overlay_timer.start()

    def on_mask_edited(self, dragging):
        """El usuario mueve un tirador de la mascara. Mientras arrastra solo
        se actualiza el velo rojo (barato y fluido); la foto se revela una
        sola vez al soltar, como en Lightroom."""
        self._last_edit_key = "mask"  # el borrador puede saltarse el detalle
        self.mask_save_timer.start()
        self._throttle_overlay()
        if not dragging:
            self._throttle_render()

    def _suspend_mask_overlay(self):
        self._overlay_suspended = True
        self.preview.set_mask_overlay(None)

    def _resume_mask_overlay(self):
        self._overlay_suspended = False
        self._update_mask_overlay()

    def _update_mask_overlay(self):
        if self.panel_stack.currentIndex() != 2 or not self.preview._has_photo:
            self.preview.set_mask_overlay(None)
            self.preview.set_mask_handles(None)
            return
        m = self._current_mask()
        # tiradores de edicion (solo lineal/radial y sin pinceles activos)
        self.preview.set_mask_handles(
            None if (m is None or self.mask_brush_on
                     or self.mask_refine_sign) else m)
        if m is None:
            self.preview.set_mask_overlay(None)
            return
        self._show_mask_overlay(m)

    def on_mask_dragging(self, _kind, mask):
        """Velo azul en vivo mientras arrastras una mascara nueva."""
        if self.panel_stack.currentIndex() == 2 and self.preview._has_photo:
            self._show_mask_overlay(mask)

    def _show_mask_overlay(self, m):
        """Pinta el velo azul de la mascara `m` sobre la foto."""
        if (not self.mask_show_overlay.isChecked()
                or getattr(self, "_overlay_suspended", False)):
            self.preview.set_mask_overlay(None)
            return
        pm = self.preview._item.pixmap()
        full_h, full_w = pm.height(), pm.width()
        if not full_h or not full_w:
            self.preview.set_mask_overlay(None)
            return
        # el velo se calcula a resolucion reducida: 10x mas rapido y la
        # diferencia no se ve (es un tinte suave escalado por la GPU)
        scale = min(800.0 / max(full_h, full_w), 1.0)
        h = max(int(full_h * scale), 1)
        w = max(int(full_w * scale), 1)

        def get_ai(kind):
            cache = self.mask_ai_cache.get(self.current_path) or {}
            base_map = engine.ai_source_map(cache, kind)
            if base_map is None:
                return None
            return engine.transform_ai_map(base_map, self.current_edits, h, w)

        wmap = engine.mask_weight(m, h, w, get_ai)
        if wmap is None:
            self.preview.set_mask_overlay(None)
            self.statusBar().showMessage("Máscara IA: calculando…")
            return
        rgba = np.zeros((h, w, 4), np.uint8)  # memoria BGRA
        rgba[..., 0] = 40
        rgba[..., 1] = 60
        rgba[..., 2] = 235
        rgba[..., 3] = (wmap * 110.0).astype(np.uint8)
        qimg = QImage(rgba.data, w, h, w * 4, QImage.Format_ARGB32).copy()
        self.preview.set_mask_overlay(qimg, full_w / w)

    # ---------- mezclador HSL y color de punto ----------

    def on_band_selected(self, band):
        self.current_band = band
        for key, btn in self.band_buttons.items():
            btn.setChecked(key == band)
        self._sync_hsl_sliders()
        self._update_hsl_gradients()

    def _update_hsl_gradients(self):
        """Degradados de las pistas segun la banda elegida: indican hacia
        donde se mueve el color en cada direccion."""
        h = BAND_HUES[self.current_band]
        set_slider_gradient(self.hsl_sliders["h"],
                            [hue_color(h - 30), hue_color(h), hue_color(h + 30)])
        set_slider_gradient(self.hsl_sliders["s"],
                            [hue_color(h, s=0.04, v=0.60), hue_color(h, s=1.0)])
        set_slider_gradient(self.hsl_sliders["l"],
                            [hue_color(h, s=0.90, v=0.15),
                             hue_color(h, s=0.45, v=1.0)])

    def _sync_hsl_sliders(self):
        for comp in "hsl":
            v = self.current_edits.get(f"hsl_{self.current_band}_{comp}", 0.0)
            slider = self.hsl_sliders[comp]
            slider.blockSignals(True)
            slider.setValue(int(round(v)))
            slider.blockSignals(False)
            self.hsl_value_labels[comp].setText(str(int(v)))

    def on_hsl_slider(self, comp, value):
        self.current_edits[f"hsl_{self.current_band}_{comp}"] = float(value)
        self.hsl_value_labels[comp].setText(str(int(value)))
        if self.current_path:
            self.store.set(self.current_path, self.current_edits)
        self._throttle_render()

    def on_profile_changed(self, index):
        if self.a_original.isChecked():
            self.a_original.setChecked(False)
        self.current_edits["profile"] = PROFILE_OPTIONS[index][1]
        if self.current_path:
            self.store.set(self.current_path, self.current_edits)
        self.render_timer.start()

    def reset_calibration(self):
        for key in ("cal_shadow_tint", "cal_red_hue", "cal_red_sat",
                    "cal_green_hue", "cal_green_sat", "cal_blue_hue",
                    "cal_blue_sat"):
            self.current_edits[key] = 0.0
        self._sync_sliders()
        if self.current_path:
            self.store.set(self.current_path, self.current_edits)
        self.request_render()

    def reset_hsl(self):
        for band in engine.HSL_BANDS:
            for comp in "hsl":
                self.current_edits[f"hsl_{band}_{comp}"] = 0.0
        self._sync_hsl_sliders()
        if self.current_path:
            self.store.set(self.current_path, self.current_edits)
        self.request_render()

    def toggle_picker(self, on):
        if on and self.a_brush.isChecked():
            self.a_brush.setChecked(False)
        self.preview.set_picker_mode(on)
        if on:
            self.statusBar().showMessage(
                "Haz clic en la foto sobre el color que quieras ajustar")

    def on_color_picked(self, color):
        self.pick_btn.setChecked(False)
        h, s, v, _a = color.getHsvF()
        self.current_edits["pc_sample"] = [max(h, 0.0) * 360.0, s, v]
        self._update_pc_swatch()
        if self.current_path:
            self.store.set(self.current_path, self.current_edits)
        self.request_render()
        self.statusBar().showMessage(
            "Muestra tomada — mueve Matiz / Saturación / Luminancia")

    def _update_pc_swatch(self):
        sample = self.current_edits.get("pc_sample")
        if sample:
            c = QColor.fromHsvF(sample[0] / 360.0, sample[1], sample[2])
            self.pc_swatch.setStyleSheet(
                f"background: {c.name()}; border: 1px solid #666; border-radius: 3px;")
            self.pc_swatch.setToolTip(f"Color muestreado: {c.name()}")
            # el deslizador de matiz recorre toda la rueda desde la muestra
            h0 = float(sample[0])
            set_slider_gradient(self.sliders["pc_hue"],
                                [hue_color(h0 + t) for t in range(-180, 181, 45)])
            set_slider_gradient(self.sliders["pc_sat"],
                                [hue_color(h0, s=0.04, v=0.60),
                                 hue_color(h0, s=1.0)])
            set_slider_gradient(self.sliders["pc_lum"],
                                [hue_color(h0, s=0.90, v=0.15),
                                 hue_color(h0, s=0.45, v=1.0)])
        else:
            self.pc_swatch.setStyleSheet(
                "background: #2c2c2c; border: 1px dashed #555; border-radius: 3px;")
            self.pc_swatch.setToolTip("Sin muestra")
            for key in ("pc_hue", "pc_sat", "pc_lum"):
                set_slider_gradient(self.sliders[key], None)

    def reset_point_color(self):
        self.current_edits["pc_sample"] = []
        for key in ("pc_hue", "pc_sat", "pc_lum"):
            self.current_edits[key] = 0.0
        self.current_edits["pc_range"] = 30.0
        self._sync_sliders()
        if self.current_path:
            self.store.set(self.current_path, self.current_edits)
        self.request_render()

    def reset_current_curve(self):
        key = CURVE_CHANNELS[self.channel_combo.currentIndex()][0]
        self.current_edits[key] = [list(p) for p in engine.DEFAULT_CURVE]
        self._sync_curve_widget()
        if self.current_path:
            self.store.set(self.current_path, self.current_edits)
        self.request_render()

    def toggle_original(self, on):
        if not self.current_path:
            return
        e = self.current_edits
        if (e.get("crop") or e.get("rot90") or e.get("straighten")
                or e.get("flip_h") or e.get("flip_v")):
            self.fit_next = True  # el original no esta recortado: reencuadrar
        self.statusBar().showMessage(
            "Mostrando la foto original — pulsa O para volver a tu edición"
            if on else "Mostrando tu edición")
        self.request_render()

    def restore_photo(self):
        if not self.current_path:
            return
        if QMessageBox.question(
                self, APP_NAME,
                "¿Quitar toda la edición de esta foto y volver al original?\n"
                "(ajustes, curvas, color, recorte y corrector)") != QMessageBox.Yes:
            return
        self.reset_edits()
        self.statusBar().showMessage("Foto restaurada al original")

    def reset_edits(self):
        had_strokes = bool(self.current_edits.get("heal_strokes")
                           or self.current_edits.get("erase_ops"))
        self.current_edits = engine.full_edits()
        self._sync_sliders()
        path = self.current_path
        if path:
            self.store.set(path, self.current_edits)
            if had_strokes:
                # la base en cache ya tenia el borrado aplicado: recargar
                self.base_cache.pop(path, None)
                self.render_cache.pop(path, None)
                self.no_faces.discard(path)
                self.ai_cache.pop(path, None)
                self.face_cache.pop(path, None)
                self.healed.discard(path)
                self.erased.discard(path)
                if path not in self.decoding:
                    self.decoding.add(path)
                    self.preview.clear_photo("Cargando foto…")
                    self.fast_pool.start(DecodeJob(path, self))
                return
        self.request_render()

    def copy_edits(self):
        self.copied_edits = engine.full_edits(self.current_edits)
        self.copied_edits["heal_strokes"] = []  # los trazos son de cada foto
        self.copied_edits["erase_ops"] = []     # los borrados IA tambien
        self.statusBar().showMessage("Ajustes copiados")

    def paste_edits(self):
        if not self.copied_edits:
            self.statusBar().showMessage("No hay ajustes copiados")
            return
        self._apply_to_selection(self.copied_edits, "Ajustes pegados")

    def _apply_to_selection(self, edits, message):
        items = self.film.selectedItems()
        if not items:
            return
        for item in items:
            path = item.data(Qt.UserRole)
            merged = engine.full_edits(edits)
            # cada foto conserva sus propios trazos del corrector y borrados
            saved = self.store.get(path)
            merged["heal_strokes"] = saved.get("heal_strokes", [])
            merged["erase_ops"] = saved.get("erase_ops", [])
            self.store.set(path, merged)
            self._update_film_icon(path)  # insignia de edicion al dia
            if path == self.current_path:
                self.current_edits = merged
                self._sync_sliders()
                self.request_render()
        self.statusBar().showMessage(f"{message} en {len(items)} foto(s)")

    # ---------- preajustes ----------

    def refresh_presets(self):
        self.preset_list.clear()
        self.preset_list.addItems(presets.list_presets())

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Guardar preajuste", "Nombre del preajuste:")
        if ok and name.strip():
            presets.save_preset(name.strip(), self.current_edits)
            self.refresh_presets()
            self.statusBar().showMessage(f"Preajuste «{name.strip()}» guardado")

    def apply_preset(self):
        item = self.preset_list.currentItem()
        if not item:
            self.statusBar().showMessage("Elige un preajuste de la lista")
            return
        edits = presets.load_preset(item.text())
        self._apply_to_selection(edits, f"Preajuste «{item.text()}» aplicado")

    def delete_preset(self):
        item = self.preset_list.currentItem()
        if not item:
            return
        if QMessageBox.question(self, "Eliminar", f"¿Eliminar el preajuste «{item.text()}»?") \
                == QMessageBox.Yes:
            presets.delete_preset(item.text())
            self.refresh_presets()

    # ---------- fusion HDR (bracketing) ----------

    def merge_hdr(self):
        """Junta las tomas de un bracketing en una sola foto.

        Con varias fotos seleccionadas en la tira, esas son la tanda. Sin
        seleccion (o con una sola), se buscan las tandas de la carpeta.
        """
        from photoraw.ui.hdr_dialog import HdrDialog

        if not self.folder:
            QMessageBox.information(self, APP_NAME,
                                    "Abre primero una carpeta de fotos.")
            return

        paths = [Path(i.data(Qt.UserRole)) for i in self.film.selectedItems()]
        auto = len(paths) < 2
        if auto:
            groups = self._detect_hdr_groups()
            if not groups:
                QMessageBox.information(
                    self, APP_NAME,
                    "No he encontrado ninguna tanda de bracketing en esta "
                    "carpeta.\n\nUna tanda son varias fotos disparadas "
                    "seguidas con exposiciones distintas. Si sabes cuáles "
                    "son, selecciónalas en la tira (Ctrl+clic) y vuelve a "
                    "pulsar Fusionar HDR.")
                return
        else:
            groups = [paths]

        dlg = HdrDialog(groups, self, auto_detected=auto)
        if dlg.exec() != QDialog.Accepted:
            return
        self._run_hdr_merge(dlg.groups_to_merge, dlg.params)

    def _detect_hdr_groups(self):
        """Busca tandas de bracketing entre las fotos de la tira."""
        files = [Path(self.film.item(i).data(Qt.UserRole))
                 for i in range(self.film.count())]
        # los HDR ya fusionados no son material de partida
        files = [f for f in files if not f.stem.endswith(hdr.SUFFIX)]
        progress = QProgressDialog("Buscando tandas de bracketing…", None,
                                   0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(400)
        try:
            return hdr.detect_groups(
                files, progress_cb=lambda p: (progress.setValue(int(p * 100)),
                                              QApplication.processEvents()))
        except Exception as exc:
            self.statusBar().showMessage(f"No se pudo buscar el bracketing: {exc}")
            return []
        finally:
            progress.close()

    def _run_hdr_merge(self, groups, params):
        """Fusiona las tandas a resolucion completa y las mete en la tira."""
        progress = QProgressDialog("Fusionando…", "Cancelar", 0, len(groups), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        done, reduced, errors, sin_exif = [], [], [], []
        for i, group in enumerate(groups):
            if progress.wasCanceled():
                break
            name = hdr.describe_group(group)
            # se pidio HDR real pero a estas tomas les falta el dato de
            # exposicion: se fusionan con el metodo natural (lo hace fuse_paths)
            if params.get("method") == hdr.HDR and hdr.exposures(group) is None:
                sin_exif.append(name)
            progress.setValue(i)
            progress.setLabelText(f"Fusionando {name}… (puede tardar un rato)")
            QApplication.processEvents()
            try:
                out, was_reduced = self._fuse_full(group, params, progress, name)
                dest = hdr.save(out, hdr.output_path(group, self.folder))
                del out
                done.append(dest)
                if was_reduced:
                    reduced.append(dest.name)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        progress.setValue(len(groups))

        for dest in done:
            self._add_photo_to_film(dest)
        if done:
            self.pool.start(ThumbJob(done, self.signals, self.thumb_cancel))
            item = self._film_item(str(done[-1]))
            if item is not None:
                self.film.setCurrentItem(item)   # abre el HDR recien hecho
            self.statusBar().showMessage(
                f"{len(done)} foto(s) HDR creada(s) en {self.folder}")

        msg = []
        if done:
            msg.append("HDR creado(s):\n" + "\n".join(f"• {d.name}" for d in done))
        if reduced:
            msg.append("A media resolución por falta de memoria:\n"
                       + "\n".join(f"• {n}" for n in reduced))
        if sin_exif:
            msg.append("Fusionadas con el método natural porque no traen los "
                       "datos de exposición en el EXIF:\n"
                       + "\n".join(f"• {n}" for n in sin_exif))
        if errors:
            msg.append("No se pudieron fusionar:\n" + "\n".join(errors))
        if msg:
            QMessageBox.information(self, APP_NAME, "\n\n".join(msg))

    def _fuse_full(self, group, params, progress, name):
        """Fusiona una tanda a resolucion completa.

        Tres RAW de muchos megapixeles a la vez son varios GB de pirámides
        en memoria, asi que si no cabe se reintenta a media resolucion en vez
        de dejar al usuario sin su HDR. Devuelve (imagen, se_redujo).
        """
        def report(p):
            progress.setLabelText(
                f"Leyendo las tomas de {name}… {int(p * 100)} %")
            QApplication.processEvents()

        try:
            return hdr.fuse_paths(group, params, progress_cb=report), False
        except (MemoryError, cv2.error):
            progress.setLabelText(
                f"{name}: no cabe en memoria, probando a media resolución…")
            QApplication.processEvents()
            return hdr.fuse_paths(group, params, half_size=True), True

    def _add_photo_to_film(self, path):
        """Mete una foto nueva en la tira, en su sitio segun el nombre."""
        path = Path(path)
        existing = self._film_item(str(path))
        if existing is not None:
            return existing
        placeholder = QPixmap(150, 150)
        placeholder.fill(Qt.darkGray)
        item = QListWidgetItem(QIcon(placeholder), path.name)
        item.setData(Qt.UserRole, str(path))
        item.setSizeHint(QSize(165, 180))
        row = self.film.count()
        for i in range(self.film.count()):
            other = Path(self.film.item(i).data(Qt.UserRole)).name.lower()
            if other > path.name.lower():
                row = i
                break
        self.film.insertItem(row, item)
        return item

    # ---------- exportar ----------

    def _export_ai_step(self, path, key, work, label, progress, meta=False):
        """Un paso de IA de la exportacion, con memoria en el disco.

        Los modelos trabajan aqui sobre la foto ENTERA (no sobre la vista
        previa reducida), y eso son minutos: ~1 s por cada mosaico de 512 px,
        117 mosaicos en una foto de 24 MP. Guardar el resultado hace que la
        segunda exportacion de la misma foto -- reexportar a otra carpeta,
        cambiar de tamano de salida, repetir el lote -- salga al instante.

        La clave lleva la huella de la imagen de entrada y de los ajustes, asi
        que si tocas el revelado el resultado guardado deja de valer solo."""
        hit = diskcache.load_result(path, key)
        if hit is not None:
            return hit if meta else hit[0]
        progress.setLabelText(label)
        QApplication.processEvents()
        result = work()
        if meta:
            diskcache.save_result(path, key, result[0], meta=result[1])
        else:
            diskcache.save_result(path, key, result)
        return result

    @staticmethod
    def _q8(arr):
        """Redondea a 8 bits, que es como se guarda en el cache. Sin esto la
        huella de los pasos siguientes cambiaria entre la exportacion que
        calcula y la que lee de cache, y ya nunca acertarian."""
        return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(
            np.uint8).astype(np.float32) / 255.0

    @staticmethod
    def _export_progress(progress, label):
        """Callback de porcentaje para los modelos: ademas de mover la barra,
        atiende al boton Cancelar (antes la ventana se quedaba clavada en 0 %
        durante los minutos que tardaba la IA)."""
        def cb(p):
            if progress.wasCanceled():
                raise ai.Cancelled()
            progress.setLabelText(f"{label} {min(int(p * 100), 99)} %")
            QApplication.processEvents()
        return cb

    def export_selected(self):
        items = self.film.selectedItems()
        if not items:
            QMessageBox.information(self, APP_NAME, "Selecciona al menos una foto en la tira.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Carpeta de destino para los JPEG")
        if not out_dir:
            return
        out_dir = Path(out_dir)

        # tamano de salida: original o superresolucion IA (Real-ESRGAN)
        opciones = ["Tamaño original",
                    "2× — superresolución IA",
                    "4× — superresolución IA"]
        eleccion, ok = QInputDialog.getItem(
            self, "Exportar", "Tamaño de exportación:", opciones, 0, False)
        if not ok:
            return
        sr_scale = {opciones[0]: 1, opciones[1]: 2, opciones[2]: 4}[eleccion]
        if sr_scale > 1 and not upscale.model_available():
            if QMessageBox.question(
                    self, APP_NAME,
                    "Hace falta descargar el modelo de superresolución (~64 MB) "
                    "una sola vez.\n¿Descargar ahora?") != QMessageBox.Yes:
                return
            dl = QProgressDialog("Descargando modelo…", None, 0, 100, self)
            dl.setWindowModality(Qt.WindowModal)
            dl.setMinimumDuration(0)
            try:
                upscale.download_model(
                    lambda p: (dl.setValue(int(p * 100)),
                               QApplication.processEvents()))
            except Exception as exc:
                dl.close()
                QMessageBox.warning(self, APP_NAME, f"Falló la descarga: {exc}")
                return
            dl.close()

        progress = QProgressDialog("Exportando…", "Cancelar", 0, len(items), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        import cv2
        done = 0
        for i, item in enumerate(items):
            if progress.wasCanceled():
                break
            path = Path(item.data(Qt.UserRole))
            progress.setLabelText(f"Exportando {path.name}…")
            progress.setValue(i)
            QApplication.processEvents()
            try:
                base = loader.load_full(path)
                edits = engine.full_edits(self.store.get(path))
                strokes = edits.get("heal_strokes")
                if strokes and heal.model_available():
                    def do_heal(b=base, s=strokes):
                        mask = heal.rasterize_strokes(s, *b.shape[:2])
                        return self._q8(heal.inpaint(
                            b, mask,
                            progress_cb=self._export_progress(
                                progress, f"Corrector en {path.name}…")))
                    base = self._export_ai_step(
                        path, "heal2-" + diskcache.result_key(base, strokes),
                        do_heal, f"Corrector en {path.name}…", progress)
                erase_ops = edits.get("erase_ops")
                if erase_ops and generative.model_available():
                    def do_erase(b=base, ops=erase_ops):
                        out = b
                        for j, op in enumerate(ops):
                            hb, wb = out.shape[:2]
                            wmap = generative.decode_map(op["map"], hb, wb)
                            if wmap is None:
                                continue
                            out = generative.erase(
                                out, wmap, seed=int(op.get("seed", 0)),
                                progress_cb=self._export_progress(
                                    progress,
                                    f"Borrado generativo en {path.name}… "
                                    f"({j + 1}/{len(ops)})"))
                        return self._q8(out)
                    base = self._export_ai_step(
                        path,
                        "erase2-" + diskcache.result_key(
                            base, json.dumps(erase_ops, sort_keys=True,
                                             default=str)),
                        do_erase,
                        f"Borrado generativo en {path.name}… (puede tardar)",
                        progress)
                ai_masks = {}
                mask_types = {m.get("type")
                              for m in (edits.get("masks") or [])}
                if (mask_types & {"subject", "background"}
                        and masks_ai.model_available()):
                    ai_masks["subject"] = self._export_ai_step(
                        path, "subj-" + diskcache.result_key(base),
                        lambda b=base: masks_ai.subject_mask(b),
                        f"Máscara IA en {path.name}…", progress)
                if "face_part" in mask_types and face_parse.model_available():
                    ai_masks["face_labels"] = self._export_ai_step(
                        path, "fpl-" + diskcache.result_key(base),
                        lambda b=base: face_parse.parse_labels(b),
                        f"Retrato IA en {path.name}…", progress)
                ai_masks = ai_masks or None
                original = base
                mask_adj = [m.get("adjust") or {}
                            for m in (edits.get("masks") or [])]
                need_den = any(a.get("ai_denoise") for a in mask_adj)
                need_fac = any(a.get("ai_face") for a in mask_adj)
                denoised = faced = None
                amount = edits.get("ai_denoise", 0.0) / 100.0
                if (amount > 0 or need_den) and ai.model_available():
                    denoised = self._export_ai_step(
                        path, "den-" + diskcache.result_key(base),
                        lambda b=base: ai.denoise(
                            b, progress_cb=self._export_progress(
                                progress, f"Ruido IA en {path.name}…")),
                        f"IA en {path.name}… (puede tardar un rato)", progress)
                    if amount > 0:
                        base = base * (1.0 - amount) + denoised * amount
                f_amount = edits.get("ai_face", 0.0) / 100.0
                if (f_amount > 0 or need_fac) and faces.models_available():
                    def do_faces(b=original):
                        arr, n = faces.enhance_faces(
                            b, progress_cb=self._export_progress(
                                progress, f"IA rostros en {path.name}…"))
                        # "no hay caras" tambien se guarda (array de mentira y
                        # meta=0), si no se rebuscarian en cada exportacion
                        return (arr, n) if n else (np.zeros((1, 1, 3), np.uint8), 0)
                    faced, n = self._export_ai_step(
                        path,
                        f"fac-{faces.current_model()}-"
                        + diskcache.result_key(original),
                        do_faces, f"IA rostros en {path.name}…",
                        progress, meta=True)
                    if not n:
                        faced = None
                    elif f_amount > 0:
                        base = np.clip(base + (faced - original) * f_amount, 0.0, 1.0)
                out = engine.apply_edits(base, edits, ai_masks=ai_masks,
                                         denoised=denoised, faced=faced,
                                         is_raw=loader.is_raw(path))
                if sr_scale > 1:
                    progress.setLabelText(
                        f"Superresolución {sr_scale}× en {path.name}…")
                    QApplication.processEvents()
                    out = upscale.upscale(
                        out.astype(np.float32) / 255.0, scale=sr_scale,
                        progress_cb=self._export_progress(
                            progress,
                            f"Superresolución {sr_scale}× en {path.name}…"))
                dest = out_dir / (path.stem + ".jpg")
                cv2.imwrite(str(dest), cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                done += 1
            except ai.Cancelled:
                break    # pulsaste Cancelar mientras trabajaba un modelo
            except Exception as exc:
                self.statusBar().showMessage(f"Error con {path.name}: {exc}")
        progress.setValue(len(items))
        QMessageBox.information(self, APP_NAME,
                                f"Exportadas {done} de {len(items)} fotos a:\n{out_dir}")

    def closeEvent(self, event):
        self.thumb_cancel["stop"] = True
        # Detener la IA al cerrar. Sin esto, Qt espera educadamente a que el
        # trabajo en curso termine antes de acabar el proceso: cerrabas la
        # ventana, desaparecia de la pantalla... y PhotoRAW seguia vivo e
        # invisible quemando la GPU hasta acabar el borrado (asi aparecio un
        # proceso fantasma con 10 minutos de CPU y sin ninguna ventana).
        self.ai_cancel["stop"] = True
        self.pool.clear()        # los trabajos que aun no habian empezado
        self.fast_pool.clear()
        if self.mask_save_timer.isActive():
            self.mask_save_timer.stop()
            self._save_masks()
        if self.store:
            self.store.save()
        super().closeEvent(event)


class StartupSplash(QWidget):
    """Pantalla de carga del arranque.

    Abrir los modelos de IA bloquea el programa unos segundos (el del
    corrector, ~11 s), asi que en vez de enseñar la ventana principal
    congelada se enseña esto: una pantalla que dice en que va. El programa
    aparece cuando ya esta todo listo.

    Va por pasos y no lleva animacion a proposito: mientras un modelo se
    abre no se puede repintar nada, asi que un aro girando se quedaria
    clavado y pareceria colgado. El texto se cambia ANTES de cada paso."""

    ANCHO, ALTO = 460, 210

    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(self.ANCHO, self.ALTO)
        pantalla = QApplication.primaryScreen().geometry()
        self.move(pantalla.center().x() - self.ANCHO // 2,
                  pantalla.center().y() - self.ALTO // 2)

        caja = QVBoxLayout(self)
        caja.setContentsMargins(34, 28, 34, 26)
        caja.setSpacing(6)

        titulo = QLabel(APP_NAME)
        titulo.setStyleSheet("font-size: 30px; font-weight: 600; color: #f0f0f0;")
        caja.addWidget(titulo)
        lema = QLabel("Revelado RAW con IA en tu propia tarjeta gráfica")
        lema.setStyleSheet("color: #8a8a8a;")
        caja.addWidget(lema)
        caja.addStretch(1)

        self.paso_txt = QLabel("Iniciando…")
        self.paso_txt.setStyleSheet("color: #d8d8d8;")
        caja.addWidget(self.paso_txt)
        self.barra = QProgressBar()
        self.barra.setFixedHeight(6)
        self.barra.setTextVisible(False)
        caja.addWidget(self.barra)

    def paso(self, texto, hecho, total):
        self.paso_txt.setText(texto)
        self.barra.setRange(0, total)
        self.barra.setValue(hecho)
        # repintar AHORA: en cuanto empiece el paso, el programa se queda
        # sordo hasta que termine
        QApplication.processEvents()

    def paintEvent(self, _event):
        pt = QPainter(self)
        pt.setRenderHint(QPainter.Antialiasing)
        pt.setPen(QPen(QColor(70, 70, 70), 1))
        pt.setBrush(QColor(30, 30, 30))
        pt.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1),
                           10, 10)
        pt.end()


def _cargar_ia(splash):
    """Mira el equipo y abre los modelos antes de enseñar el programa."""
    splash.paso("Buscando tarjeta gráfica…", 0, 5)
    info = hardware.detectar()
    splash.paso(hardware.resumen(), 1, 5)

    tareas = [("el corrector", heal), ("las máscaras de sujeto", masks_ai),
              ("la superresolución", upscale)]
    tareas = [(n, m) for n, m in tareas if m.model_available()]
    total = len(tareas) + 2
    primera = None
    for i, (nombre, modulo) in enumerate(tareas):
        splash.paso(f"Cargando {nombre}…", i + 1, total)
        try:
            sesion = modulo._get_session()
            primera = primera or sesion
        except Exception:
            pass   # sin esa IA se arranca igual; ya avisara al usarla
    if primera is not None:
        # la prueba de fuego: ver que proveedor uso de verdad un modelo ya
        # cargado (CUDA puede estar listado y luego caer a CPU sin avisar)
        info = hardware.confirmar_con_sesion(primera)
    splash.paso("Abriendo PhotoRAW…", total - 1, total)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(DARK_STYLE)
    splash = StartupSplash()
    splash.show()
    app.processEvents()
    _cargar_ia(splash)
    win = MainWindow()
    win.show()
    splash.close()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
