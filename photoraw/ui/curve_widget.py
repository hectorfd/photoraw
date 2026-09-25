"""Widget interactivo de curva de puntos, estilo Lightroom."""
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtWidgets import QWidget

from photoraw.engine import pchip_lut

MARGIN = 10
HIT_RADIUS = 12


class CurveWidget(QWidget):
    curveChanged = Signal(list)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(230)
        self.setCursor(Qt.CrossCursor)
        self.points = [[0.0, 0.0], [1.0, 1.0]]
        self.color = QColor("#e8e8e8")
        self.drag_index = None
        self.hist = None          # histograma de luminancia (256) de la foto
        self.parametric = None    # curva de los deslizadores, o None
        self.setToolTip("Clic en la curva: añade un punto y arrástralo\n"
                        "Doble clic o clic derecho en un punto: lo quita")

    def set_curve(self, points, color=None):
        self.points = [list(p) for p in points]
        if color is not None:
            self.color = QColor(color)
        self.drag_index = None
        self.update()

    def set_histogram(self, hist):
        """Histograma de luminancia de la foto, de fondo como en Lightroom:
        para ver en que parte de la curva caen los tonos que quieres mover."""
        self.hist = None if hist is None else np.asarray(hist, float)
        self.update()

    def set_parametric(self, lut):
        """Curva que forman los deslizadores de abajo (Iluminaciones, Claros,
        Oscuros, Sombras). Antes no se dibujaba: movias un deslizador y la
        linea seguia recta."""
        self.parametric = None if lut is None else np.asarray(lut, float)
        self.update()

    def _curve_y(self, x):
        return float(np.interp(x, np.linspace(0.0, 1.0, 256),
                               pchip_lut(self.points, n=256)))

    # ---- conversion de coordenadas ----

    def _rect(self):
        w = self.width() - 2 * MARGIN
        h = self.height() - 2 * MARGIN
        return MARGIN, MARGIN, max(w, 1), max(h, 1)

    def _to_widget(self, x, y):
        ox, oy, w, h = self._rect()
        return QPointF(ox + x * w, oy + (1.0 - y) * h)

    def _from_widget(self, pos):
        ox, oy, w, h = self._rect()
        x = (pos.x() - ox) / w
        y = 1.0 - (pos.y() - oy) / h
        return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)

    def _hit_test(self, pos):
        for i, (px, py) in enumerate(self.points):
            wp = self._to_widget(px, py)
            if (wp - pos).manhattanLength() <= HIT_RADIUS:
                return i
        return None

    # ---- dibujo ----

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        ox, oy, w, h = self._rect()

        p.fillRect(self.rect(), QColor("#1b1b1b"))
        if self.hist is not None and self.hist.max() > 0:
            # misma escala que el histograma grande: los picos se cortan
            tope = max(float(np.percentile(self.hist, 99.0)) * 1.2,
                       float(self.hist.max()) * 0.15, 1.0)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 28))
            for i in range(256):
                alto = min(self.hist[i] / tope, 1.0) * h
                if alto >= 1:
                    x0 = ox + w * i / 256
                    p.drawRect(QRectF(x0, oy + h - alto, w / 256 + 0.6, alto))
        grid_pen = QPen(QColor("#333333"))
        p.setPen(grid_pen)
        for i in range(5):
            gx = ox + w * i / 4
            gy = oy + h * i / 4
            p.drawLine(int(gx), oy, int(gx), oy + h)
            p.drawLine(ox, int(gy), ox + w, int(gy))
        p.setPen(QPen(QColor("#444444"), 1, Qt.DashLine))
        p.drawLine(self._to_widget(0, 0), self._to_widget(1, 1))

        xs = np.linspace(0.0, 1.0, 128)
        if self.parametric is not None:
            # la de los deslizadores, discreta, detras de la de puntos
            par = np.interp(xs, np.linspace(0.0, 1.0, len(self.parametric)),
                            self.parametric)
            p.setPen(QPen(QColor(232, 232, 232, 110), 1.5))
            prev = self._to_widget(xs[0], par[0])
            for x, y in zip(xs[1:], par[1:]):
                cur = self._to_widget(x, y)
                p.drawLine(prev, cur)
                prev = cur

        lut = pchip_lut(self.points, n=128)
        p.setPen(QPen(self.color, 2))
        prev = self._to_widget(xs[0], lut[0])
        for x, y in zip(xs[1:], lut[1:]):
            cur = self._to_widget(x, y)
            p.drawLine(prev, cur)
            prev = cur

        p.setBrush(QBrush(QColor("#f0f0f0")))
        p.setPen(QPen(QColor("#1b1b1b"), 1))
        for px, py in self.points:
            p.drawEllipse(self._to_widget(px, py), 5, 5)

    # ---- interaccion ----

    def mousePressEvent(self, event):
        idx = self._hit_test(event.position())
        if event.button() == Qt.RightButton:
            if idx is not None and len(self.points) > 2:
                self.points.pop(idx)
                self.update()
                self.curveChanged.emit([list(p) for p in self.points])
            return
        if event.button() != Qt.LeftButton:
            return
        if idx is None:
            # el punto nuevo nace SOBRE la curva (en su x): antes nacia donde
            # hacias clic y la curva pegaba un salto hasta el. Luego se arrastra
            x, _y = self._from_widget(event.position())
            if any(abs(px - x) < 0.03 for px, _ in self.points):
                return
            y = self._curve_y(x)
            self.points.append([x, y])
            self.points.sort(key=lambda p: p[0])
            idx = next(i for i, p in enumerate(self.points) if p == [x, y])
        self.drag_index = idx
        self.update()

    def mouseDoubleClickEvent(self, event):
        """Doble clic en un punto intermedio: fuera (como en Lightroom)."""
        idx = self._hit_test(event.position())
        if idx is not None and 0 < idx < len(self.points) - 1:
            self.points.pop(idx)
            self.drag_index = None
            self.update()
            self.curveChanged.emit([list(p) for p in self.points])

    def mouseMoveEvent(self, event):
        if self.drag_index is None:
            return
        x, y = self._from_widget(event.position())
        i = self.drag_index
        lo = self.points[i - 1][0] + 0.02 if i > 0 else 0.0
        hi = self.points[i + 1][0] - 0.02 if i < len(self.points) - 1 else 1.0
        self.points[i][0] = min(max(x, lo), hi)
        self.points[i][1] = y
        self.update()
        self.curveChanged.emit([list(p) for p in self.points])

    def mouseReleaseEvent(self, _event):
        if self.drag_index is not None:
            self.drag_index = None
            self.curveChanged.emit([list(p) for p in self.points])


# Zonas del histograma que se pueden arrastrar, como en Lightroom: cada trozo
# del eje de tonos mueve el deslizador que trabaja sobre esos tonos
HIST_ZONES = (
    ("blacks",     "Negros",     0.00, 0.13),
    ("shadows",    "Sombras",    0.13, 0.35),
    ("exposure",   "Exposición", 0.35, 0.65),
    ("highlights", "Luces",      0.65, 0.87),
    ("whites",     "Blancos",    0.87, 1.00),
)


class HistogramWidget(QWidget):
    """Widget de histograma RGB + luminancia, estilo Lightroom.

    Ademas de mirar, se puede tocar: al pasar el raton se ilumina la zona
    (Negros, Sombras, Exposicion, Luces, Blancos) y arrastrando a izquierda
    o derecha se mueve ese deslizador. Doble clic la devuelve a cero."""

    MARGIN = 4

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(80)
        self.setMaximumHeight(100)
        self.setMouseTracking(True)
        self._hist_r = None
        self._hist_g = None
        self._hist_b = None
        self._hist_lum = None
        self._sliders = {}        # clave -> (QSlider, etiqueta de valor)
        self._hover = None        # indice de zona bajo el raton
        self._drag = None         # (indice, x inicial, valor inicial)

    def bind_sliders(self, sliders, labels):
        """Los deslizadores de Ajustes que mueve cada zona. Se mueven los de
        verdad (setValue), asi el guardado y el revelado van por el mismo
        camino que si los tocaras a mano."""
        self._sliders = {k: (sliders[k], labels.get(k))
                         for k, *_ in HIST_ZONES if k in sliders}

    # ---- interaccion ----

    def _zone_at(self, x):
        w = self.width() - 2 * self.MARGIN
        t = (x - self.MARGIN) / max(w, 1)
        for i, (_k, _n, a, b) in enumerate(HIST_ZONES):
            if t < b or i == len(HIST_ZONES) - 1:
                return i
        return None

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        i = self._zone_at(event.position().x())
        entry = self._sliders.get(HIST_ZONES[i][0])
        if entry is None:
            return
        self._drag = (i, event.position().x(), entry[0].value())
        self.setCursor(Qt.SizeHorCursor)
        self.update()

    def mouseMoveEvent(self, event):
        x = event.position().x()
        if self._drag is None:
            i = self._zone_at(x)
            if i != self._hover:
                self._hover = i
                self.setCursor(Qt.SizeHorCursor if i is not None
                               else Qt.ArrowCursor)
                self.update()
            return
        i, x0, v0 = self._drag
        slider = self._sliders[HIST_ZONES[i][0]][0]
        # recorrer el ancho entero del histograma = medio recorrido del
        # deslizador: rapido para lo grueso y aun fino pixel a pixel
        span = slider.maximum() - slider.minimum()
        w = max(self.width() - 2 * self.MARGIN, 1)
        nuevo = int(round(v0 + (x - x0) / w * span * 0.5))
        slider.setValue(min(max(nuevo, slider.minimum()), slider.maximum()))
        self.update()

    def mouseReleaseEvent(self, _event):
        if self._drag is not None:
            self._drag = None
            self.update()

    def mouseDoubleClickEvent(self, event):
        """Doble clic en una zona: su ajuste vuelve a cero."""
        i = self._zone_at(event.position().x())
        entry = self._sliders.get(HIST_ZONES[i][0])
        if entry is not None:
            slider = entry[0]
            slider.setValue(getattr(slider, "default_value", 0))
        self._drag = None
        self.update()

    def leaveEvent(self, _event):
        if self._drag is None:
            self._hover = None
            self.update()

    def set_image(self, img):
        """Calcula el histograma de la foto revelada: uint8 (lo que devuelve
        el motor) o float 0..1.

        Antes daba por hecho float 0..1 y le llega uint8: multiplicar un
        uint8 por 255 da la vuelta (10 -> 246, 200 -> 56), asi que el
        histograma salia AL REVES, las sombras a la derecha."""
        if img is None or img.size == 0:
            self._hist_r = self._hist_g = self._hist_b = self._hist_lum = None
            self.update()
            return

        flat = img.reshape(-1, 3)
        if flat.dtype != np.uint8:
            flat = (np.clip(flat, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        cuenta = lambda v: np.bincount(v, minlength=256)[:256]
        self._hist_r = cuenta(flat[:, 0])
        self._hist_g = cuenta(flat[:, 1])
        self._hist_b = cuenta(flat[:, 2])
        lum = flat @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        self._hist_lum = cuenta(np.clip(lum + 0.5, 0, 255).astype(np.uint8))
        self.update()

    @property
    def lum_hist(self):
        return self._hist_lum
    
    def paintEvent(self, _event):
        pt = QPainter(self)
        pt.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        margin = 4
        
        # Fondo oscuro siempre visible
        pt.fillRect(0, 0, w, h, QColor(30, 30, 30))
        
        if self._hist_r is None:
            pt.setPen(QPen(QColor(80, 80, 80)))
            pt.setBrush(Qt.NoBrush)
            pt.drawRect(0, 0, w - 1, h - 1)
            pt.end()
            return
        
        # Escala: el pico mas alto aplastaba todo lo demas (un cielo liso o
        # un fondo negro son una sola barra altisima). Se escala a un pico
        # "tipico" y lo que lo pase se corta arriba, como en Lightroom
        todos = np.concatenate([self._hist_r, self._hist_g, self._hist_b])
        max_val = max(float(np.percentile(todos, 99.0)) * 1.2,
                      float(todos.max()) * 0.15, 1.0)
        if todos.max() == 0:
            pt.end()
            return
        
        def draw_hist(hist, color):
            pt.setPen(Qt.NoPen)
            pt.setBrush(QColor(color[0], color[1], color[2], 100))
            
            for i in range(256):
                x = margin + int(i * (w - 2 * margin) / 256)
                x_next = margin + int((i + 1) * (w - 2 * margin) / 256)
                bar_w = max(x_next - x, 1)
                bar_h = int(min(hist[i] / max_val, 1.0) * (h - 2 * margin))
                if bar_h > 0:
                    pt.drawRect(x, h - margin - bar_h, bar_w, bar_h)
        
        # Dibujar canales (RGB + luminancia)
        draw_hist(self._hist_lum, (200, 200, 200))  # Luminancia primero (fondo)
        draw_hist(self._hist_r, (220, 80, 80))  # Rojo
        draw_hist(self._hist_g, (80, 180, 80))  # Verde
        draw_hist(self._hist_b, (80, 120, 220)) # Azul

        self._paint_zone(pt, w, h)

        # Borde sutil
        pt.setPen(QPen(QColor(60, 60, 60)))
        pt.setBrush(Qt.NoBrush)
        pt.drawRect(0, 0, w - 1, h - 1)

        pt.end()

    def _paint_zone(self, pt, w, h):
        """Zona bajo el raton (o la que arrastras): velo claro + su nombre y
        valor abajo a la izquierda, como en Lightroom."""
        i = self._drag[0] if self._drag is not None else self._hover
        if i is None:
            return
        key, name, a, b = HIST_ZONES[i]
        m = self.MARGIN
        iw = w - 2 * m
        x0 = m + a * iw
        pt.setPen(Qt.NoPen)
        pt.setBrush(QColor(255, 255, 255, 34 if self._drag else 22))
        pt.drawRect(QRectF(x0, m, (b - a) * iw, h - 2 * m))

        entry = self._sliders.get(key)
        texto = name
        if entry is not None and entry[1] is not None:
            val = entry[1].text()
            if val and not val.startswith("-") and val not in ("0",):
                val = "+" + val
            texto = f"{name}  {val}"
        f = pt.font()
        f.setPointSizeF(8.5)
        f.setBold(True)
        pt.setFont(f)
        fm = pt.fontMetrics()
        tw = fm.horizontalAdvance(texto) + 10
        th = fm.height() + 4
        caja = QRectF(m + 3, h - m - th - 3, tw, th)
        pt.setBrush(QColor(0, 0, 0, 150))
        pt.drawRoundedRect(caja, 3, 3)
        pt.setPen(QColor(235, 235, 235))
        pt.drawText(caja, Qt.AlignCenter, texto)
