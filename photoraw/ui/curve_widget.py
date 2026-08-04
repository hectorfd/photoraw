"""Widget interactivo de curva de puntos, estilo Lightroom."""
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF
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

    def set_curve(self, points, color=None):
        self.points = [list(p) for p in points]
        if color is not None:
            self.color = QColor(color)
        self.drag_index = None
        self.update()

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
        grid_pen = QPen(QColor("#333333"))
        p.setPen(grid_pen)
        for i in range(5):
            gx = ox + w * i / 4
            gy = oy + h * i / 4
            p.drawLine(int(gx), oy, int(gx), oy + h)
            p.drawLine(ox, int(gy), ox + w, int(gy))
        p.setPen(QPen(QColor("#444444"), 1, Qt.DashLine))
        p.drawLine(self._to_widget(0, 0), self._to_widget(1, 1))

        lut = pchip_lut(self.points, n=128)
        xs = np.linspace(0.0, 1.0, 128)
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
            x, y = self._from_widget(event.position())
            self.points.append([x, y])
            self.points.sort(key=lambda p: p[0])
            idx = next(i for i, p in enumerate(self.points) if p == [x, y])
        self.drag_index = idx
        self.update()

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


class HistogramWidget(QWidget):
    """Widget de histograma RGB + luminancia, estilo Lightroom."""
    
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(80)
        self.setMaximumHeight(100)
        self._hist_r = None
        self._hist_g = None
        self._hist_b = None
        self._hist_lum = None
    
    def set_image(self, img):
        """Calcula histograma de una imagen float32 RGB 0..1."""
        if img is None or img.size == 0:
            self._hist_r = self._hist_g = self._hist_b = self._hist_lum = None
            self.update()
            return
        
        # Calcular histogramas (256 bins)
        flat = img.reshape(-1, 3)
        self._hist_r = np.histogram((flat[:, 0] * 255).astype(np.uint8), 
                                     bins=256, range=(0, 256))[0]
        self._hist_g = np.histogram((flat[:, 1] * 255).astype(np.uint8), 
                                     bins=256, range=(0, 256))[0]
        self._hist_b = np.histogram((flat[:, 2] * 255).astype(np.uint8), 
                                     bins=256, range=(0, 256))[0]
        
        # Luminancia
        lum = flat @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        self._hist_lum = np.histogram((lum * 255).astype(np.uint8), 
                                       bins=256, range=(0, 256))[0]
        
        self.update()
    
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
        
        # Normalizar histogramas al maximo global
        max_val = max(self._hist_r.max(), self._hist_g.max(), 
                      self._hist_b.max(), self._hist_lum.max())
        if max_val == 0:
            pt.end()
            return
        
        def draw_hist(hist, color):
            pt.setPen(Qt.NoPen)
            pt.setBrush(QColor(color[0], color[1], color[2], 100))
            
            for i in range(256):
                x = margin + int(i * (w - 2 * margin) / 256)
                x_next = margin + int((i + 1) * (w - 2 * margin) / 256)
                bar_w = max(x_next - x, 1)
                bar_h = int(hist[i] / max_val * (h - 2 * margin))
                if bar_h > 0:
                    pt.drawRect(x, h - margin - bar_h, bar_w, bar_h)
        
        # Dibujar canales (RGB + luminancia)
        draw_hist(self._hist_lum, (200, 200, 200))  # Luminancia primero (fondo)
        draw_hist(self._hist_r, (220, 80, 80))  # Rojo
        draw_hist(self._hist_g, (80, 180, 80))  # Verde
        draw_hist(self._hist_b, (80, 120, 220)) # Azul
        
        # Borde sutil
        pt.setPen(QPen(QColor(60, 60, 60)))
        pt.setBrush(Qt.NoBrush)
        pt.drawRect(0, 0, w - 1, h - 1)
        
        pt.end()
