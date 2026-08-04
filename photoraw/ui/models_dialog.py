"""Ventana "Modelos de IA": que IAs tiene el proyecto, cuales estan
descargadas y botones para bajarlas o borrarlas una a una.

La idea: el repositorio no lleva los ~3,3 GB de modelos. Se clona, se
instalan los requirements y desde aqui cada uno baja solo las IAs que vaya
a usar. La ventana se dibuja a partir de photoraw.models.AI_MODELS, asi que
no hay que tocarla al anadir una IA nueva.
"""
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel,
                               QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from photoraw import models

VERDE = "#5fbf6a"
GRIS = "#8a8a8a"
AMBAR = "#d8a13a"


def _mb(n_bytes):
    gb = n_bytes / 1_000_000_000
    return f"{gb:.1f} GB" if gb >= 1 else f"{n_bytes / 1_000_000:.0f} MB"


class _Signals(QObject):
    progress = Signal(str, float)   # clave del modelo, 0..1
    done = Signal(str, str)         # clave, mensaje de error ("" si fue bien)


class _DownloadJob(QRunnable):
    """Baja un modelo en segundo plano para no congelar la ventana."""

    def __init__(self, model, signals):
        super().__init__()
        self.model = model
        self.signals = signals

    def run(self):
        try:
            self.model.download(
                lambda p: self.signals.progress.emit(self.model.key, p))
            error = ""
        except Exception as exc:
            error = str(exc)
        self.signals.done.emit(self.model.key, error)


class ModelsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Modelos de IA")
        self.resize(780, 620)
        self.pool = QThreadPool.globalInstance()
        self.signals = _Signals()
        self.signals.progress.connect(self._on_progress)
        self.signals.done.connect(self._on_done)
        self.rows = {}          # clave -> dict de widgets
        self.descargando = set()

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(18, 16, 18, 14)
        raiz.setSpacing(10)

        self.titulo = QLabel()
        self.titulo.setStyleSheet("font-size: 15px; font-weight: 600;")
        raiz.addWidget(self.titulo)

        pie_texto = QLabel(
            "Las IAs no vienen con el programa: se descargan una sola vez y "
            "quedan en <code>~/.photoraw/models</code>. Puedes bajar solo las "
            "que uses y borrar las que no.")
        pie_texto.setWordWrap(True)
        pie_texto.setStyleSheet(f"color: {GRIS};")
        raiz.addWidget(pie_texto)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        # sin barra horizontal: las descripciones se parten en varias lineas
        # en vez de estirar la fila y dejar los botones fuera de la ventana
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        contenedor = QWidget()
        self.lista = QVBoxLayout(contenedor)
        self.lista.setContentsMargins(0, 4, 6, 4)
        self.lista.setSpacing(8)
        for modelo in models.AI_MODELS:
            self.lista.addWidget(self._fila(modelo))
        self.lista.addStretch(1)
        scroll.setWidget(contenedor)
        raiz.addWidget(scroll, 1)

        pie = QHBoxLayout()
        self.btn_faltan = QPushButton("Descargar los que faltan")
        self.btn_faltan.clicked.connect(self._descargar_faltantes)
        pie.addWidget(self.btn_faltan)
        pie.addStretch(1)
        cerrar = QPushButton("Cerrar")
        cerrar.clicked.connect(self.accept)
        pie.addWidget(cerrar)
        raiz.addLayout(pie)

        self.refrescar()

    # ---------- construccion ----------

    def _fila(self, modelo):
        marco = QFrame()
        marco.setFrameShape(QFrame.StyledPanel)
        marco.setStyleSheet(
            "QFrame { background: #2b2b2b; border: 1px solid #3a3a3a;"
            " border-radius: 6px; }")
        caja = QHBoxLayout(marco)
        caja.setContentsMargins(12, 10, 12, 10)
        caja.setSpacing(12)

        izq = QVBoxLayout()
        izq.setSpacing(2)
        # todas las etiquetas parten linea y pueden encoger: si no, el texto
        # mas largo fija el ancho minimo de la fila y los botones de la
        # derecha se salen de la ventana
        cabecera = QLabel(f"<b>{modelo.name}</b> "
                          f"<span style='color:{GRIS};'>· {modelo.tool}</span>")
        cabecera.setWordWrap(True)
        cabecera.setMinimumWidth(1)
        izq.addWidget(cabecera)
        desc = QLabel(modelo.what)
        desc.setWordWrap(True)
        desc.setMinimumWidth(1)   # que pueda encoger: manda el ancho de la
                                  # ventana, no el largo de la descripcion
        desc.setStyleSheet(f"color: {GRIS}; border: none;")
        izq.addWidget(desc)
        if modelo.needed_by:
            aviso = QLabel("Lo necesitan: " + ", ".join(modelo.needed_by))
            aviso.setWordWrap(True)
            aviso.setMinimumWidth(1)
            aviso.setStyleSheet(f"color: {AMBAR}; border: none;")
            izq.addWidget(aviso)
        caja.addLayout(izq, 1)

        der = QVBoxLayout()
        der.setSpacing(4)
        estado = QLabel()
        estado.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        estado.setStyleSheet("border: none;")
        der.addWidget(estado)
        barra = QProgressBar()
        barra.setFixedWidth(180)
        barra.setTextVisible(True)
        barra.hide()
        der.addWidget(barra)
        boton = QPushButton()
        boton.setFixedWidth(180)
        boton.clicked.connect(lambda _=False, m=modelo: self._pulsado(m))
        der.addWidget(boton)
        caja.addLayout(der)

        marco.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.rows[modelo.key] = {"estado": estado, "boton": boton,
                                 "barra": barra, "modelo": modelo}
        return marco

    # ---------- estado ----------

    def refrescar(self):
        hechos, total, en_disco = models.summary()
        self.titulo.setText(f"{hechos} de {total} modelos instalados · "
                            f"{_mb(en_disco)} en disco")
        for clave, w in self.rows.items():
            modelo = w["modelo"]
            if clave in self.descargando:
                continue
            w["barra"].hide()
            w["boton"].setEnabled(True)
            if modelo.installed():
                w["estado"].setText(
                    f"<span style='color:{VERDE};'>● Instalado</span> "
                    f"<span style='color:{GRIS};'>{_mb(modelo.disk_bytes())}</span>")
                w["boton"].setText("Eliminar")
            else:
                w["estado"].setText(f"<span style='color:{GRIS};'>○ Falta</span>")
                w["boton"].setText(f"Descargar ({modelo.size_mb} MB)")
        faltan = models.missing()
        self.btn_faltan.setEnabled(bool(faltan) and not self.descargando)
        self.btn_faltan.setText(
            "Todo descargado" if not faltan
            else f"Descargar los {len(faltan)} que faltan")

    # ---------- acciones ----------

    def _pulsado(self, modelo):
        if modelo.installed():
            self._eliminar(modelo)
        else:
            self._descargar(modelo)

    def _eliminar(self, modelo):
        aviso = (f"¿Borrar {modelo.name} del disco?\n\n"
                 f"Se liberan {_mb(modelo.disk_bytes())}. La herramienta "
                 f"«{modelo.tool}» dejará de funcionar hasta que lo vuelvas "
                 f"a descargar desde aquí.")
        if modelo.needed_by:
            aviso += "\n\nOJO: también lo necesitan " + \
                     ", ".join(modelo.needed_by) + "."
        if QMessageBox.question(self, "Modelos de IA", aviso) != QMessageBox.Yes:
            return
        try:
            modelo.remove()
        except OSError as exc:
            QMessageBox.warning(
                self, "Modelos de IA",
                f"No se pudo borrar (¿lo está usando la app?):\n{exc}")
        self.refrescar()

    def _descargar(self, modelo):
        if modelo.key in self.descargando:
            return
        self.descargando.add(modelo.key)
        w = self.rows[modelo.key]
        w["boton"].setEnabled(False)
        w["boton"].setText("Descargando…")
        w["estado"].setText(f"<span style='color:{AMBAR};'>● Descargando</span>")
        w["barra"].setRange(0, 100)
        w["barra"].setValue(0)
        w["barra"].show()
        self.btn_faltan.setEnabled(False)
        self.pool.start(_DownloadJob(modelo, self.signals))

    def _descargar_faltantes(self):
        for modelo in models.missing():
            self._descargar(modelo)

    def _on_progress(self, clave, fraccion):
        w = self.rows.get(clave)
        if w:
            w["barra"].setValue(int(fraccion * 100))

    def _on_done(self, clave, error):
        self.descargando.discard(clave)
        if error:
            QMessageBox.warning(
                self, "Modelos de IA",
                f"No se pudo descargar {self.rows[clave]['modelo'].name}:\n{error}")
        self.refrescar()
