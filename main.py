import os
import sys

# ANTES de importar numpy. La OpenBLAS que trae numpy (0.3.33) revienta con
# un acceso de memoria (0xC0000005) si dos hilos multiplican matrices a la
# vez con su reparto interno en 22 hilos: pasaba al mover la curva, cuando
# el revelado, la vista de detalle y el histograma calculan a la vez, y la
# app se cerraba sin aviso. Reproducido 3/3 en 15 s; con 1 hilo, 0/3 en 20 s
# (y mas rapido: las matrices de PhotoRAW son de 3x3, repartirlas no compensa)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _show_error(text):
    """Muestra el error en una ventana de Windows (no hay consola visible)."""
    import ctypes
    ctypes.windll.user32.MessageBoxW(0, text, "PhotoRAW - Error", 0x10)


if __name__ == "__main__":
    try:
        from photoraw.ui.main_window import main
    except ImportError as exc:
        _show_error(
            "Faltan librerías por instalar (quizá la instalación aún no termina).\n\n"
            f"Detalle: {exc}"
        )
        sys.exit(1)
    try:
        main()
    except Exception as exc:
        import traceback
        _show_error("La aplicación tuvo un error:\n\n" + traceback.format_exc())
        sys.exit(1)
