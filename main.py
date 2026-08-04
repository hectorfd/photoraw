import sys


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
