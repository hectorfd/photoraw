"""Que tarjeta grafica hay y hasta donde puede llegar PhotoRAW en este PC.

PhotoRAW se disena para una NVIDIA con CUDA. En un PC sin ella la mayoria
de las IAs seguirian "funcionando" por CPU, pero tardando de minutos a
horas — y una herramienta que parece disponible y luego cuelga el programa
media hora es peor que una herramienta desactivada. Por eso al arrancar se
mira el equipo y se elige perfil:

  completo  — hay CUDA: todas las herramientas disponibles.
  limitado  — no hay: se apagan las IAs que solo son razonables en GPU,
              y el revelado (que va por CPU) funciona igual de bien.
"""
import subprocess

COMPLETO = "completo"
LIMITADO = "limitado"

_info = None


def _nvidia_smi():
    """(nombre, VRAM en MB) de la primera tarjeta NVIDIA, o (None, 0)."""
    try:
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        primera = salida.stdout.strip().splitlines()[0]
        nombre, vram = primera.split(",")
        return nombre.strip(), int(vram)
    except Exception:
        return None, 0


def detectar(forzar=False):
    """Mira el equipo una sola vez y cachea el resultado."""
    global _info
    if _info is not None and not forzar:
        return _info

    try:
        import onnxruntime as ort
        proveedores = list(ort.get_available_providers())
    except Exception:
        proveedores = []
    cuda_disponible = "CUDAExecutionProvider" in proveedores
    nombre, vram = _nvidia_smi()

    if cuda_disponible and nombre:
        motivo = f"{nombre} con {vram / 1024:.0f} GB"
    elif cuda_disponible:
        motivo = "CUDA disponible (no se pudo leer el modelo de tarjeta)"
    elif nombre:
        # hay NVIDIA pero onnxruntime no la ve: casi siempre es que esta
        # instalado el paquete de CPU, o faltan las librerias de CUDA
        motivo = (f"{nombre} detectada, pero onnxruntime no puede usarla "
                  "(instala onnxruntime-gpu y las librerías CUDA)")
    else:
        motivo = "sin tarjeta NVIDIA compatible con CUDA"

    _info = {
        "perfil": COMPLETO if cuda_disponible else LIMITADO,
        "cuda": cuda_disponible,
        "gpu": nombre,
        "vram_mb": vram,
        "motivo": motivo,
        "proveedores": proveedores,
    }
    return _info


def confirmar_con_sesion(sesion):
    """Ajusta el perfil con la verdad de verdad: que proveedor acabo usando
    un modelo ya cargado. CUDA puede aparecer listado y luego fallar al
    crear la sesion (DLL que faltan), y entonces ONNX cae a CPU sin avisar."""
    info = detectar()
    try:
        real = sesion.get_providers()[0]
    except Exception:
        return info
    if real == "CPUExecutionProvider" and info["perfil"] == COMPLETO:
        info["perfil"] = LIMITADO
        info["cuda"] = False
        info["motivo"] = ("CUDA falló al arrancar y los modelos están "
                          "cayendo a CPU")
    return info


def perfil():
    return detectar()["perfil"]


def hay_cuda():
    return detectar()["cuda"]


def resumen():
    """Una linea para la barra de estado / la ventana de modelos."""
    info = detectar()
    if info["perfil"] == COMPLETO:
        return f"Perfil completo · {info['motivo']}"
    return f"Perfil limitado · {info['motivo']}"
