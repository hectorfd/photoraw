"""Motor de procesamiento de imagen: aplica los ajustes sobre un array RGB."""
import colorsys
import copy
import math

import numpy as np
import cv2


def auto_levels(img, shadow_clip=0.5, highlight_clip=0.5):
    """Auto-niveles profesional basado en histograma.
    
    Analiza la distribucion tonal y ajusta automaticamente los puntos
    de negro y blanco para maximizar el rango dinamico.
    
    Args:
        img: float32 RGB 0..1
        shadow_clip: porcentaje de sombras a recortar (0.5 = 0.5% mas oscuro -> negro)
        highlight_clip: porcentaje de altas luces a recortar (0.5 = 0.5% mas claro -> blanco)
    
    Returns:
        float32 RGB 0..1 con rango optimizado
    """
    flat = img.reshape(-1, 3).mean(axis=1)  # luminancia promedio
    
    # Calcular puntos de corte basados en percentiles
    black_point = np.percentile(flat, shadow_clip) 
    white_point = np.percentile(flat, 100.0 - highlight_clip)
    
    # Seguridad: evitar division por cero
    if white_point - black_point < 0.05:
        return img
    
    # Estirar al rango completo
    img = (img - black_point) / (white_point - black_point)
    return np.clip(img, 0.0, 1.0)


def _luminance(img):
    """Luminancia BT.709."""
    return (img * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(axis=-1)


def auto_exposure(img, target=0.42):
    """Calcula el EV para que la mediana de luminancia llegue a `target`.

    `img` ya viene con la curva de gama aplicada (igual que se ve en
    pantalla), asi que el objetivo tiene que estar en ese mismo espacio:
    0.42 es el equivalente perceptual al 18% gris fotografico, NO 0.18
    directo (eso asumiria luz lineal y da resultados oscuros/erroneos).
    Usa la mediana en vez de la media geometrica: no se deja enganar por
    fotos con mucho cielo o mucha sombra en el encuadre.

    Returns:
        float EV optimo, rango -3.0 .. +3.0
    """
    lum = _luminance(img)
    med = float(np.median(lum))
    if med <= 1e-6:
        return 0.0
    ev = float(np.clip(np.log2(target / med), -3.0, 3.0))
    return round(ev, 2)


def smart_exposure(img, ev):
    """Exposicion inteligente: protege highlights al subir, sombras al bajar.
    
    Rapido: calcula luminancia a 1/8 res, genera mapa de factor, lo aplica.
    
    Args:
        img: float32 RGB 0..1
        ev: valor EV (-3.0 .. +3.0)
    
    Returns:
        float32 RGB 0..1
    """
    if ev == 0.0:
        return img
    
    h, w = img.shape[:2]
    sf = 2.0 ** ev
    sw, sh = max(w // 8, 1), max(h // 8, 1)
    
    # Luminancia a baja res (muy rapido)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    lum = small[..., 0] * 0.2126 + small[..., 1] * 0.7152 + small[..., 2] * 0.0722
    
    # Mapa de factor: 0.4 (protege) a 1.0 (ajusta completo)
    if ev > 0:
        mask = np.clip(1.0 - lum * 1.667, 0.0, 1.0)  # 1/0.6 = 1.667
    else:
        mask = np.clip(lum * 2.5, 0.0, 1.0)  # 1/0.4 = 2.5
    mask *= mask
    factor_small = (0.4 + 0.6 * mask).astype(np.float32)
    
    # Escalar factor y aplicar
    factor = cv2.resize(factor_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(img * (1.0 + (sf - 1.0) * factor)[..., None], 0.0, 1.0)


def auto_tone(img, target=0.42):
    """Correccion automatica de exposicion al abrir un RAW (como hacen
    Lightroom / Windows Fotos): solo entra cuando la foto esta realmente
    oscura (contraluces, interiores), y no toca las que ya estan bien
    expuestas. Mide la mediana de luminancia (en espacio gamma, igual que
    ve el ojo) y mezcla mapeo tonal adaptativo en proporcion a cuanto le
    falta a esa mediana para llegar a `target`.

    Args:
        img: float32 RGB 0..1 (salida del revelado RAW, ya con gamma)
        target: mediana de luminancia deseada para una foto "bien expuesta"
    Returns:
        float32 RGB 0..1
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(w // 8, 1), max(h // 8, 1)), interpolation=cv2.INTER_AREA)
    med = float(np.median(_luminance(small)))
    if med >= target:
        return img
    # `float(...)` a proposito: np.clip devuelve un np.float64 y multiplicar
    # la foto (float32) por un escalar float64 la asciende ENTERA a float64.
    # Asi salia de aqui, y el resto del revelado seguia en float64: el doble
    # de memoria y unas 3,5 veces mas lento por operacion, sin ninguna
    # ganancia de calidad visible (la foto acaba en 8 bits por canal)
    deficit = float(np.clip((target - med) / target, 0.0, 1.0))
    weight = deficit ** 1.5
    if weight < 0.01:
        return img
    mapped = tone_mapping(img, key=0.35)
    mapped *= weight                    # mapped es nuestro: se puede pisar
    mapped += img * (1.0 - weight)
    return mapped


def tone_mapping(img, key=0.18, saturation=1.0):
    """Mapeo tonal adaptativo (Reinhard modificado).
    
    Convierte datos lineales del RAW a una imagen con apariencia natural,
    preservando contraste local mientras comprime el rango dinamico.
    
    Args:
        img: float32 RGB 0..1 (lineal del RAW)
        key: punto medio del mapeo (0.18 = 18% gris, estandar fotografico)
        saturation: control de saturacion (1.0 = normal, >1 = mas saturado)
    
    Returns:
        float32 RGB 0..1 con tonos mapeados
    """
    # Calcular luminancia. Se deja tal cual (una copia de la foto entera y
    # luego sumar el eje del color) a proposito: hacerlo con _luminance suma
    # los tres canales en otro orden y en float32 eso mueve el ultimo bit —
    # movia 1/255 en un pixel de cada 100.000, invisible pero distinto
    lum = img * np.array([0.2126, 0.7152, 0.0722], np.float32)
    lum = lum.sum(axis=-1)

    # Luminancia promedio geometrico (mas preciso que aritmetico)
    log_lum = np.log(lum + 1e-6)
    log_avg = np.exp(log_lum.mean())
    del lum, log_lum

    # Factor de escala basado en el key value
    scale = key / (log_avg + 1e-6)

    # Aplicar escala
    img_scaled = img * scale

    # Mapeo tonal: division con compresion suave
    # Formula: L_out = L_in / (1 + L_in) - comparte highlights
    den = img_scaled + 1.0
    img_mapped = np.divide(img_scaled, den, out=img_scaled)
    del den
    
    # Control de saturacion: mezclar con luminancia
    if saturation != 1.0:
        lum_new = img_mapped * np.array([0.2126, 0.7152, 0.0722], np.float32)
        lum_new = lum_new.sum(axis=-1, keepdims=True)
        img_mapped = lum_new + (img_mapped - lum_new) * saturation
    
    return np.clip(img_mapped, 0.0, 1.0, out=img_mapped)


def adaptive_contrast(img, clip_limit=2.0, grid_size=8):
    """Contraste adaptativo local (CLAHE - Contrast Limited Adaptive Histogram Equalization).
    
    Mejora el contraste local preservando el contraste global.
    Util para revelar detalle en sombras e iluminaciones sin lavar la imagen.
    
    Args:
        img: float32 RGB 0..1
        clip_limit: limite de contraste (2.0 = moderado, 4.0 = fuerte)
        grid_size: tamano de la grilla local (8 = 8x8 bloques)
    
    Returns:
        float32 RGB 0..1 con contraste mejorado
    """
    # Convertir a LAB (luminancia + crominancia)
    lab = cv2.cvtColor(np.clip(img, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2LAB)
    
    # Aplicar CLAHE solo al canal de luminancia (L)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    lab[..., 0] = clahe.apply((lab[..., 0] * 255).astype(np.uint8)).astype(np.float32) / 255.0
    
    # Volver a RGB
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


DEFAULT_CURVE = [[0.0, 0.0], [1.0, 1.0]]

DEFAULT_EDITS = {
    "exposure": 0.0,     # EV, -3.0 .. +3.0
    "contrast": 0.0,     # -100 .. 100
    "highlights": 0.0,   # -100 .. 100
    "shadows": 0.0,      # -100 .. 100
    "whites": 0.0,       # -100 .. 100
    "blacks": 0.0,       # -100 .. 100
    "temperature": 0.0,  # -100 (frio) .. 100 (calido)
    # Mapeo tonal adaptativo (Reinhard modificado)
    "tone_map": 0.0,     # 0..100 (0 = desactivado, 100 = maximo efecto)
    # Contraste adaptativo local (CLAHE)
    "adaptive_contrast": 0.0,  # 0..100 (0 = desactivado, 100 = maximo efecto)
    "tint": 0.0,         # -100 (verde) .. 100 (magenta)
    "saturation": 0.0,   # -100 .. 100
    "vibrance": 0.0,     # -100 .. 100
    "texture": 0.0,      # -100 .. 100
    "clarity": 0.0,      # -100 .. 100 (contraste local en medios tonos)
    "dehaze": 0.0,       # -100 .. 100 (borrar neblina)
    # Enfoque (panel Detalle)
    "sharp_amount": 0.0,   # 0 .. 150
    "sharp_radius": 1.0,   # 0.5 .. 3.0
    "sharp_detail": 25.0,  # 0 .. 100
    "sharp_masking": 0.0,  # 0 .. 100
    # Reduccion de ruido manual
    "nr_luminance": 0.0,   # 0 .. 100
    "nr_color": 0.0,       # 0 .. 100
    # Reduccion de ruido IA (intensidad de mezcla; se aplica fuera del motor)
    "ai_denoise": 0.0,     # 0 .. 100
    # Retoque de rostros IA (intensidad de mezcla; se aplica fuera del motor)
    "ai_face": 0.0,        # 0 .. 100
    # Grano de pelicula
    "grain_amount": 0.0,   # 0 .. 100
    "grain_size": 25.0,    # 0 .. 100
    "grain_rough": 50.0,   # 0 .. 100
    # Pincel corrector: trazos [[radio, [[x, y], ...]], ...] normalizados 0..1
    # (se aplican con LaMa fuera del motor; propios de cada foto, no se copian)
    "heal_strokes": [],
    # Borrado generativo: [{"map": png_base64, "seed": n}, ...] en orden
    # (se aplican con difusion fuera del motor; propios de cada foto)
    "erase_ops": [],
    # Curva de tonos parametrica
    "p_highlights": 0.0,  # iluminaciones, -100 .. 100
    "p_lights": 0.0,      # claros
    "p_darks": 0.0,       # oscuros
    "p_shadows": 0.0,     # sombras
    # Curvas de puntos (lista de [x, y] en 0..1)
    "curve_rgb": DEFAULT_CURVE,
    "curve_r": DEFAULT_CURVE,
    "curve_g": DEFAULT_CURVE,
    "curve_b": DEFAULT_CURVE,
}

# Mezclador HSL: 8 bandas de color, cada una con matiz/saturacion/luminancia
HSL_BANDS = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
HSL_CENTERS = [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 280.0, 320.0]
DEFAULT_EDITS.update({f"hsl_{b}_{c}": 0.0 for b in HSL_BANDS for c in "hsl"})

# Color de punto: se muestrea un color de la foto y se ajusta solo ese color
DEFAULT_EDITS.update({
    "pc_sample": [],   # color muestreado [matiz 0..360, sat 0..1, valor 0..1]
    "pc_hue": 0.0,     # -100 .. 100 (rotacion de matiz hasta +-180 grados)
    "pc_sat": 0.0,     # -100 .. 100
    "pc_lum": 0.0,     # -100 .. 100
    "pc_range": 30.0,  # 5 .. 100 (amplitud en grados alrededor de la muestra)
})

# Perfil: interpretacion base del RAW antes de cualquier ajuste del usuario
# (como Adobe Color / Vivido / Retrato... en Lightroom)
DEFAULT_EDITS.update({"profile": "standard"})

# Curva base comun a TODOS los perfiles: es la interpretacion "Estandar",
# la que saca el RAW de su linealidad y lo deja con brillo correcto.
PROFILE_BASE_CURVE = [[0.0, 0.0], [0.10, 0.16], [0.25, 0.34], [0.45, 0.53],
                      [0.65, 0.73], [0.85, 0.89], [1.0, 1.0]]

# Caracter de cada perfil: curva de estilo que se aplica ENCIMA de la base
# (no la reemplaza) y saturacion tipo vitalidad. Asi todos los perfiles son
# variaciones del Estandar y no saltos bruscos de brillo.
PROFILE_RECIPES = {
    "standard":  (None, 1.0),
    "vivid":     ([[0.0, 0.0], [0.25, 0.225], [0.5, 0.51], [0.75, 0.79], [1.0, 1.0]], 1.22),
    "portrait":  ([[0.0, 0.0], [0.25, 0.245], [0.5, 0.505], [0.75, 0.765], [1.0, 1.0]], 1.06),
    "landscape": ([[0.0, 0.0], [0.25, 0.232], [0.5, 0.508], [0.75, 0.782], [1.0, 1.0]], 1.15),
    "flat":      ([[0.0, 0.06], [0.5, 0.5], [1.0, 0.94]], 0.88),
    "bw":        ([[0.0, 0.0], [0.25, 0.235], [0.5, 0.505], [0.75, 0.775], [1.0, 1.0]], 0.0),
}

# Calibracion de camara (como en Lightroom): redefine los primarios y
# tine las sombras. Actua sobre la mezcla de canales, antes que todo lo demas
DEFAULT_EDITS.update({
    "cal_shadow_tint": 0.0,  # -100 (verde) .. 100 (magenta)
    "cal_red_hue": 0.0,      # -100 .. 100 (rotacion del primario, +-30 grados)
    "cal_red_sat": 0.0,      # -100 .. 100
    "cal_green_hue": 0.0,
    "cal_green_sat": 0.0,
    "cal_blue_hue": 0.0,
    "cal_blue_sat": 0.0,
})

# Mascaras con ajustes locales: lista de diccionarios
#   {"type": "linear"|"radial"|"brush"|"subject"|"background",
#    geometria normalizada 0..1 segun el tipo, "invert": 0/1,
#    "adjust": {"exposure": ..., "contrast": ..., ...}}
# Las coordenadas son relativas a la foto YA recortada/girada (lo que se ve)
DEFAULT_EDITS.update({"masks": []})

MASK_ADJUST_KEYS = ("exposure", "contrast", "highlights", "shadows",
                    "temperature", "tint", "saturation")
# IA local por mascara: se mezcla el resultado IA solo en la zona de la mascara
MASK_AI_KEYS = ("ai_denoise", "ai_face")

# Mascaras de retrato: que numeros de zona (CelebAMask-HQ, ver face_parse.py)
# forman cada mascara que se le ofrece al usuario
FACE_PART_IDS = {
    "skin":  (1, 7, 8, 10, 14),  # cara, orejas, nariz y cuello
    "brows": (2, 3),
    "eyes":  (4, 5),
    "lips":  (12, 13),
    "teeth": (11,),
    "hair":  (17,),
}

# Recorte y transformacion (se aplican antes que el resto de ajustes)
DEFAULT_EDITS.update({
    "crop": [],          # [x0, y0, x1, y1] normalizado sobre la foto ya girada
    "straighten": 0.0,   # enderezar, grados -45 .. 45
    "rot90": 0,          # cuartos de vuelta en sentido horario (0..3)
    "flip_h": 0,         # volteo horizontal (0/1)
    "flip_v": 0,         # volteo vertical (0/1)
})


def full_edits(partial=None):
    """Copia completa de los ajustes por defecto, actualizada con `partial`."""
    edits = copy.deepcopy(DEFAULT_EDITS)
    if partial:
        edits.update(copy.deepcopy(dict(partial)))
    return edits


def pchip_lut(points, n=256):
    """Tabla de mapeo suave y monotona (interpolacion PCHIP) para una curva
    de puntos [[x, y], ...] en 0..1."""
    pts = sorted((float(x), float(y)) for x, y in points)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    grid = np.linspace(0.0, 1.0, n)
    if len(xs) < 2:
        return np.clip(np.full(n, ys[0] if len(ys) else 0.0), 0.0, 1.0)

    h = np.maximum(np.diff(xs), 1e-9)
    d = np.diff(ys) / h
    m = np.zeros(len(xs))
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, len(xs) - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])

    idx = np.clip(np.searchsorted(xs, grid) - 1, 0, len(h) - 1)
    t = np.clip((grid - xs[idx]) / h[idx], 0.0, 1.0)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    y = h00 * ys[idx] + h10 * h[idx] * m[idx] + h01 * ys[idx + 1] + h11 * h[idx] * m[idx + 1]
    y = np.where(grid <= xs[0], ys[0], y)
    y = np.where(grid >= xs[-1], ys[-1], y)
    return np.clip(y, 0.0, 1.0)


def _parametric_lut(shadows, darks, lights, highlights, n=256):
    """Curva parametrica: cuatro regiones con transicion suave."""
    x = np.linspace(0.0, 1.0, n)

    def bump(center, width=0.38):
        t = np.clip((x - center) / width, -1.0, 1.0)
        return (np.cos(t * np.pi) + 1.0) / 2.0

    y = x + 0.15 * (shadows * bump(0.08) + darks * bump(0.35)
                    + lights * bump(0.65) + highlights * bump(0.92))
    return np.clip(y, 0.0, 1.0)


def _compose(base, lut):
    """Compone dos tablas: resultado(x) = lut(base(x))."""
    grid = np.linspace(0.0, 1.0, len(lut))
    return np.interp(base, grid, lut)


# Resolucion de las tablas de curvas: 4096 niveles evita el bandeado y
# permite aplicarlas por indexado directo (mucho mas rapido que interpolar)
LUT_N = 4096


def _tone_curve_lut(exposure, contrast, shadows, highlights, whites, blacks,
                    n=LUT_N):
    """Curva unificada tipo Lightroom PV2012.

    Los 6 sliders modifiban una sola curva que se aplica como LUT.
    Orden de operaciones (como Lightroom):
      1. Blancos / Negros (mueven los extremos del histograma)
      2. Sombras / Luces (modifican sus zonas con curva suave)
      3. Exposicion (multiplica en lineal, ancla negros)
      4. Contraste (S-curve desde 0.5)

    Args:
        exposure: -3.0 .. +3.0 (EV)
        contrast: -1.0 .. +1.0
        shadows: -1.0 .. +1.0
        highlights: -1.0 .. +1.0
        whites: -1.0 .. +1.0
        blacks: -1.0 .. +1.0
    Returns:
        LUT float32 de n valores en 0..1
    """
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)

    # 1. Blancos y negros: mueven los extremos del histograma
    #    Whites: estira/comprime el extremo blanco
    #    Blacks: estira/comprime el extremo negro
    #    Funcion: soft clip en los extremos
    if whites or blacks:
        # Punto blanco efectivo (1.0 = sin cambio)
        white_point = 1.0 - whites * 0.15   # whites +1 → 0.85
        # Punto negro efectivo (0.0 = sin cambio)
        black_point = -blacks * 0.15         # blacks +1 → -0.15 (aclara, mate)
        # Remapear de [black_point, white_point] a [0, 1]
        span = max(white_point - black_point, 0.01)
        x = np.clip((x - black_point) / span, 0.0, 1.0)

    # 2. Sombras y luces: curvas suaves tipo sigmoid por zona
    #    Sombras: afecta desde ~0.05 hasta ~0.5 (protege negros puros)
    #    Luces: afecta desde ~0.5 hasta ~0.95 (protege blancos puros)
    if shadows or highlights:
        # Zona de sombras: pico en 0.15, ancho 0.35
        sh_mask = np.exp(-((x - 0.15) / 0.35) ** 2)
        # Proteger negros puros: rampa de 0 en x=0 a 1 en x=0.08
        sh_mask *= np.clip(x / 0.08, 0.0, 1.0)
        # Zona de luces: pico en 0.85, ancho 0.35
        hi_mask = np.exp(-((x - 0.85) / 0.35) ** 2)
        # Proteger blancos puros: rampa de 1 en x=0.92 a 0 en x=1.0
        hi_mask *= np.clip((1.0 - x) / 0.08, 0.0, 1.0)
        x = np.clip(x + shadows * 0.25 * sh_mask + highlights * 0.25 * hi_mask,
                     0.0, 1.0)

    # 3. Exposicion: multiplica en escala lineal (como Lightroom)
    #    Ancla negros: pixel 0 sigue siendo 0
    if exposure:
        scale = 2.0 ** exposure
        x = np.clip(x * scale, 0.0, 1.0)

    # 4. Contraste: S-curve suave desde el punto medio 0.5
    #    Positivo = mas contraste (curva mas pronunciada); negativo = menos
    #    contraste (acerca todo al gris medio, como bajar el contraste real)
    if contrast:
        if contrast > 0:
            # k crece con la intensidad: 1 (casi lineal) .. 10 (S marcada)
            k = 1.0 + contrast * 9.0
            sigmoid = 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
            # normalizar para que siga tocando 0 y 1 en los extremos
            s0 = 1.0 / (1.0 + np.exp(k * 0.5))
            s1 = 1.0 / (1.0 + np.exp(-k * 0.5))
            sigmoid = (sigmoid - s0) / max(s1 - s0, 1e-6)
            x = x * (1.0 - contrast) + sigmoid * contrast
        else:
            # reduce el rango tonal en torno al gris medio (mas plano)
            x = 0.5 + (x - 0.5) * (1.0 + contrast)
        x = np.clip(x, 0.0, 1.0)

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _build_channel_luts(e):
    """Devuelve (lut_r, lut_g, lut_b) combinadas, o None si no hay curvas activas."""
    par = (e["p_shadows"], e["p_darks"], e["p_lights"], e["p_highlights"])
    has_par = any(par)
    has_rgb = e["curve_rgb"] != DEFAULT_CURVE
    has_chan = any(e[k] != DEFAULT_CURVE for k in ("curve_r", "curve_g", "curve_b"))
    if not (has_par or has_rgb or has_chan):
        return None

    base = np.linspace(0.0, 1.0, LUT_N)
    if has_par:
        base = _parametric_lut(par[0] / 100.0, par[1] / 100.0,
                               par[2] / 100.0, par[3] / 100.0, n=LUT_N)
    if has_rgb:
        base = _compose(base, pchip_lut(e["curve_rgb"], LUT_N))

    luts = []
    for key in ("curve_r", "curve_g", "curve_b"):
        if e[key] != DEFAULT_CURVE:
            luts.append(_compose(base, pchip_lut(e[key], LUT_N)))
        else:
            luts.append(base)
    return [lut.astype(np.float32) for lut in luts]


def _luminance(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _lut_index(img):
    """Indices uint16 para leer una tabla (LUT) de LUT_N entradas.

    Escrito paso a paso y reutilizando el mismo array en vez de
    `(np.clip(img, 0, 1) * (LUT_N - 1) + 0.5).astype(np.uint16)`: esa version
    en una linea deja cuatro copias intermedias de la foto entera por el
    camino, y en una vista previa de 3 megapixeles cada copia son 37 MB que
    hay que reservar, llenar y tirar. El resultado es el mismo numero.
    """
    x = np.clip(img, 0.0, 1.0)
    x *= (LUT_N - 1)
    x += 0.5
    return x.astype(np.uint16)


def _sharpen_mask(img, masking):
    """Mascara de bordes para el enfoque: 1 = se enfoca, 0 = protegido.
    Se calcula a media resolucion: la mascara es suave y asi cuesta 1/4."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(w // 2, 8), max(h // 2, 8)),
                       interpolation=cv2.INTER_AREA)
    lum = np.ascontiguousarray(_luminance(small), dtype=np.float32)
    soft = cv2.GaussianBlur(lum, (0, 0), 1.0)
    gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx * gx + gy * gy)
    edge = np.clip(grad / (np.percentile(grad, 95.0) + 1e-6), 0.0, 1.0)
    edge = cv2.GaussianBlur(edge, (0, 0), 1.0)
    mask = np.clip(edge / max(masking, 1e-6), 0.0, 1.0)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


def _inscribed_rect(w, h, angle_rad):
    """Mayor rectangulo con la proporcion original inscrito en la imagen
    girada `angle_rad`: evita que aparezcan bordes vacios al enderezar."""
    if w <= 0 or h <= 0:
        return w, h
    sin_a, cos_a = abs(math.sin(angle_rad)), abs(math.cos(angle_rad))
    width_longer = w >= h
    side_long, side_short = (w, h) if width_longer else (h, w)
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def _apply_geometry(base, e):
    """Giros de 90, volteos, enderezado y recorte. Devuelve siempre una copia."""
    if e.get("_skip_geometry"):
        return base.copy()
    img = base
    k = int(e.get("rot90", 0)) % 4
    if k:
        img = np.rot90(img, -k)  # sentido horario
    if e.get("flip_h"):
        img = img[:, ::-1]
    if e.get("flip_v"):
        img = img[::-1]
    ang = float(e.get("straighten", 0.0))
    if ang:
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        wr, hr = _inscribed_rect(w, h, math.radians(ang))
        x0 = max(int((w - wr) / 2.0), 0)
        y0 = max(int((h - hr) / 2.0), 0)
        img = img[y0:y0 + max(int(hr), 1), x0:x0 + max(int(wr), 1)]
    crop = e.get("crop")
    if crop and not e.get("_skip_crop"):
        h, w = img.shape[:2]
        xa, ya = int(crop[0] * w), int(crop[1] * h)
        xb, yb = int(crop[2] * w), int(crop[3] * h)
        if xb - xa >= 8 and yb - ya >= 8:
            img = img[ya:yb, xa:xb]
    # nunca devolver memoria compartida con la base: los ajustes se aplican
    # in place y corromperian la cache (p. ej. un recorte a lo ancho completo
    # es una vista contigua que ascontiguousarray NO copia)
    out = np.ascontiguousarray(img)
    if np.may_share_memory(out, base):
        out = out.copy()
    return out


def _rasterize_mask_strokes(strokes, h, w):
    """Trazos normalizados -> mapa de peso 0..1 con borde suavizado.
    Cada trazo puede llevar un tercer elemento -1: borra (quita) en vez
    de anadir. Se aplican en orden."""
    m8 = np.zeros((h, w), np.uint8)
    scale = max(h, w)
    for stroke in strokes:
        radius, points = stroke[0], stroke[1]
        value = 0 if (len(stroke) > 2 and stroke[2] < 0) else 255
        thickness = max(int(radius * scale * 2), 3)
        pts = np.array([[int(x * w), int(y * h)] for x, y in points], np.int32)
        if len(pts) == 1:
            cv2.circle(m8, tuple(pts[0]), max(thickness // 2, 2), value, -1)
        else:
            cv2.polylines(m8, [pts], False, value, thickness=thickness,
                          lineType=cv2.LINE_8)
    soft = cv2.GaussianBlur(m8.astype(np.float32) / 255.0,
                            (0, 0), max(scale * 0.003, 2.0))
    return np.clip(soft, 0.0, 1.0)


def _apply_refine(wmap, strokes, h, w):
    """Refinado a mano de una mascara: trazos de pincel que ANADEN (+1) o
    QUITAN (-1) zonas sobre el mapa ya calculado, en el orden pintado."""
    if not strokes:
        return wmap
    sign_of = lambda s: -1 if (len(s) > 2 and s[2] < 0) else 1
    i, n = 0, len(strokes)
    while i < n:
        sign = sign_of(strokes[i])
        group = []
        while i < n and sign_of(strokes[i]) == sign:
            group.append((strokes[i][0], strokes[i][1]))
            i += 1
        m = _rasterize_mask_strokes(group, h, w)
        wmap = np.clip(wmap + m if sign > 0 else wmap - m, 0.0, 1.0)
    return wmap


def mask_weight(mask, h, w, get_ai=None):
    """Mapa de peso 0..1 de una mascara para una imagen de h x w.
    `get_ai(tipo)` provee el mapa de la segmentacion IA (o None)."""
    kind = mask.get("type")
    if kind == "linear":
        x0, y0 = mask["x0"] * w, mask["y0"] * h
        x1, y1 = mask["x1"] * w, mask["y1"] * h
        dx, dy = x1 - x0, y1 - y0
        norm = dx * dx + dy * dy
        if norm < 1e-6:
            wmap = np.ones((h, w), np.float32)
        else:
            xs = np.arange(w, dtype=np.float32)[None, :]
            ys = np.arange(h, dtype=np.float32)[:, None]
            t = ((xs - x0) * dx + (ys - y0) * dy) / norm
            wmap = 1.0 - _smoothstep(t)  # 1 donde empiezas, 0 donde sueltas
    elif kind == "radial":
        cx, cy = mask["cx"] * w, mask["cy"] * h
        rx = max(mask["rx"] * w, 2.0)
        ry = max(mask["ry"] * h, 2.0)
        f = max(mask.get("feather", 50.0) / 100.0, 0.02)
        xs = np.arange(w, dtype=np.float32)[None, :]
        ys = np.arange(h, dtype=np.float32)[:, None]
        nd = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
        wmap = _smoothstep((1.0 + f - nd) / (2.0 * f))
    elif kind == "brush":
        wmap = _rasterize_mask_strokes(mask.get("strokes") or [], h, w)
    elif kind in ("subject", "background"):
        base_map = get_ai("subject") if get_ai else None
        if base_map is None:
            return None
        wmap = base_map if kind == "subject" else 1.0 - base_map
    elif kind == "face_part":
        base_map = (get_ai("face_" + mask.get("part", "skin"))
                    if get_ai else None)
        if base_map is None:
            return None
        wmap = base_map
    else:
        return None
    if mask.get("invert"):
        wmap = 1.0 - wmap
    # el refinado va al final: el usuario pinta sobre lo que esta viendo
    wmap = _apply_refine(wmap, mask.get("refine"), h, w)
    return wmap.astype(np.float32)


def mask_weight_base(mask, e, h, w, ai_masks=None):
    """Mapa de peso de una mascara pero en el marco de la foto BASE (h x w,
    antes de la geometria): calcula el mapa en el marco visible y deshace
    recorte -> enderezado -> volteos -> giros. Las zonas que quedaron fuera
    del encuadre no son seleccionables (valen 0). Para el borrado generativo,
    que trabaja sobre la foto original como el corrector."""
    k = int(e.get("rot90", 0)) % 4
    h1, w1 = (w, h) if k % 2 else (h, w)
    ang = float(e.get("straighten", 0.0))
    if ang:
        wr, hr = _inscribed_rect(w1, h1, math.radians(ang))
        hs, ws = max(int(hr), 1), max(int(wr), 1)
        ys = max(int((h1 - hr) / 2.0), 0)
        xs = max(int((w1 - wr) / 2.0), 0)
    else:
        hs, ws = h1, w1
    crop = e.get("crop")
    if crop:
        xa, ya = int(crop[0] * ws), int(crop[1] * hs)
        xb, yb = int(crop[2] * ws), int(crop[3] * hs)
        if xb - xa < 8 or yb - ya < 8:
            crop = None
    hc, wc = (yb - ya, xb - xa) if crop else (hs, ws)

    def get_ai(kind):
        m0 = ai_source_map(ai_masks, kind)
        return None if m0 is None else transform_ai_map(m0, e, hc, wc)

    wmap = mask_weight(mask, hc, wc, get_ai)
    if wmap is None:
        return None
    if crop:
        full = np.zeros((hs, ws), np.float32)
        full[ya:yb, xa:xb] = wmap
        wmap = full
    if ang:
        full = np.zeros((h1, w1), np.float32)
        full[ys:ys + hs, xs:xs + ws] = wmap
        m = cv2.getRotationMatrix2D((w1 / 2.0, h1 / 2.0), -ang, 1.0)
        wmap = cv2.warpAffine(full, m, (w1, h1), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    if e.get("flip_v"):
        wmap = wmap[::-1]
    if e.get("flip_h"):
        wmap = wmap[:, ::-1]
    if k:
        wmap = np.rot90(wmap, k)
    return np.ascontiguousarray(wmap, dtype=np.float32)


def transform_ai_map(map2d, e, h, w):
    """Aplica la misma geometria de la foto al mapa IA y lo escala a h x w."""
    m = _apply_geometry(map2d, e)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0)


def ai_source_map(ai_masks, kind):
    """Mapa pre-geometria 0..1 para una mascara IA: "subject" tal cual, y
    "face_<zona>" se construye desde el mapa de etiquetas del retrato."""
    if not ai_masks:
        return None
    if kind.startswith("face_"):
        labels = ai_masks.get("face_labels")
        ids = FACE_PART_IDS.get(kind[5:])
        if labels is None or ids is None:
            return None
        m = np.isin(labels, np.array(ids, np.uint8)).astype(np.float32)
        # borde ligeramente suavizado para fusiones naturales
        m = cv2.GaussianBlur(m, (0, 0), max(max(labels.shape) / 800.0, 1.0))
        return np.clip(m, 0.0, 1.0)
    return ai_masks.get(kind)


def _blend_ai_masked(img, e, ai_masks, denoised, faced, base):
    """IA local por mascara: mezcla el resultado de la IA (ruido / rostros)
    solo en la zona de cada mascara. Va justo despues de la geometria, sobre
    la imagen base, igual que la mezcla global que hace RenderJob."""
    todo = [m for m in (e.get("masks") or [])
            if any((m.get("adjust") or {}).get(k) for k in MASK_AI_KEYS)]
    if not todo:
        return img
    h, w = img.shape[:2]

    def get_ai(kind):
        m0 = ai_source_map(ai_masks, kind)
        return None if m0 is None else transform_ai_map(m0, e, h, w)

    orig = img       # base tras geometria, antes de cualquier mezcla IA
    den_geo = fac_geo = None
    for m in todo:
        a = m.get("adjust") or {}
        a_den = a.get("ai_denoise", 0.0) / 100.0
        a_fac = a.get("ai_face", 0.0) / 100.0
        if a_den <= 0 and a_fac <= 0:
            continue
        wmap = mask_weight(m, h, w, get_ai)
        if wmap is None:
            continue  # mascara IA sin su mapa calculado todavia
        if a_den > 0 and denoised is not None and denoised.shape == base.shape:
            if den_geo is None:
                den_geo = _apply_geometry(denoised, e)
            wc = (wmap * a_den)[..., None]
            img = img * (1.0 - wc) + den_geo * wc
        if a_fac > 0 and faced is not None and faced.shape == base.shape:
            if fac_geo is None:
                fac_geo = _apply_geometry(faced, e)
            # rostros: se suma solo la diferencia, para no deshacer el denoise
            img = np.clip(img + (fac_geo - orig) * (wmap * a_fac)[..., None],
                          0.0, 1.0)
    return img


def _apply_masks(img, e, ai_masks=None):
    """Ajustes locales: cada mascara aplica su receta pesada por su mapa."""
    todo = [m for m in (e.get("masks") or [])
            if any((m.get("adjust") or {}).get(k) for k in MASK_ADJUST_KEYS)]
    if not todo:
        return img
    h, w = img.shape[:2]

    def get_ai(kind):
        m0 = ai_source_map(ai_masks, kind)
        return None if m0 is None else transform_ai_map(m0, e, h, w)

    for m in todo:
        wmap = mask_weight(m, h, w, get_ai)
        if wmap is None:
            continue  # mascara IA sin su mapa calculado todavia
        a = m.get("adjust") or {}
        wc = wmap[..., None]
        ev = a.get("exposure", 0.0)
        if ev:
            img = img * np.power(2.0, ev * wmap)[..., None]
        temp = a.get("temperature", 0.0) / 100.0
        tint = a.get("tint", 0.0) / 100.0
        if temp:
            img[..., 0] *= 1.0 + 0.30 * temp * wmap
            img[..., 2] *= 1.0 - 0.30 * temp * wmap
        if tint:
            img[..., 1] *= 1.0 - 0.20 * tint * wmap
        sh = a.get("shadows", 0.0) / 100.0
        hi = a.get("highlights", 0.0) / 100.0
        if sh or hi:
            lum = np.clip(_luminance(img), 0.0, 1.0)
            if sh:
                mask = (1.0 - lum) ** 2
                black_ramp = np.clip(lum / 0.15, 0.0, 1.0)
                mask *= black_ramp
                img += (sh * 0.50) * (mask * wmap)[..., None]
            if hi:
                img *= 1.0 + (hi * 0.55) * (lum ** 2 * wmap)[..., None]
        c = a.get("contrast", 0.0) / 100.0
        if c:
            img = img + (img - 0.5) * (0.8 * c) * wc
        s = a.get("saturation", 0.0) / 100.0
        if s:
            lum = _luminance(img)[..., None]
            img = lum + (img - lum) * (1.0 + s * wc)
        img = np.clip(img, 0.0, 1.0)
    return img


def _apply_profile(img, name):
    """Interpretacion base del RAW. Todos los perfiles parten de la misma
    curva base (la del Estandar) y le suman su caracter encima, para que
    cambiar de perfil sea una variacion y no un salto de brillo."""
    if name == "":
        return img
    recipe = PROFILE_RECIPES.get(name)
    if recipe is None:
        return img
    style, sat = recipe

    # base y estilo se componen en una sola tabla: estilo(base(x))
    lut = pchip_lut(PROFILE_BASE_CURVE, LUT_N).astype(np.float32)
    if style is not None:
        slut = pchip_lut(style, LUT_N).astype(np.float32)
        lut = slut[(lut * (LUT_N - 1) + 0.5).astype(np.uint16)]
    img = lut[_lut_index(img)]

    if name == "bw":
        lum = np.clip(_luminance(img), 0.0, 1.0).astype(np.float32)
        return np.repeat(lum[..., None], 3, axis=2)

    if sat != 1.0:
        # al estilo vitalidad: refuerza menos lo ya saturado (cuida la piel)
        lum = _luminance(img)[..., None]
        rango = np.clip((img.max(-1) - img.min(-1)) * 1.5, 0.0, 1.0)
        factor = 1.0 + (sat - 1.0) * (1.0 - rango)[..., None]
        # img viene de leer la tabla, asi que es nuestro y se puede pisar
        img -= lum
        img *= factor
        img += lum
        np.clip(img, 0.0, 1.0, out=img)
    return img


def _apply_calibration(img, e):
    """Calibracion de camara: rota/satura los primarios RGB con una matriz
    de mezcla de canales que conserva los blancos, y tine las sombras."""
    prim = [e["cal_red_hue"], e["cal_red_sat"], e["cal_green_hue"],
            e["cal_green_sat"], e["cal_blue_hue"], e["cal_blue_sat"]]
    tint = e["cal_shadow_tint"] / 100.0
    if not (any(prim) or tint):
        return img

    if any(prim):
        cols = []
        for base_hue, hue, sat in ((0.0, prim[0], prim[1]),
                                   (120.0, prim[2], prim[3]),
                                   (240.0, prim[4], prim[5])):
            h = ((base_hue + hue / 100.0 * 30.0) % 360.0) / 360.0
            p = np.array(colorsys.hsv_to_rgb(h, 1.0, 1.0), np.float32)
            # saturacion del primario: -100 casi gris, +100 reforzado
            scale = 1.0 + sat / 100.0 * 0.8
            p = p.mean() + (p - p.mean()) * scale
            cols.append(p)
        m = np.stack(cols, axis=1)  # columnas = primarios nuevos
        rows = m.sum(axis=1, keepdims=True)
        m = m / np.where(np.abs(rows) < 1e-3, 1e-3, rows)  # blanco intacto
        img = np.clip(img @ m.T.astype(np.float32), 0.0, 1.0)

    if tint:
        # matiz de sombras: verde (-) / magenta (+), solo en zonas oscuras
        lum = np.clip(_luminance(img), 0.0, 1.0)
        mask = (1.0 - lum) ** 2 * tint
        img[..., 1] = np.clip(img[..., 1] * (1.0 - 0.25 * mask), 0.0, 1.0)
    return img


def _smoothstep(x):
    t = np.clip(x, 0.0, 1.0)
    # (t*t) * (3 - 2t), en ese orden y reusando arrays: la cuenta es la misma
    out = t * t
    t *= -2.0
    t += 3.0
    out *= t
    return out


def _apply_hsl(img, e):
    """Mezclador HSL por bandas de color + ajuste de color de punto."""
    band_vals = [(e[f"hsl_{b}_h"], e[f"hsl_{b}_s"], e[f"hsl_{b}_l"])
                 for b in HSL_BANDS]
    has_mix = any(v for triple in band_vals for v in triple)
    sample = e.get("pc_sample")
    has_pc = bool(sample) and any((e["pc_hue"], e["pc_sat"], e["pc_lum"]))
    if not (has_mix or has_pc):
        return img

    # el clip ya devuelve float32: un astype aqui solo duplicaria la foto
    hsv = cv2.cvtColor(np.clip(img, 0.0, 1.0), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # los grises y los pixeles casi negros no tienen matiz fiable: se protegen
    w_base = _smoothstep(s / 0.18) * _smoothstep(v / 0.08)

    if has_mix:
        # interpolacion alrededor de la rueda: cada pixel mezcla las dos
        # bandas vecinas segun su matiz. Tabla de 361 grados + indexado
        # directo (interpolar por pixel es ~30x mas lento)
        xs = [HSL_CENTERS[-1] - 360.0] + HSL_CENTERS + [HSL_CENTERS[0] + 360.0]
        h_idx = np.clip(h, 0.0, 360.0).astype(np.int16)

        def band_lut(i):
            ys = ([band_vals[-1][i]] + [bv[i] for bv in band_vals]
                  + [band_vals[0][i]])
            table = (np.interp(np.arange(361), xs, ys) / 100.0).astype(np.float32)
            return table[h_idx]

        h = h + band_lut(0) * 30.0 * w_base
        s = s * (1.0 + band_lut(1) * w_base)
        v = v * (1.0 + band_lut(2) * 0.7 * w_base)

    if has_pc:
        h0 = float(sample[0])
        rng = max(float(e["pc_range"]), 5.0)
        dist = np.abs((h - h0 + 180.0) % 360.0 - 180.0)
        w = np.exp(-(dist / rng) ** 2) * w_base
        h = h + e["pc_hue"] / 100.0 * 180.0 * w
        s = s * (1.0 + e["pc_sat"] / 100.0 * w)
        v = v * (1.0 + e["pc_lum"] / 100.0 * 0.7 * w)

    hsv[..., 0] = h % 360.0
    hsv[..., 1] = np.clip(s, 0.0, 1.0)
    hsv[..., 2] = np.clip(v, 0.0, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def apply_edits(base, edits, ai_masks=None, denoised=None, faced=None, is_raw=False):
    """base: float32 RGB en rango 0..1. Devuelve uint8 RGB listo para mostrar.
    `ai_masks`: mapas de segmentacion IA pre-geometria, p. ej. {"subject": m}.
    `denoised` / `faced`: resultados IA pre-geometria (misma forma que base),
    para las mascaras con ruido IA / rostros IA locales."""
    e = {**DEFAULT_EDITS, **edits}
    img = _apply_geometry(base, e)
    if denoised is not None or faced is not None:
        img = _blend_ai_masked(img, e, ai_masks, denoised, faced, base)

    # Perfil base y calibracion de camara: primero, definen el punto
    # de partida sobre el que actua todo lo demas
    # Solo se aplica a RAW (JPEGs ya tienen el procesamiento de camara)
    if is_raw:
        img = auto_tone(img)
        img = _apply_profile(img, e.get("profile", "standard"))
    img = _apply_calibration(img, e)

    # Mapeo tonal adaptativo: comprime el rango dinamico del RAW de forma
    # inteligente, preservando contraste local (como Lightroom / Capture One)
    tm = e.get("tone_map", 0.0) / 100.0
    if tm > 0:
        img = img * (1.0 - tm) + tone_mapping(img) * tm

    # Balance de blancos (temperatura / matiz)
    temp = e["temperature"] / 100.0
    tint = e["tint"] / 100.0
    if temp or tint:
        img *= np.array([1.0 + 0.30 * temp,
                         1.0 - 0.20 * tint,
                         1.0 - 0.30 * temp], np.float32)

    # Curva tonal unificada (Lightroom PV2012 style):
    # Los 6 sliders (exposure, contrast, shadows, highlights, whites, blacks)
    # generan UNA sola curva que se aplica como LUT.
    tone_lut = _tone_curve_lut(
        exposure=e["exposure"] / 3.0,      # normalizar a -1..1
        contrast=e["contrast"] / 100.0,    # -1..1
        shadows=e["shadows"] / 100.0,      # -1..1
        highlights=e["highlights"] / 100.0, # -1..1
        whites=e["whites"] / 100.0,        # -1..1
        blacks=e["blacks"] / 100.0,        # -1..1
    )
    # la misma curva para los tres canales: se lee la tabla de una pasada en
    # vez de canal a canal (leer `idx[..., c]` va salteado por la memoria)
    img = tone_lut[_lut_index(img)]
    np.clip(img, 0.0, 1.0, out=img)

    # Borrar neblina (dehaze)
    dh = e["dehaze"] / 100.0
    if dh > 0:
        # Estimacion de neblina con el canal oscuro suavizado; el mapa es
        # muy suave, asi que se calcula a 1/4 de resolucion (mucho mas rapido)
        small = cv2.resize(img, None, fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_AREA)
        dark = cv2.GaussianBlur(small.min(axis=-1), (0, 0), 15 * 0.25)
        dark = cv2.resize(dark, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        t = np.clip(dh * 0.7 * dark, 0.0, 0.9)[..., None]
        img = np.clip((img - t) / (1.0 - t), 0.0, 1.0)
    elif dh < 0:
        a = -dh * 0.5
        img = img * (1.0 - a) + a  # añade neblina (mezcla hacia blanco)

    # Contraste adaptativo local (CLAHE): mejora el contraste en sombras
    # e iluminaciones sin lavar la imagen general
    ac = e.get("adaptive_contrast", 0.0) / 100.0
    if ac > 0:
        # CLAHE con intensidad variable: suave (clip_limit 2.0) a fuerte (4.0)
        img = img * (1.0 - ac) + adaptive_contrast(img, clip_limit=2.0 + ac * 2.0) * ac

    # Curva de tonos (parametrica + puntos + por canal), por indexado directo
    luts = _build_channel_luts(e)
    if luts is not None:
        idx = _lut_index(img)
        for c in range(3):
            img[..., c] = luts[c][idx[..., c]]
        del idx

    # Saturacion y vitalidad
    sat = e["saturation"] / 100.0
    vib = e["vibrance"] / 100.0
    if sat or vib:
        lum = _luminance(img)[..., None]
        factor = 1.0 + sat
        if vib:
            current = img.max(-1) - img.min(-1)
            factor = factor + vib * (1.0 - np.clip(current * 2.0, 0.0, 1.0))
            factor = factor[..., None]
        # lum + (img - lum) * factor, hecho encima del propio img
        img -= lum
        img *= factor
        img += lum
        np.clip(img, 0.0, 1.0, out=img)

    # Mezclador HSL y color de punto
    img = _apply_hsl(img, e)

    # Mascaras con ajustes locales (degradados, pincel, sujeto/fondo IA)
    img = _apply_masks(img, e, ai_masks)

    # Vista previa de la mascara de enfoque (Alt sobre el deslizador Mascara):
    # blanco = zonas que reciben enfoque, negro = protegidas
    if e.get("_show_mask"):
        img = np.clip(img, 0.0, 1.0)
        mask = _sharpen_mask(img, e["sharp_masking"] / 100.0)
        gray = (mask * 255.0 + 0.5).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)

    # En el borrador de edicion rapida se omiten los efectos de detalle
    # (ruido, textura, enfoque, grano): no se aprecian mientras se arrastra
    # un ajuste tonal y cuestan la mitad del render
    if e.get("_draft_skip_detail"):
        return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    # Reduccion de ruido (antes del enfoque, como debe ser)
    nl = e["nr_luminance"] / 100.0
    nc = e["nr_color"] / 100.0
    if nl or nc:
        ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if nl:
            # bilateral: suaviza el grano conservando los bordes
            ycc[..., 0] = cv2.bilateralFilter(
                ycc[..., 0], 0, 0.03 + nl * 0.22, 2.0 + nl * 3.0)
        if nc:
            # el ruido de color se difumina en los canales de croma
            sigma = 1.0 + nc * 6.0
            ycc[..., 1] = cv2.GaussianBlur(ycc[..., 1], (0, 0), sigma)
            ycc[..., 2] = cv2.GaussianBlur(ycc[..., 2], (0, 0), sigma)
        img = np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)

    # Claridad: contraste local de radio grande, solo en medios tonos
    # (protege sombras y luces puras para no ensuciar negros/blancos, como
    # en Lightroom). Se calcula sobre la luminancia y se aplica por igual a
    # los 3 canales para no generar franjas de color.
    if e["clarity"]:
        lum = _luminance(img)
        blur = cv2.GaussianBlur(lum, (0, 0), 30.0)
        detail = lum - blur
        protect = np.clip(1.0 - (2.0 * lum - 1.0) ** 2, 0.0, 1.0)
        boost = (e["clarity"] / 100.0 * 0.6) * detail * protect
        img += boost[..., None]
        np.clip(img, 0.0, 1.0, out=img)

    # Textura (detalle de frecuencia media; negativo suaviza)
    if e["texture"]:
        blur = cv2.GaussianBlur(img, (0, 0), 4.0)
        # img + (img - blur) * k, reusando blur como cuaderno de notas
        np.subtract(img, blur, out=blur)
        blur *= (e["texture"] / 100.0 * 0.9)
        img += blur
        np.clip(img, 0.0, 1.0, out=img)

    # Enfoque: cantidad / radio / detalle / mascara
    if e["sharp_amount"]:
        amount = e["sharp_amount"] / 100.0 * 1.2
        sigma = max(float(e["sharp_radius"]), 0.3)
        high = img - cv2.GaussianBlur(img, (0, 0), sigma)
        # Detalle bajo = solo bordes fuertes (evita amplificar el ruido fino)
        detail = e["sharp_detail"] / 100.0
        thr = (1.0 - detail) ** 2 * 0.03
        if thr > 0:
            high = np.sign(high) * np.maximum(np.abs(high) - thr, 0.0)
        # Mascara: limita el enfoque a las zonas con bordes
        if e["sharp_masking"]:
            mask = _sharpen_mask(img, e["sharp_masking"] / 100.0)
            high *= mask[..., None]
        high *= amount
        img += high
        np.clip(img, 0.0, 1.0, out=img)

    # Grano de pelicula: monocromatico, mas fuerte en tonos medios (como el
    # grano real de negativo), deterministico para que no "hierva" al editar
    if e["grain_amount"]:
        h, w = img.shape[:2]
        # tamano de celda relativo a la resolucion: mismo aspecto en la vista
        # previa y en la exportacion a resolucion completa
        rel = max(h, w) / 2200.0
        cell = max((1.0 + e["grain_size"] / 100.0 * 2.5) * rel, 1.0)
        rng = np.random.default_rng(42)
        gh, gw = max(int(h / cell), 8), max(int(w / cell), 8)
        noise = rng.standard_normal((gh, gw)).astype(np.float32)
        softness = (100.0 - e["grain_rough"]) / 100.0
        if softness > 0.05:
            noise = cv2.GaussianBlur(noise, (0, 0), 0.4 + softness * 1.2)
            noise /= max(float(noise.std()), 1e-6)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        lum = _luminance(img)
        mid = np.clip(4.0 * lum * (1.0 - lum), 0.15, 1.0)
        # mismo orden que antes —(k * noise) * mid— para no mover ni un bit:
        # en float32 multiplicar en otro orden puede cambiar el ultimo decimal
        noise *= (e["grain_amount"] / 100.0 * 0.12)
        noise *= mid
        img += noise[..., None]
        np.clip(img, 0.0, 1.0, out=img)

    img *= 255.0
    img += 0.5
    return img.astype(np.uint8)
