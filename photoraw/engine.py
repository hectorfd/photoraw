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


AUTO_TARGET = 0.42

# Cuantiles con que se compara el RAW con la vista previa de la camara
CAMERA_QS = np.linspace(0.005, 0.995, 41)
# Pendiente maxima de la curva de camara: mas que esto, en las sombras mas
# hondas, solo sacaria ruido a la luz
CAMERA_MAX_SLOPE = 8.0


def camera_base_lut(base, camera):
    """Curva base para que el RAW abra con el reparto de tonos con que lo
    revelo la camara (`camera` = `loader.camera_tones`).

    Se empareja cuantil a cuantil el brillo del RAW con el de la vista previa
    incrustada: el tono que en el RAW deja por debajo al 30 % de la foto va
    al tono que deja por debajo al 30 % en la vista previa. Eso reproduce el
    brillo y el contraste de la camara foto a foto (la noche sigue siendo
    noche, la toma quemada sale quemada), cosa que ninguna curva fija
    conseguia: el iPhone aclara cada foto distinto. Se mide sobre la foto
    ENTERA y sin recortar, para que recortar no cambie los tonos."""
    small = np.ascontiguousarray(base[::8, ::8])
    lum = _luminance(small)
    a = np.quantile(lum, CAMERA_QS).astype(np.float64)
    b = np.asarray(camera, np.float64)
    # puntos de control estrictamente crecientes, anclados en 0 y 1
    xs, ys = [0.0], [0.0]
    for x, y in zip(a, b):
        if x > xs[-1] + 0.004:
            xs.append(x)
            ys.append(max(y, ys[-1]))
    if xs[-1] < 0.999:
        xs.append(1.0)
        ys.append(max(1.0, ys[-1]))
    xs, ys = np.array(xs), np.array(ys)
    # sin escalones: se limita la pendiente de cada tramo
    for i in range(1, len(xs)):
        ys[i] = min(ys[i], ys[i - 1] + (xs[i] - xs[i - 1]) * CAMERA_MAX_SLOPE)
    ys = np.minimum(ys, 1.0)
    # se aclara un poco la nube de puntos (cuantiles vecinos meten dientes)
    keep = np.unique(np.round(np.linspace(0, len(xs) - 1, 12)).astype(int))
    return pchip_lut(np.stack([xs[keep], ys[keep]], 1).tolist(),
                     LUT_N).astype(np.float32)


def auto_tone_weight(img, target=AUTO_TARGET):
    """Cuanto hay que levantar esta foto, de 0 (nada) a 1 (todo).

    Va aparte de `auto_tone` porque la medida tiene que salir de la foto
    ENTERA: la vista de detalle revela solo el trozo que estas mirando, y si
    cada trozo se midiera a si mismo, un recorte de una pared clara y otro de
    una sombra saldrian con exposiciones distintas."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(w // 8, 1), max(h // 8, 1)),
                       interpolation=cv2.INTER_AREA)
    med = float(np.median(_luminance(small)))
    if med >= target:
        return 0.0
    # `float(...)` a proposito: np.clip devuelve un np.float64 y multiplicar
    # la foto (float32) por un escalar float64 la asciende ENTERA a float64.
    # Asi salia de aqui, y el resto del revelado seguia en float64: el doble
    # de memoria y unas 3,5 veces mas lento por operacion, sin ninguna
    # ganancia de calidad visible (la foto acaba en 8 bits por canal)
    deficit = float(np.clip((target - med) / target, 0.0, 1.0))
    weight = deficit ** 1.5
    return weight if weight >= 0.01 else 0.0


def auto_tone(img, target=AUTO_TARGET, weight=None, log_avg=None):
    """Correccion automatica de exposicion al abrir un RAW (como hacen
    Lightroom / Windows Fotos): solo entra cuando la foto esta realmente
    oscura (contraluces, interiores), y no toca las que ya estan bien
    expuestas. Mide la mediana de luminancia (en espacio gamma, igual que
    ve el ojo) y mezcla mapeo tonal adaptativo en proporcion a cuanto le
    falta a esa mediana para llegar a `target`.

    Args:
        img: float32 RGB 0..1 (salida del revelado RAW, ya con gamma)
        target: mediana de luminancia deseada para una foto "bien expuesta"
        weight: fuerza ya medida sobre la foto entera (vista de detalle);
                si no se pasa, se mide aqui sobre `img`
    Returns:
        float32 RGB 0..1
    """
    if weight is None:
        weight = auto_tone_weight(img, target)
    if weight <= 0.0:
        return img
    mapped = tone_mapping(img, key=0.35, log_avg=log_avg)
    mapped *= weight                    # mapped es nuestro: se puede pisar
    mapped += img * (1.0 - weight)
    return mapped


def log_avg_luminance(img):
    """Luminancia media geometrica: el brillo "de referencia" de una foto.

    Se mide sobre la foto ENTERA. La vista de detalle revela trozos sueltos y
    cada trozo tiene su propio brillo medio; si cada uno se midiera a si
    mismo, el trozo de una sombra saldria aclarado y el de una pared clara
    oscurecido, y no pegarian ni entre ellos ni con la foto."""
    lum = img * np.array([0.2126, 0.7152, 0.0722], np.float32)
    lum = lum.sum(axis=-1)
    return float(np.exp(np.log(lum + 1e-6).mean()))


def tone_mapping(img, key=0.18, saturation=1.0, log_avg=None):
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
    # Luminancia promedio geometrico (mas preciso que aritmetico). Se calcula
    # tal cual (una copia de la foto entera y luego sumar el eje del color) a
    # proposito: hacerlo con _luminance suma los tres canales en otro orden y
    # en float32 eso mueve el ultimo bit — movia 1/255 en un pixel de cada
    # 100.000, invisible pero distinto
    if log_avg is None:
        log_avg = log_avg_luminance(img)

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
    "temperature": 0.0,  # -100 (frio) .. 100 (calido)  (JPG y mascaras)
    # Balance de blancos real de los RAW (ver wb.py): Kelvin y matiz como en
    # Lightroom. wb_temp 0 = "como se disparo" (los dos a la vez)
    "wb_temp": 0.0,      # 2000 .. 50000 K
    "wb_tint": 0.0,      # -150 (verde) .. 150 (magenta)
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
    "sharp_amount": 0.0,   # 0 .. 300
    "sharp_radius": 1.0,   # 0.5 .. 3.0
    "sharp_detail": 25.0,  # 0 .. 100
    "sharp_masking": 0.0,  # 0 .. 100
    # Reduccion de ruido manual
    "nr_luminance": 0.0,   # 0 .. 100
    # 1 de fabrica: se probo 25 (como Lightroom) y a Hector le quitaba
    # demasiada calidad; lo sube a mano cuando la foto lo pide
    "nr_color": 1.0,       # 0 .. 100
    # Reduccion de ruido IA (intensidad de mezcla; se aplica fuera del motor)
    "ai_denoise": 0.0,     # 0 .. 100
    # Retoque de rostros IA (intensidad de mezcla; se aplica fuera del motor)
    "ai_face": 0.0,        # 0 .. 100
    # --- Efectos creativos (panel Efectos) -----------------------------
    # Todos van al final del revelado, sobre la foto ya ajustada, y en este
    # mismo orden: primero los que cambian el contraste y el color, luego los
    # que anaden luz difusa, y el grano el ultimo (como en el laboratorio)
    "dramatic_amount": 0.0,      # 0 .. 100  fuerza del look dramatico
    "dramatic_contrast": 0.0,    # 0 .. 100  contraste local extra
    "dramatic_brightness": 0.0,  # -100 .. 100
    "dramatic_saturation": 0.0,  # -100 .. 100
    "mood_preset": "none",       # clave de MOOD_RECIPES
    "mood_amount": 50.0,         # 0 .. 100  mezcla del virado de color
    "tone_hi_hue": 45.0,         # 0 .. 360  virado de las luces
    "tone_hi_sat": 0.0,          # 0 .. 100
    "tone_sh_hue": 220.0,        # 0 .. 360  virado de las sombras
    "tone_sh_sat": 0.0,          # 0 .. 100
    "tone_balance": 0.0,         # -100 (mas sombra) .. 100 (mas luz)
    "matte_amount": 0.0,         # 0 .. 100  negros levantados, look mate
    "matte_fade": 50.0,          # 0 .. 100  cuanto se levanta el negro
    "matte_contrast": 0.0,       # -100 .. 100  contraste de compensacion
    "mystical_amount": 0.0,      # 0 .. 100  luz difusa de ensueno
    "mystical_shadows": 25.0,    # 0 .. 100  cuanto entra en las sombras
    "mystical_smooth": 0.0,      # -100 .. 100  suavizado del detalle fino
    "glow_mode": "orton",        # clave de GLOW_MODES
    "glow_amount": 0.0,          # 0 .. 100
    "glow_smooth": 50.0,         # 0 .. 100  radio del difuminado
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
# MEDIDA en Camera Raw 18 (perfil Adobe Color, todo a cero) con IMG_4210:
# cuantil a cuantil, lo que PhotoRAW abre con el perfil RAW (exposicion base
# del DNG incluida) frente a lo que ensena Photoshop. Es fija, como la de
# Adobe: la noche sigue oscura y cada foto conserva su exposicion
PROFILE_BASE_CURVE = [
    [0.0, 0.0], [0.0039, 0.02], [0.0078, 0.027], [0.0157, 0.055],
    [0.0235, 0.133], [0.0314, 0.196], [0.0471, 0.29], [0.0627, 0.341],
    [0.0941, 0.455], [0.1255, 0.529], [0.1882, 0.675], [0.251, 0.78],
    [0.3137, 0.875], [0.3765, 0.951], [0.502, 0.983], [0.6902, 0.999],
    [1.0, 1.0]]

# Caracter de cada perfil: curva de estilo que se aplica ENCIMA de la base
# (no la reemplaza) y saturacion tipo vitalidad. Asi todos los perfiles son
# variaciones del Estandar y no saltos bruscos de brillo.
PROFILE_RECIPES = {
    "standard":  (None, 1.0),
    "vivid":     ([[0.0, 0.0], [0.25, 0.225], [0.5, 0.51], [0.75, 0.79], [1.0, 1.0]], 1.22),
    "portrait":  ([[0.0, 0.0], [0.25, 0.245], [0.5, 0.505], [0.75, 0.765], [1.0, 1.0]], 1.06),
    "landscape": ([[0.0, 0.0], [0.25, 0.232], [0.5, 0.508], [0.75, 0.782], [1.0, 1.0]], 1.15),
    "bw":        ([[0.0, 0.0], [0.25, 0.235], [0.5, 0.505], [0.75, 0.775], [1.0, 1.0]], 0.0),
}

# El perfil "RAW" no tiene receta a proposito: es la AUSENCIA de todas. Con
# el puesto, el revelado se salta la interpretacion base del archivo -- ni
# arreglo automatico de exposicion ni curva base -- y te ensena lo que hay
# en el archivo, sin lectura previa de nadie. Es el punto de partida honesto
# para revelar desde cero (ver apply_edits).
RAW_PROFILE = "raw"
# El viejo "Plano" ocupaba esta misma casilla en el menu
PROFILE_ALIASES = {"flat": RAW_PROFILE}

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

# La curva de la mascara va aparte de los demas ajustes: todos ellos son un
# numero (y valen 0 cuando no se tocan), mientras que esto es una lista de
# puntos. Por eso no entra en MASK_ADJUST_KEYS, que es lo que la interfaz
# recorre para fabricar deslizadores.
MASK_CURVE_KEY = "curve"

MASK_ADJUST_KEYS = ("exposure", "contrast", "highlights", "shadows",
                    "whites", "blacks", "temperature", "tint",
                    "saturation", "vibrance", "clarity", "texture",
                    "dehaze", "sharp_amount")
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
    """Curva parametrica: cuatro regiones con transicion suave, como la de
    Lightroom.

    Antes se sumaban cuatro campanas sin mas, y medido: Sombras +100 subia
    el negro puro a 34/255 y Iluminaciones -100 bajaba el blanco a 221 (foto
    lavada); Iluminaciones +100 quemaba de golpe el 14 % de los tonos, y
    Claros +100 con Iluminaciones -100 hacia que un 11 % de la curva BAJARA
    (tonos invertidos: manchas en cielos y pieles). Ahora el negro y el blanco
    quedan clavados, cada campana se apaga al llegar a su extremo y la curva
    nunca baja ni se aplana del todo."""
    x = np.linspace(0.0, 1.0, n)

    def bump(center, width=0.25):
        t = np.clip((x - center) / width, -1.0, 1.0)
        return (np.cos(t * np.pi) + 1.0) / 2.0

    # se apaga en los extremos: el negro y el blanco no se mueven. La rampa
    # mide lo mismo que la amplitud (0,16): asi una campana sola nunca se pasa
    # del blanco ni baja del negro, y el re-escalado de abajo casi no actua
    # (con una rampa mas corta Iluminaciones +100 oscurecia los medios tonos)
    ancla = np.clip(np.minimum(x, 1.0 - x) / 0.16, 0.0, 1.0)
    y = x + 0.16 * ancla * (shadows * bump(0.14) + darks * bump(0.37)
                            + lights * bump(0.63) + highlights * bump(0.86))
    # Nunca se queda plana. Con la pendiente minima de antes (0,12) Oscuros
    # +100 e Iluminaciones +100 juntos dejaban una meseta entre las dos
    # campanas: de 112 a 176 salia 132..146, los medios tonos (la piel) casi
    # de un solo gris — medido en IMG_4205, cara plana y sin color. Ahora la
    # pendiente no baja de 0,45 (con un minimo suave, sin esquinas) y la luz
    # que eso anade se descuenta de los tramos que suben mas de la cuenta, no
    # de toda la curva: asi el ajuste conserva su fuerza donde la tiene
    pend = np.diff(y) * (n - 1)
    suelo, k = 0.45, 0.06
    nueva = suelo + k * np.logaddexp(0.0, (pend - suelo) / k)
    sobra = np.maximum(nueva - 1.0, 0.0)
    falta = nueva.sum() - pend.sum()
    if falta > 0 and sobra.sum() > 0:
        nueva -= sobra * min(falta / sobra.sum(), 1.0)
    y = np.concatenate([[0.0], np.cumsum(nueva)])
    return y / y[-1]


def _compose(base, lut):
    """Compone dos tablas: resultado(x) = lut(base(x))."""
    grid = np.linspace(0.0, 1.0, len(lut))
    return np.interp(base, grid, lut)


# Resolucion de las tablas de curvas: 4096 niveles evita el bandeado y
# permite aplicarlas por indexado directo (mucho mas rapido que interpolar)
LUT_N = 4096

# La exposicion multiplica valores que ya llevan gamma (~2,2): para que +1 en
# el deslizador sea un paso de luz de verdad (el doble de luz) se multiplica
# por 2^(1/2,2). Con el /3 de antes +1 daba 0,8 pasos (medido en IMG_4188).
EXPOSURE_GAMMA = 2.2


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
    #    Antes se reescalaba la foto entera entre dos puntos nuevos: Negros
    #    -100 oscurecia tambien las luces y Blancos +100 quemaba el 5 % de un
    #    golpe (medido en IMG_4188). Ahora cada uno empuja su extremo y se
    #    apaga hacia el otro: el blanco mueve sobre todo el cuarto de arriba,
    #    el negro el de abajo, y los medios tonos casi no se enteran.
    if whites or blacks:
        x = x + whites * 0.22 * x ** 3 + blacks * 0.22 * (1.0 - x) ** 3
        x = np.clip(x, 0.0, 1.0)

    # 2. Sombras y luces: se ESTIRA la zona, no se le suma luz.
    #
    #    Antes se sumaba una campana de luz a los tonos oscuros. Sumar sube el
    #    nivel pero deja los tonos igual de juntos: dos sombras que se
    #    diferenciaban en 0,03 se siguen diferenciando en 0,03, solo que ahora
    #    son grises claros. No aparece ni un detalle nuevo — es exactamente el
    #    aspecto de "sombra pintada de gris". Y peor: la bajada de la campana
    #    apretaba todo lo que habia entre 0,15 y 0,40 en una franja estrecha
    #    (medido: los tonos salian con el 39 % de su separacion original), que
    #    es la meseta gris que se veia.
    #
    #    Estirar con una gamma hace lo contrario: separa los tonos oscuros
    #    entre si, que es lo que hace visible el detalle que ya estaba en el
    #    archivo. Ademas el negro puro se queda clavado en 0 solo (cero
    #    elevado a lo que sea es cero), sin necesidad de rampas de proteccion.
    if shadows:
        # +1 -> gamma 0,57 (estira las sombras); -1 -> 4,0 (las hunde)
        g = 1.0 / (1.0 + shadows * 0.75)
        estirada = np.power(x, g, dtype=np.float32)
        # solo la zona oscura, desvaneciendose hacia los medios tonos
        m = np.clip(1.0 - x / 0.60, 0.0, 1.0) ** 1.4
        x = x + (estirada - x) * m
    if highlights:
        # lo mismo visto desde el blanco: se trabaja sobre 1-x, asi que el
        # blanco puro queda clavado en 1 por la misma razon
        inv = 1.0 - x
        g = 1.0 / (1.0 - highlights * 0.75)
        recogida = 1.0 - np.power(inv, g, dtype=np.float32)
        m = np.clip((x - 0.40) / 0.60, 0.0, 1.0) ** 1.4
        x = x + (recogida - x) * m
    if shadows or highlights:
        x = np.clip(x, 0.0, 1.0)

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
            # S normalizada cuya pendiente crece de forma pareja con el
            # deslizador: +100 = los medios tonos con ~1,9 veces su
            # contraste. Antes se mezclaba ADEMAS una S que se endurecia (k
            # hasta 10): los dos efectos se multiplicaban, de 0 a +50 casi no
            # pasaba nada y de +50 a +100 las sombras se iban a negro
            k = contrast * 7.0
            sigmoid = 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
            # normalizar para que siga tocando 0 y 1 en los extremos
            s0 = 1.0 / (1.0 + np.exp(k * 0.5))
            s1 = 1.0 / (1.0 + np.exp(-k * 0.5))
            x = (sigmoid - s0) / max(s1 - s0, 1e-6)
        else:
            # S invertida: aplana los medios tonos pero deja el negro en 0 y
            # el blanco en 1. Antes se apretaba todo hacia 0,5 en linea recta
            # y a -100 la foto entera quedaba de un solo gris.
            c = -contrast
            k = 1.0 + c * 6.0
            s0 = 1.0 / (1.0 + np.exp(k * 0.5))
            s1 = 1.0 / (1.0 + np.exp(-k * 0.5))
            s = np.clip(x * (s1 - s0) + s0, 1e-6, 1.0 - 1e-6)
            inversa = 0.5 + np.log(s / (1.0 - s)) / k
            x = x * (1.0 - c) + inversa * c
        x = np.clip(x, 0.0, 1.0)

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _guided_self(lum, r, eps):
    """Filtro guiado de la luminancia sobre si misma: la suaviza como un
    desenfoque de radio `r` pero respeta los bordes que contrastan mas que
    ~sqrt(eps). Se calcula a resolucion reducida y se amplia (la version
    "rapida" del filtro): el resultado es suave de por si."""
    h, w = lum.shape
    s = max(1, int(r // 4))
    small = lum if s == 1 else cv2.resize(
        lum, (max(w // s, 1), max(h // s, 1)), interpolation=cv2.INTER_AREA)
    k = 2 * max(1, int(round(r / s))) + 1
    box = lambda m: cv2.boxFilter(m, -1, (k, k), borderType=cv2.BORDER_REFLECT)
    media = box(small)
    var = box(small * small) - media * media
    a = var / (var + eps)
    b = media - a * media
    a, b = box(a), box(b)
    if s > 1:
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return a * lum + b


def _chroma_nr(ycc, amount, region=None):
    """Reduccion de ruido de color, in place sobre los canales Cr/Cb.

    Antes era un desenfoque gaussiano del color: limpiaba, pero corria el
    rojo de los labios o el azul del sillon por encima de la piel. Ahora es
    un bilateral conjunto (brillo + color): promedia los puntitos morados y
    verdes del ruido, que se desvian poco, pero no cruza un cambio de color
    o de brillo de verdad. El radio va en proporcion a la foto ENTERA, asi
    limpia lo mismo en pantalla, con zoom y al exportar (antes, a resolucion
    completa, el mismo valor limpiaba la mitad)."""
    h, w = ycc.shape[:2]
    lado = max(region[0], region[1]) if region else max(h, w)
    sigma = (1.0 + amount * 6.0) * lado / 2200.0
    # el ruido de color es de grano grueso y el ojo no ve detalle fino en
    # color: se puede calcular a menor resolucion y ampliar
    f = int(np.clip(sigma // 3, 1, 4))
    small = ycc if f == 1 else cv2.resize(
        ycc, (max(w // f, 1), max(h // f, 1)), interpolation=cv2.INTER_AREA)
    # COPIA: con f == 1 `small` es la propia foto, y bajar aqui el brillo
    # de la guia la oscurecia a la mitad
    guia = np.array(small, dtype=np.float32, copy=True)
    guia[..., 0] *= 0.5            # el grano de luz no debe frenar el promedio
    s = max(sigma / f, 0.8)
    # Un primer suavizado corto: el ruido de color de un movil se desvia
    # ~0,03 por canal, y el bilateral lo tomaba por "borde" y no lo tocaba
    # (a 25 solo quitaba el 30 %). Suavizado, el ruido ya no parece borde y
    # los cambios de color de verdad (labio/piel, gafas) siguen siendolo
    guia = cv2.GaussianBlur(guia, (0, 0), max(s * 0.5, 0.7))
    d = int(min(2 * int(np.ceil(2.0 * s)) + 1, 25))
    out = cv2.bilateralFilter(guia, d, 0.04 + 0.08 * amount, s)
    for c in (1, 2):
        ch = out[..., c]
        if f > 1:
            ch = cv2.resize(ch, (w, h), interpolation=cv2.INTER_LINEAR)
        ycc[..., c] = ch


def _local_tones(img, shadows, highlights, region=None):
    """Luces y Sombras por zonas, como Lightroom.

    Antes eran parte de la curva unica: una curva que sube las sombras tiene
    que apretar lo que hay encima, y ademas trata igual un pixel oscuro de la
    textura de un jersey claro que una sombra de verdad. Medido en IMG_4188:
    Sombras +100 dejaba el 73 % del detalle de las sombras y Luces -100 el
    70 % del de las luces — el "gris lavado".

    Ahora se separa la foto en ZONAS (la luminancia suavizada respetando
    bordes, `_guided_self`) y DETALLE (lo que queda). La curva de luces y
    sombras mueve solo las zonas; el detalle se vuelve a poner encima intacto
    (y en sombras levantadas, crecido en la misma proporcion que la zona,
    para que no se aplane). Asi bajar Luces recupera las nubes en vez de
    volverlas gris, y subir Sombras abre la sombra sin lavarla.
    Trabaja con proporciones entre canales, asi que el color no se pierde."""
    lum = np.maximum(_luminance(img), 0.0).astype(np.float32)
    lado = max(region[0], region[1]) if region else max(img.shape[:2])
    # radio relativo al tamano de la foto ENTERA: la vista de detalle y la
    # exportacion ven las mismas zonas que la vista previa (y 2r cabe en el
    # margen de DETAIL_MARGIN a resolucion completa)
    zonas = _guided_self(lum, max(2.0, 0.007 * lado), 0.01)
    np.clip(zonas, 0.0, 1.0, out=zonas)
    curva = _tone_curve_lut(0.0, 0.0, shadows, highlights, 0.0, 0.0)
    nuevas = curva[_lut_index(zonas)]
    crece = np.clip(nuevas / np.maximum(zonas, 1e-3), 1.0, 3.0)
    if highlights < 0:
        # en textura muy marcada el filtro se lleva parte del detalle a la
        # zona y al bajarla se aplanaba algo (medido: 86-92 %). Bajar Luces
        # es para RECUPERAR, asi que en las luces el detalle se refuerza
        crece += (-highlights * 0.35) * np.clip((zonas - 0.4) / 0.6, 0.0, 1.0)
    lum_nueva = nuevas + (lum - zonas) * crece
    ganancia = np.maximum(lum_nueva, 0.0) / np.maximum(lum, 1e-4)
    img *= ganancia[..., None]
    return img


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


def _poly_weight(mask, h, w):
    """Mapa de peso de una mascara de puntos (cuadrado, rectangulo o libre).

    `points` son las esquinas en coordenadas 0..1, en orden. Cuatro para el
    rectangulo de siempre, pero valen las que sean: moviendo las esquinas por
    separado sale un trapecio, que es lo que hace falta para agarrar una
    ventana o una puerta vistas en perspectiva —rectangulos en la pared, pero
    nunca rectangulos en la foto.

    El desvanecido se mide en FRACCION del tamano de la mascara, no en
    pixeles. Asi el borde se ve igual de suave en la pantalla y en la foto
    exportada al doble de resolucion; con pixeles fijos, lo que en pantalla
    era una transicion amable salia como un corte al exportar.
    """
    pts = mask.get("points") or []
    if len(pts) < 3:
        return None
    poly = np.array([[float(x) * w, float(y) * h] for x, y in pts], np.float32)
    relleno = np.zeros((h, w), np.uint8)
    cv2.fillPoly(relleno, [np.round(poly).astype(np.int32)], 255)

    f = max(float(mask.get("feather", 50.0)) / 100.0, 0.0)
    if f <= 1e-3:
        return (relleno > 0).astype(np.float32)
    # tamano de referencia: el lado corto del rectangulo que la envuelve
    lado = max(min(np.ptp(poly[:, 0]), np.ptp(poly[:, 1])), 4.0)
    radio = max(lado * f * 0.5, 1.0)
    # Distancia con signo al borde: positiva dentro, negativa fuera.
    # DIST_MASK_PRECISE y no la mascara de 3x3: esa aproxima la distancia a
    # saltos de rey y deja rayos visibles saliendo de cada esquina cuando el
    # desvanecido es amplio.
    dentro = cv2.distanceTransform(relleno, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    fuera = cv2.distanceTransform(255 - relleno, cv2.DIST_L2,
                                  cv2.DIST_MASK_PRECISE)
    firmada = dentro - fuera
    return _smoothstep(firmada / (2.0 * radio) + 0.5).astype(np.float32)


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
    elif kind == "poly":
        wmap = _poly_weight(mask, h, w)
        if wmap is None:
            return None
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


def _mask_box(wmap, margin=8):
    """Rectangulo donde la mascara pesa algo (con un margen, para que los
    filtros de radio no se noten cortados en el borde).

    Es lo que hace que ajustar una mascara no cueste lo mismo que revelar la
    foto entera: fuera de ese rectangulo el peso es cero y aplicar la receta
    ahi seria calcular para multiplicar por 0. Devuelve None si la mascara
    esta vacia."""
    hit = wmap > 1e-4
    rows = np.flatnonzero(hit.any(axis=1))
    if not rows.size:
        return None
    cols = np.flatnonzero(hit.any(axis=0))
    h, w = wmap.shape
    return (max(int(rows[0]) - margin, 0), min(int(rows[-1]) + 1 + margin, h),
            max(int(cols[0]) - margin, 0), min(int(cols[-1]) + 1 + margin, w))


def mask_has_curve(mask):
    """Si esta mascara trae una curva que de verdad cambia algo.

    Una curva "sin tocar" es la diagonal: entra un tono y sale el mismo. Se
    compara convirtiendo a listas porque los puntos vienen del JSON y podrian
    llegar como tuplas o con enteros donde habia decimales.
    """
    puntos = (mask.get("adjust") or {}).get(MASK_CURVE_KEY)
    if not puntos or len(puntos) < 2:
        return False
    return [[float(x), float(y)] for x, y in puntos] != DEFAULT_CURVE


def _mask_recipe(sub, wmap, a):
    """Aplica la receta de UNA mascara sobre `sub` (vista de la foto, se
    modifica en el sitio), pesada por `wmap`.

    Mismas formulas que el panel general, para que un ajuste valga lo mismo
    dentro que fuera de una mascara, y cada efecto se mezcla con el peso:
    resultado = foto + (foto_con_el_efecto - foto) * peso."""
    wc = wmap[..., None]

    # 1. Tono: los seis mandos forman UNA curva, igual que arriba
    tone = {k: a.get(k, 0.0) for k in ("exposure", "contrast", "highlights",
                                       "shadows", "whites", "blacks")}
    if any(tone.values()):
        lut = _tone_curve_lut(exposure=tone["exposure"] / EXPOSURE_GAMMA,
                              contrast=tone["contrast"] / 100.0,
                              shadows=tone["shadows"] / 100.0,
                              highlights=tone["highlights"] / 100.0,
                              whites=tone["whites"] / 100.0,
                              blacks=tone["blacks"] / 100.0)
        toned = lut[_lut_index(sub)]
        toned -= sub
        toned *= wc
        sub += toned

    # 1b. Curva de tonos de la mascara. Va detras de los seis mandos, igual
    # que en el panel general: primero se coloca el tono y despues la curva
    # dibuja sobre lo ya colocado.
    if mask_has_curve({"adjust": a}):
        lut = pchip_lut(a[MASK_CURVE_KEY], LUT_N).astype(np.float32)
        curved = lut[_lut_index(sub)]
        curved -= sub
        curved *= wc
        sub += curved

    # 2. Balance de blancos
    temp = a.get("temperature", 0.0) / 100.0
    tint = a.get("tint", 0.0) / 100.0
    if temp:
        sub[..., 0] *= 1.0 + 0.30 * temp * wmap
        sub[..., 2] *= 1.0 - 0.30 * temp * wmap
    if tint:
        sub[..., 1] *= 1.0 - 0.20 * tint * wmap

    # 3. Neblina: quitarla en la zona (o anadirla, en negativo)
    dh = a.get("dehaze", 0.0) / 100.0
    if dh:
        if dh > 0:
            small = cv2.resize(sub, None, fx=0.25, fy=0.25,
                               interpolation=cv2.INTER_AREA)
            dark = cv2.GaussianBlur(small.min(axis=-1), (0, 0), 15 * 0.25)
            dark = cv2.resize(dark, (sub.shape[1], sub.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
            t = np.clip(dh * 0.7 * dark, 0.0, 0.9)[..., None]
            out = np.clip((sub - t) / (1.0 - t), 0.0, 1.0)
        else:
            k = -dh * 0.5
            out = sub * (1.0 - k) + k
        out -= sub
        out *= wc
        sub += out

    # 4. Color
    s = a.get("saturation", 0.0) / 100.0
    vib = a.get("vibrance", 0.0) / 100.0
    if s or vib:
        lum = _luminance(sub)[..., None]
        factor = 1.0 + s
        if vib:
            current = sub.max(-1) - sub.min(-1)
            factor = factor + vib * (1.0 - np.clip(current * 2.0, 0.0, 1.0))
            factor = factor[..., None]
        # lum + (sub - lum) * factor, pero solo lo que marque el peso
        sub -= lum
        sub *= 1.0 + (factor - 1.0) * wc
        sub += lum

    # 5. Detalle (los mismos filtros del panel Detalle, en la zona)
    clarity = a.get("clarity", 0.0) / 100.0
    if clarity:
        lum = _luminance(sub)
        detail = lum - cv2.GaussianBlur(lum, (0, 0), 30.0)
        protect = np.clip(1.0 - (2.0 * lum - 1.0) ** 2, 0.0, 1.0)
        detail *= protect
        detail *= clarity * 0.6
        sub += (detail * wmap)[..., None]
    tex = a.get("texture", 0.0) / 100.0
    if tex:
        blur = cv2.GaussianBlur(sub, (0, 0), 4.0)
        np.subtract(sub, blur, out=blur)
        blur *= tex * 0.9
        blur *= wc
        sub += blur
    sharp = a.get("sharp_amount", 0.0) / 100.0
    if sharp:
        high = sub - cv2.GaussianBlur(sub, (0, 0), 1.0)
        high *= sharp * 1.2
        high *= wc
        sub += high

    np.clip(sub, 0.0, 1.0, out=sub)


def _apply_masks(img, e, ai_masks=None, region=None):
    """Ajustes locales: cada mascara aplica su receta pesada por su mapa."""
    todo = [m for m in (e.get("masks") or [])
            if any((m.get("adjust") or {}).get(k) for k in MASK_ADJUST_KEYS)
            or mask_has_curve(m)]
    if not todo:
        return img
    h, w = img.shape[:2]
    img = np.ascontiguousarray(img)
    # Sin region, el trozo ES la foto entera. Con ella, las mascaras se
    # calculan al tamano de la foto completa y se recorta el pedazo que
    # toca: sus coordenadas van de 0 a 1 sobre la foto, no sobre el trozo
    gh, gw, oy, ox = region if region else (h, w, 0, 0)

    def get_ai(kind):
        m0 = ai_source_map(ai_masks, kind)
        return None if m0 is None else transform_ai_map(m0, e, gh, gw)

    for m in todo:
        wmap = mask_weight(m, gh, gw, get_ai)
        if wmap is None:
            continue  # mascara IA sin su mapa calculado todavia
        if region:
            wmap = np.ascontiguousarray(wmap[oy:oy + h, ox:ox + w])
        box = _mask_box(wmap)
        if box is None:
            continue  # mascara vacia (invertida sobre si misma, sin trazos...)
        y0, y1, x0, x1 = box
        _mask_recipe(img[y0:y1, x0:x1], wmap[y0:y1, x0:x1],
                     m.get("adjust") or {})
    return img


def _apply_profile(img, name, base_lut=None):
    """Interpretacion base del RAW. Todos los perfiles parten de la misma
    curva base (la del Estandar) y le suman su caracter encima, para que
    cambiar de perfil sea una variacion y no un salto de brillo.
    `base_lut`: curva base propia de esta foto (`camera_base_lut`); sin
    ella se usa la fija `PROFILE_BASE_CURVE`."""
    if name == "":
        return img
    recipe = PROFILE_RECIPES.get(name)
    if recipe is None:
        return img
    style, sat = recipe

    # base y estilo se componen en una sola tabla: estilo(base(x))
    if base_lut is not None:
        lut = base_lut
    else:
        lut = pchip_lut(PROFILE_BASE_CURVE, LUT_N).astype(np.float32)
    if style is not None:
        slut = pchip_lut(style, LUT_N).astype(np.float32)
        lut = slut[(lut * (LUT_N - 1) + 0.5).astype(np.uint16)]
    # canal a canal, como Camera Raw: con la curva fija medida ahi (ver
    # PROFILE_BASE_CURVE) el color de la piel sale igual que en Photoshop
    # (a 6,3 b 4,6 frente a 6,7 / 4,4 en IMG_4210). Conservar la proporcion
    # entre canales se probo y la dejaba naranja
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


def detail_geometry(full, edits):
    """La foto entera a resolucion completa, ya recortada y girada."""
    return _apply_geometry(full, {**DEFAULT_EDITS, **edits})


def detail_stats(geo, edits, is_raw=False, camera=None, base=None):
    """Medidas que solo tienen sentido mirando la foto ENTERA.

    Dos pasos del revelado no son locales: el arreglo automatico de
    exposicion y el mapeo tonal miran el brillo medio de toda la foto para
    decidir cuanto levantarla. Si cada trozo se midiera a si mismo, no
    pegarian entre ellos. Se calcula una vez, sobre la foto reducida (son
    promedios: no hace falta la resolucion completa), y se le pasa a cada
    trozo. Devuelve un diccionario para `apply_edits(stats=...)`.
    `camera` / `base`: tonos de la camara y la foto sin recortar, para la
    curva base de camara (ver `camera_base_lut`)."""
    e = {**DEFAULT_EDITS, **edits}
    img = geo
    out = {}
    profile = PROFILE_ALIASES.get(e.get("profile", "standard"),
                                  e.get("profile", "standard"))
    if is_raw and profile != RAW_PROFILE:
        # igual que en apply_edits: exposicion y curva fija tipo Camera Raw
        if e["exposure"]:
            img = np.clip(img * np.float32(2.0 ** (e["exposure"] / EXPOSURE_GAMMA)),
                          0.0, 1.0)
        img = _apply_profile(img, profile)
    if e.get("tone_map", 0.0) > 0:
        out["tm_log"] = log_avg_luminance(_apply_calibration(img, e))
    return out


# Margen que se revela de mas alrededor del trozo y luego se tira. Los filtros
# de radio (claridad mira a 30 px, enfoque, ruido) necesitan ver mas alla del
# borde; sin esto se notaria una linea justo donde acaba el trozo.
DETAIL_MARGIN = 64


def render_detail(geo, edits, box, is_raw=False, ai_masks=None,
                  stats=None, should_stop=None, margin=DETAIL_MARGIN,
                  wb_color=None):
    """Revela a resolucion completa SOLO el trozo que se esta mirando.

    `geo` es la foto completa ya pasada por `detail_geometry`, y `box` es el
    rectangulo visible (x0, y0, x1, y1) en coordenadas 0..1 sobre ella. Se
    revela ese trozo con un margen extra que se descarta al final.

    Devuelve (imagen uint8 del trozo, (x0, y0, x1, y1) en pixeles de `geo`),
    o None si el rectangulo se queda en nada."""
    gh, gw = geo.shape[:2]
    x0 = int(np.clip(box[0], 0.0, 1.0) * gw)
    y0 = int(np.clip(box[1], 0.0, 1.0) * gh)
    x1 = int(np.ceil(np.clip(box[2], 0.0, 1.0) * gw))
    y1 = int(np.ceil(np.clip(box[3], 0.0, 1.0) * gh))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    # trozo ampliado con el margen (recortado contra los bordes de la foto)
    mx0, my0 = max(x0 - margin, 0), max(y0 - margin, 0)
    mx1, my1 = min(x1 + margin, gw), min(y1 + margin, gh)
    sub = np.ascontiguousarray(geo[my0:my1, mx0:mx1])

    out = apply_edits(sub, edits, ai_masks=ai_masks, is_raw=is_raw,
                      should_stop=should_stop, region=(gh, gw, my0, mx0),
                      stats=stats, wb_color=wb_color)
    # fuera el margen: se revelo solo para que los filtros vieran el entorno
    out = out[y0 - my0:y0 - my0 + (y1 - y0), x0 - mx0:x0 - mx0 + (x1 - x0)]
    return np.ascontiguousarray(out), (x0, y0, x1, y1)


class Stale(Exception):
    """Este revelado ya no vale: mientras se calculaba, has vuelto a mover
    algo y hay uno mas nuevo esperando."""


def _check(stop):
    """Punto de control: suelta el trabajo si el resultado ya no sirve.

    Un revelado de calidad son varias decimas de segundo (en una foto con
    enfoque, claridad y HSL puestos, mas de un segundo). Sin estos cortes,
    mover un ajuste mientras uno esta en marcha obligaba a esperar a que
    terminase entero... para tirarlo a la basura al llegar, porque ya habia
    quedado viejo. Ahora se abandona en el primer control."""
    if stop is not None and stop():
        raise Stale()


def apply_edits(base, edits, ai_masks=None, denoised=None, faced=None,
                is_raw=False, should_stop=None, region=None, stats=None,
                camera=None, wb_color=None):
    """base: float32 RGB en rango 0..1. Devuelve uint8 RGB listo para mostrar.
    `wb_color`: `wb.CameraColor` del RAW, para el balance en Kelvin.
    `camera`: como revelo la camara este RAW (`loader.camera_tones`); si se
    da, la curva base reproduce ese revelado en vez del aclarado automatico.
    `ai_masks`: mapas de segmentacion IA pre-geometria, p. ej. {"subject": m}.
    `denoised` / `faced`: resultados IA pre-geometria (misma forma que base),
    para las mascaras con ruido IA / rostros IA locales.
    `should_stop`: funcion que, si devuelve True, aborta con `Stale`.
    `region`: (alto, ancho, y0, x0) de la foto entera cuando `base` es solo
    un trozo de ella (vista de detalle). El que llama ya le ha aplicado la
    geometria; aqui sirve para que las mascaras caigan donde deben.
    `stats`: medidas tomadas sobre la foto ENTERA (ver `detail_stats`), para
    que un trozo se revele igual que si fuera parte de ella."""
    e = {**DEFAULT_EDITS, **edits}
    st = stats or {}
    if region is None:
        img = _apply_geometry(base, e)
    else:
        # el trozo ya llega recortado y girado (lo hizo `render_detail` sobre
        # la foto entera). Aqui NO se puede reutilizar la bandera
        # `_skip_geometry`: `transform_ai_map` la leeria y dejaria los mapas
        # de IA sin girar, descuadrados respecto al trozo
        img = np.ascontiguousarray(base)
        if np.may_share_memory(img, base):
            img = img.copy()   # apply_edits pinta encima; nunca sobre la base
    if denoised is not None or faced is not None:
        img = _blend_ai_masked(img, e, ai_masks, denoised, faced, base)
    _check(should_stop)

    # Perfil base y calibracion de camara: primero, definen el punto
    # de partida sobre el que actua todo lo demas
    # Solo se aplica a RAW (JPEGs ya tienen el procesamiento de camara), y
    # con el perfil "RAW" tampoco: ese pide el archivo sin interpretar
    profile = e.get("profile", "standard")
    profile = PROFILE_ALIASES.get(profile, profile)
    tone_exposure = e["exposure"]
    # Balance de blancos en Kelvin: lo primero, sobre la luz del RAW recien
    # revelado, igual que si la camara hubiera disparado con ese balance
    if is_raw and wb_color is not None and e.get("wb_temp"):
        from photoraw import wb
        img = wb.apply(img, wb_color.correction(e["wb_temp"],
                                                e.get("wb_tint", 0.0)))
    if is_raw and profile != RAW_PROFILE:
        # Como Camera Raw: exposicion base del DNG (ya puesta al abrir) y
        # una curva FIJA, la misma para todas las fotos. Antes se copiaba el
        # JPEG que la camara mete dentro del RAW (`camera_base_lut`), pero
        # ese JPEG ya viene procesado por zonas (la app Moment/el iPhone
        # aclaran la cara aparte), y para imitarlo con una sola curva habia
        # que levantar las sombras x8: piel naranja, velo y ruido (IMG_4210
        # frente a Photoshop, 2026-09-23). `camera` ya no se usa aqui.
        if e["exposure"]:
            # la exposicion va ANTES de la curva, como en Lightroom: en una
            # toma quemada la curva ya aplasta las luces, y bajar la
            # exposicion despues solo las volvia un gris liso (IMG_4187)
            img = img * np.float32(2.0 ** (e["exposure"] / EXPOSURE_GAMMA))
            np.clip(img, 0.0, 1.0, out=img)
            tone_exposure = 0.0
        img = _apply_profile(img, profile)
    img = _apply_calibration(img, e)
    _check(should_stop)

    # Mapeo tonal adaptativo: comprime el rango dinamico del RAW de forma
    # inteligente, preservando contraste local (como Lightroom / Capture One)
    tm = e.get("tone_map", 0.0) / 100.0
    if tm > 0:
        img = img * (1.0 - tm) + tone_mapping(img, log_avg=st.get("tm_log")) * tm

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
        exposure=tone_exposure / EXPOSURE_GAMMA,
        contrast=e["contrast"] / 100.0,    # -1..1
        # Luces y Sombras van aparte, por zonas (ver `_local_tones`)
        shadows=0.0,
        highlights=0.0,
        whites=e["whites"] / 100.0,        # -1..1
        blacks=e["blacks"] / 100.0,        # -1..1
    )
    # la misma curva para los tres canales: se lee la tabla de una pasada en
    # vez de canal a canal (leer `idx[..., c]` va salteado por la memoria)
    img = tone_lut[_lut_index(img)]
    if e["shadows"] or e["highlights"]:
        img = _local_tones(img, e["shadows"] / 100.0, e["highlights"] / 100.0,
                           region)
        np.clip(img, 0.0, 1.0, out=img)

    _check(should_stop)

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

    _check(should_stop)

    # Contraste adaptativo local (CLAHE): mejora el contraste en sombras
    # e iluminaciones sin lavar la imagen general
    ac = e.get("adaptive_contrast", 0.0) / 100.0
    if ac > 0:
        # CLAHE con intensidad variable: suave (clip_limit 2.0) a fuerte (4.0)
        img = img * (1.0 - ac) + adaptive_contrast(img, clip_limit=2.0 + ac * 2.0) * ac

    _check(should_stop)

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

    _check(should_stop)

    # Mezclador HSL y color de punto
    img = _apply_hsl(img, e)
    _check(should_stop)

    # Mascaras con ajustes locales (degradados, pincel, sujeto/fondo IA)
    img = _apply_masks(img, e, ai_masks, region)
    _check(should_stop)

    # Vista previa de la mascara de enfoque (Alt sobre el deslizador Mascara):
    # blanco = zonas que reciben enfoque, negro = protegidas
    if e.get("_show_mask"):
        img = np.clip(img, 0.0, 1.0)
        mask = _sharpen_mask(img, e["sharp_masking"] / 100.0)
        gray = (mask * 255.0 + 0.5).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)

    # En el borrador de edicion rapida se omiten los efectos de detalle
    # (ruido, textura, enfoque, grano): no se aprecian mientras se arrastra
    # un ajuste tonal y cuestan la mitad del render. Los creativos SI se
    # calculan siempre: son el efecto que estas mirando, saltarselos haria
    # que la foto cambiara de aspecto al soltar el raton.
    draft = bool(e.get("_draft_skip_detail"))
    if draft:
        img = _apply_creative(img, e, should_stop, region)
        return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    _check(should_stop)

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
            _chroma_nr(ycc, nc, region)
        img = np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)

    _check(should_stop)

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

    _check(should_stop)

    # Enfoque: cantidad / radio / detalle / mascara
    if e["sharp_amount"]:
        amount = e["sharp_amount"] / 100.0 * 1.2
        sigma = max(float(e["sharp_radius"]), 0.3)
        high = img - cv2.GaussianBlur(img, (0, 0), sigma)
        # Detalle bajo = solo bordes fuertes (evita amplificar el ruido fino).
        # El umbral se resta al alto-paso, asi que hay que medirlo contra lo
        # que de verdad mide el detalle fino de una foto: la mitad de los
        # pixeles no llegan a 2 niveles sobre 255. La constante estaba en
        # 0.03 (7,6 niveles): se comia el 89% del enfoque con el Detalle de
        # fabrica y por eso "enfocar" casi no se notaba. A 0.012 el tope son
        # 3 niveles, que sigue frenando el ruido sin borrar la textura.
        detail = e["sharp_detail"] / 100.0
        thr = (1.0 - detail) ** 2 * 0.012
        if thr > 0:
            high = np.sign(high) * np.maximum(np.abs(high) - thr, 0.0)
        # Mascara: limita el enfoque a las zonas con bordes
        if e["sharp_masking"]:
            mask = _sharpen_mask(img, e["sharp_masking"] / 100.0)
            high *= mask[..., None]
        high *= amount
        img += high
        np.clip(img, 0.0, 1.0, out=img)

    img = _apply_creative(img, e, should_stop, region)

    img *= 255.0
    img += 0.5
    return img.astype(np.uint8)


# Estado de animo: cada receta es una interpretacion de color completa, del
# tipo que en cine se llama "look". Se define con la ganancia y la gamma de
# cada canal (rojo, verde, azul) mas un empujon de saturacion: la ganancia
# tine sobre todo las luces y la gamma sobre todo los medios y las sombras.
MOOD_RECIPES = {
    "none":     None,
    "warm":     ((1.07, 1.01, 0.93), (0.97, 1.00, 1.05), 1.06),
    "cool":     ((0.94, 1.00, 1.09), (1.05, 1.01, 0.96), 1.02),
    "cine":     ((1.05, 1.00, 0.95), (1.04, 1.00, 0.93), 1.10),
    "sepia":    ((1.10, 1.00, 0.84), (0.95, 1.00, 1.10), 0.45),
    "forest":   ((0.93, 1.04, 0.97), (1.06, 0.96, 1.02), 1.08),
    "sunset":   ((1.12, 0.98, 0.88), (0.94, 1.02, 1.08), 1.12),
    "night":    ((0.92, 0.97, 1.12), (1.10, 1.04, 0.94), 0.88),
}

GLOW_MODES = ("soft_focus", "glow", "orton", "orton_soft")


def _blur_rel(img, sigma, region=None):
    """Difuminado con radio relativo al tamano de la foto.

    Un radio fijo en pixeles daria un efecto distinto en la vista previa que
    en el archivo exportado (que tiene el doble de lado). Aqui 'sigma' es el
    radio a 2200 px de lado y se escala solo, igual que hace el grano."""
    h, w = img.shape[:2]
    gh, gw = region[:2] if region else (h, w)
    s = max(sigma * max(gh, gw) / 2200.0, 0.3)
    return cv2.GaussianBlur(img, (0, 0), s)


def _dramatic(img, e, region):
    """Look dramatico: contraste local fuerte y color contenido.

    Es el 'Dramatico' de Luminar: saca la textura de las nubes, la piedra o
    la piel curtida subiendo el contraste de radio medio y bajando un poco el
    color, que es lo que le da ese aire duro de reportaje."""
    amt = e["dramatic_amount"] / 100.0
    extra = e["dramatic_contrast"] / 100.0
    bright = e["dramatic_brightness"] / 100.0
    sat = e["dramatic_saturation"] / 100.0
    if not (amt or extra or bright or sat):
        return img
    if amt or extra:
        lum = _luminance(img)
        # dos radios: uno amplio que da cuerpo y otro corto que saca textura
        detail = lum - _blur_rel(lum, 24.0, region)
        boost = detail * (amt * 1.1)
        if extra:
            fino = lum - _blur_rel(lum, 6.0, region)
            boost = boost + fino * (extra * 0.9)
        img += boost[..., None]
        if amt:
            # curva en S suave: mas cuerpo sin cerrar los negros del todo
            img += (img - 0.5) * (0.30 * amt)
        np.clip(img, 0.0, 1.0, out=img)
    if bright:
        img += bright * 0.18
        np.clip(img, 0.0, 1.0, out=img)
    # el dramatico desatura por su cuenta; el deslizador lo corrige a gusto
    total_sat = sat - 0.35 * amt
    if total_sat:
        lum = _luminance(img)[..., None]
        img -= lum
        img *= 1.0 + total_sat
        img += lum
        np.clip(img, 0.0, 1.0, out=img)
    return img


def _mood(img, e):
    """Estado de animo: la receta de color elegida, mezclada a gusto."""
    recipe = MOOD_RECIPES.get(e.get("mood_preset", "none"))
    amt = e["mood_amount"] / 100.0
    if recipe is None or amt <= 0:
        return img
    gain, gamma, sat = recipe
    # una tabla por canal: gamma primero (medios y sombras), ganancia despues
    x = np.linspace(0.0, 1.0, LUT_N, dtype=np.float32)
    luts = []
    for c in range(3):
        curva = np.clip(np.power(x, gamma[c]) * gain[c], 0.0, 1.0)
        luts.append((1.0 - amt) * x + amt * curva)   # mezcla con el original
    idx = _lut_index(img)
    for c in range(3):
        img[..., c] = luts[c][idx[..., c]]
    if sat != 1.0:
        s = 1.0 + (sat - 1.0) * amt
        lum = _luminance(img)[..., None]
        img -= lum
        img *= s
        img += lum
        np.clip(img, 0.0, 1.0, out=img)
    return img


def _hue_rgb(deg):
    """Color puro de un tono del circulo cromatico, en RGB 0..1."""
    hsv = np.array([[[deg / 2.0, 255, 255]]], np.uint8)   # OpenCV: 0..179
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32) / 255.0
    return rgb


def _toning(img, e):
    """Virado: un color en las luces y otro en las sombras.

    El virado de toda la vida (sepia en las luces, frio en las sombras). El
    equilibrio mueve la frontera entre lo que cuenta como luz y como sombra."""
    hs = e["tone_hi_sat"] / 100.0
    ss = e["tone_sh_sat"] / 100.0
    if not (hs or ss):
        return img
    lum = np.clip(_luminance(img), 0.0, 1.0)
    bal = np.clip(e["tone_balance"] / 100.0, -0.9, 0.9)
    # el equilibrio desplaza el punto medio: +100 tine mas zona como luz
    pivot = 0.5 - bal * 0.35
    hi_w = np.clip((lum - pivot) / max(1.0 - pivot, 1e-3), 0.0, 1.0) ** 1.5
    sh_w = np.clip((pivot - lum) / max(pivot, 1e-3), 0.0, 1.0) ** 1.5
    # Se suma SOLO la parte de color del tono, no el color entero: restarle su
    # propia luminancia lo deja con media cero, asi tine sin cambiar el brillo
    # ni tapar la foto. (Empujar el pixel hacia el color puro repintaba el
    # tejado de azul solido en vez de virarlo.)
    if hs:
        croma = _hue_rgb(e["tone_hi_hue"])
        croma = croma - float(croma @ np.array([0.2126, 0.7152, 0.0722],
                                               np.float32))
        img += croma * (hi_w * hs * 0.30)[..., None]
    if ss:
        croma = _hue_rgb(e["tone_sh_hue"])
        croma = croma - float(croma @ np.array([0.2126, 0.7152, 0.0722],
                                               np.float32))
        img += croma * (sh_w * ss * 0.30)[..., None]
    np.clip(img, 0.0, 1.0, out=img)
    return img


def _matte(img, e):
    """Mate: negros levantados y luces recogidas, como una copia antigua.

    Es la curva desvaida del papel viejo o del cine en Super 8: el negro deja
    de ser negro y la foto pierde el punto duro."""
    amt = e["matte_amount"] / 100.0
    if amt <= 0:
        return img
    fade = e["matte_fade"] / 100.0
    contrast = e["matte_contrast"] / 100.0
    lift = 0.16 * fade * amt        # suelo nuevo del negro
    roll = 0.06 * amt               # techo, un poco por debajo del blanco
    img *= (1.0 - lift - roll)
    img += lift
    if contrast:
        img += (img - (lift + (1.0 - roll)) * 0.5) * (0.5 * contrast)
    np.clip(img, 0.0, 1.0, out=img)
    return img


def _mystical(img, e, region):
    """Mistico: luz difusa de ensueno que se derrama por la foto.

    Las luces se difuminan y se derraman sobre lo que tienen alrededor. Con
    'Sombras' se controla cuanto invade las zonas oscuras (que es lo que da
    el aire de niebla) y con 'Suavizado' cuanto detalle fino se lima."""
    amt = e["mystical_amount"] / 100.0
    if amt <= 0:
        return img
    shadows = e["mystical_shadows"] / 100.0
    smooth = e["mystical_smooth"] / 100.0
    soft = _blur_rel(img, 26.0, region)
    lum = np.clip(_luminance(img), 0.0, 1.0)
    # por defecto entra sobre todo en las luces; 'Sombras' lo reparte
    peso = (lum ** 1.5) * (1.0 - shadows) + shadows
    peso = (peso * amt)[..., None]
    # mezcla en modo "trama": suma luz sin ensuciar, como un velo luminoso
    img += (1.0 - img) * soft * peso
    if smooth:
        # positivo lima el detalle fino, negativo lo devuelve
        fino = img - _blur_rel(img, 3.0, region)
        img -= fino * (smooth * 0.7)
    np.clip(img, 0.0, 1.0, out=img)
    return img


def _glow(img, e, region):
    """Brillo: los cuatro clasicos del difuminado.

    - Enfoque suave: la foto difuminada por encima, como una media delante
      del objetivo. Suaviza la piel sin perder la forma.
    - Brillo: solo las luces se difuminan y se derraman (halo luminoso).
    - Efecto Orton: la copia nitida y la difuminada multiplicadas — colores
      densos, luces que respiran. El clasico de paisaje.
    - Efecto Orton suave: el mismo con la mano mas ligera."""
    amt = e["glow_amount"] / 100.0
    if amt <= 0:
        return img
    modo = e.get("glow_mode", "orton")
    radio = 6.0 + e["glow_smooth"] / 100.0 * 26.0
    soft = _blur_rel(img, radio, region)
    if modo == "soft_focus":
        img += (soft - img) * (amt * 0.85)
    elif modo == "glow":
        lum = np.clip(_luminance(soft), 0.0, 1.0)
        halo = soft * (lum ** 2)[..., None]
        img += (1.0 - img) * halo * (amt * 0.9)
    else:
        # Orton: multiplicar la nitida por la difuminada oscurece y satura;
        # luego se recupera luz en modo trama para que no se cierre
        fuerza = amt * (0.55 if modo == "orton_soft" else 0.85)
        mult = img * soft
        screen = 1.0 - (1.0 - img) * (1.0 - soft)
        mezcla = mult * 0.5 + screen * 0.5
        img += (mezcla - img) * fuerza
        if modo == "orton":
            lum = _luminance(img)[..., None]
            img -= lum
            img *= 1.0 + 0.15 * amt     # el Orton clasico satura un punto
            img += lum
    np.clip(img, 0.0, 1.0, out=img)
    return img


def _apply_creative(img, e, should_stop=None, region=None):
    """Efectos creativos: los que se VEN mientras los ajustas.

    Van aparte del bloque de detalle porque el borrador rapido tambien los
    calcula: si el grano desapareciera al arrastrar y volviera al soltar, la
    foto cambiaria de aspecto delante de tus ojos."""
    _check(should_stop)
    img = _dramatic(img, e, region)
    _check(should_stop)
    img = _mood(img, e)
    img = _toning(img, e)
    img = _matte(img, e)
    _check(should_stop)
    img = _mystical(img, e, region)
    img = _glow(img, e, region)
    _check(should_stop)

    # Grano de pelicula: monocromatico, mas fuerte en tonos medios (como el
    # grano real de negativo), deterministico para que no "hierva" al editar
    if e["grain_amount"]:
        h, w = img.shape[:2]
        # tamano de celda relativo a la resolucion: mismo aspecto en la vista
        # previa y en la exportacion a resolucion completa. En la vista de
        # detalle manda el tamano de la foto entera, no el del trozo
        gh, gw = region[:2] if region else (h, w)
        rel = max(gh, gw) / 2200.0
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

    return img
