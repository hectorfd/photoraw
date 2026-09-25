"""Pincel corrector: inpainting con LaMa (ONNX + CUDA).

El usuario pinta una mascara y LaMa rellena la zona con contenido coherente
con el entorno. El modelo trabaja a 512x512 fijo, asi que se recorta la zona
con contexto alrededor, se procesa y se reintegra con borde desvanecido.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from photoraw.ai import MODEL_DIR, _make_options, _providers, _setup_dll_paths
from photoraw.generative import _sharpen_patch

LAMA_MODEL = MODEL_DIR / "lama_fp32.onnx"
LAMA_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"

_session = None


def model_available():
    return LAMA_MODEL.exists() and LAMA_MODEL.stat().st_size > 50_000_000


def download_model(progress_cb=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LAMA_MODEL.with_suffix(".onnx.tmp")

    def hook(blocks, block_size, total):
        if progress_cb and total > 0:
            progress_cb(min(blocks * block_size / total, 1.0))

    urllib.request.urlretrieve(LAMA_URL, tmp, hook)
    tmp.replace(LAMA_MODEL)


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        _setup_dll_paths()
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        _session = ort.InferenceSession(
            str(LAMA_MODEL), sess_options=_make_options(),
            providers=_providers())
    return _session


def rasterize_strokes(strokes, h, w):
    """Convierte trazos normalizados [[radio, [[x, y], ...]], ...] en una
    mascara uint8 (255 = borrar) del tamano pedido."""
    mask = np.zeros((h, w), np.uint8)
    scale = max(h, w)
    for radius, points in strokes:
        thickness = max(int(radius * scale * 2), 3)
        pts = np.array([[int(x * w), int(y * h)] for x, y in points], np.int32)
        if len(pts) == 1:
            cv2.circle(mask, tuple(pts[0]), max(thickness // 2, 2), 255, -1)
        else:
            cv2.polylines(mask, [pts], False, 255, thickness=thickness,
                          lineType=cv2.LINE_8)
    return mask


# trazos menores a esta fraccion del lado mayor toleran el relleno clasico
# de respaldo cuando el modelo LaMa no esta descargado (ver needs_model)
FAST_FRAC = 0.06


def _is_small(y0, y1, x0, x1, h, w):
    # los pequenos se curan copiando textura (_patch_heal), sin modelo
    return max(y1 - y0, x1 - x0) <= min(FAST_FRAC, PATCH_FRAC) * max(h, w)


def needs_model(strokes, h, w):
    """True si algun trazo es lo bastante grande para requerir LaMa."""
    mask = rasterize_strokes(strokes, h, w)
    n_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    for i in range(1, n_labels):
        ys, xs = np.where(labels == i)
        if not _is_small(ys.min(), ys.max(), xs.min(), xs.max(), h, w):
            return True
    return False


def _blend_mask(mcrop):
    """Desvanecido del borde del parche, en proporcion AL PROPIO PARCHE.

    Con un desvanecido fijo en pixeles (era de 4), un retoque impecable en
    pantalla salia con el borde marcado al exportar: se edita a 2200 px de
    ancho y se exporta a 4032, asi que esos 4 pixeles pasaban de ser el
    0,18 % del lado al 0,099 %. Media transicion, y el contorno asomando.

    La medida buena es el tamano del parche, que ya escala solo porque los
    trazos se guardan en coordenadas normalizadas. Lo que NO vale es atarlo
    al lado de la foto: probado, dejaba el desvanecido mas ancho que los
    retoques pequenos, y entonces el centro del parche se quedaba a medio
    aplicar (al 54 % en una mota de 20 px) con el defecto transparentandose
    por debajo. De ahi el aspecto sucio y difuso.

    Por eso ademas se ensancha el nucleo antes de difuminarlo: garantiza que
    donde pintaste el parche entra AL COMPLETO y la transicion cae por fuera,
    que es lo que hace falta para que el defecto no vuelva a asomar.
    """
    ys, xs = np.where(mcrop > 0)
    if not ys.size:
        return np.zeros(mcrop.shape + (1,), np.float32)
    lado = max(ys.max() - ys.min(), xs.max() - xs.min(), 1)
    sigma = float(np.clip(lado * 0.12, 1.5, 24.0))
    radio = int(max(1, round(sigma)))
    nucleo = cv2.dilate((mcrop > 0).astype(np.float32),
                        np.ones((radio * 2 + 1, radio * 2 + 1), np.uint8))
    return cv2.GaussianBlur(nucleo, (0, 0), sigma)[..., None]


def _fast_inpaint_region(img, mask, y0, y1, x0, x1):
    """Relleno clasico instantaneo (Telea) para manchas pequenas."""
    crop = img[y0:y1, x0:x1]
    mcrop = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    hard = cv2.dilate(mcrop, np.ones((3, 3), np.uint8))
    crop8 = (np.clip(crop, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    out8 = cv2.inpaint(crop8, hard, 5, cv2.INPAINT_TELEA)
    out = out8.astype(np.float32) / 255.0
    blend = _blend_mask(mcrop)
    img[y0:y1, x0:x1] = crop * (1.0 - blend) + out * blend


def _smooth_offset(diff, weight, sigma):
    """Rellena hacia dentro, suave, una diferencia que solo se conoce donde
    weight > 0 (el anillo de piel alrededor del parche): convolucion
    normalizada. Da el tono/color que le falta al parche en cada punto."""
    w = cv2.GaussianBlur(weight.astype(np.float32), (0, 0), sigma)
    out = np.empty_like(diff)
    for c in range(diff.shape[2]):
        out[..., c] = cv2.GaussianBlur(diff[..., c] * weight, (0, 0), sigma)
    return out / np.maximum(w, 1e-6)[..., None]


def _ring(core, width):
    k = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    return (cv2.dilate(core, k) > 0) & (core == 0)


# Parches hasta este tamano (fraccion del lado mayor) se curan copiando piel
# real de al lado, como el pincel corrector de Lightroom; los mayores van a
# LaMa, que sabe reconstruir cosas que no estan en la foto
PATCH_FRAC = 0.05
LAMA_GAMMA = 2.2
# medido: granitos en piel 0,20-0,48, junto a la comisura del labio 0,66;
# junto a juntas de baldosa 0,59-0,92 (IMG_4205 / IMG_4170). Junto a una
# linea tambien se copia (la fuente buena queda A LO LARGO de la linea y el
# anillo la encuentra); solo las lineas muy marcadas van a LaMa. Con el corte
# en 0,5 la comisura iba a LaMa y salia la mancha lisa y anaranjada
COHERENCE_MAX = 0.75


def _structure(win, side):
    """Tensor de estructura de la ventana (jxx, jyy, jxy por pixel), para
    medir lineas: sumado en una zona, su parte 'con direccion' es lo que
    tienen de lineas (juntas, bordes) y la traza, todo el detalle."""
    lum = win @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    # fuera el sombreado suave (la curva de una mejilla tambien "tiene
    # direccion"): solo cuentan las lineas finas
    lum = lum - cv2.GaussianBlur(lum, (0, 0), max(side / 3.0, 2.0))
    lum = cv2.GaussianBlur(lum, (0, 0), max(side / 12.0, 1.0))
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1)
    return gx * gx, gy * gy, gx * gy


def _aniso(J, ys, xs):
    """(energia con direccion, energia total) de los pixeles dados."""
    jxx, jyy, jxy = (float(j[ys, xs].sum()) for j in J)
    return np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2), jxx + jyy + 1e-12


def _patch_heal(img, all_mask, comp, y0, y1, x0, x1):
    """Corrector clasico: busca cerca una zona que 'encaje' (misma textura
    alrededor), la copia encima y le corrige tono y color para que empalme
    con los bordes. Devuelve False si no encuentra de donde copiar.

    Por que: LaMa en piel oscura y con ruido inventa manchas de otro tono
    (amarillentas, verdosas) y sin grano, que al aclarar la foto al revelar
    se ven como parches de plastico. Copiando piel de verdad, la textura y
    el grano son los mismos del resto de la cara."""
    h, w = img.shape[:2]
    side = max(y1 - y0, x1 - x0) + 1
    feather = int(np.clip(side * 0.12, 1.5, 24.0)) + 1
    ring_w = max(3, side // 3)
    # margen de la ventana: cabe la cola del desvanecido (~3 sigmas) y el anillo
    m = 3 * feather + ring_w + 2
    reach = int(side * 3.2) + m                   # hasta donde busca fuente
    wy0, wy1 = max(y0 - reach, 0), min(y1 + 1 + reach, h)
    wx0, wx1 = max(x0 - reach, 0), min(x1 + 1 + reach, w)
    win = img[wy0:wy1, wx0:wx1]
    busy = cv2.dilate((all_mask[wy0:wy1, wx0:wx1] > 0).astype(np.uint8),
                      np.ones((2 * feather + 3,) * 2, np.uint8))

    # la ventana del parche (con su desvanecido y su anillo), en coords de win
    ty0, ty1 = max(y0 - m, wy0) - wy0, min(y1 + 1 + m, wy1) - wy0
    tx0, tx1 = max(x0 - m, wx0) - wx0, min(x1 + 1 + m, wx1) - wx0
    core = (comp[wy0:wy1, wx0:wx1][ty0:ty1, tx0:tx1] > 0).astype(np.uint8)
    cover = cv2.dilate(core, np.ones((2 * feather + 1,) * 2, np.uint8)) > 0
    ring = _ring(cover.astype(np.uint8), ring_w) & (busy[ty0:ty1, tx0:tx1] == 0)
    if ring.sum() < 20:
        return False

    # se compara textura (paso alto), no tono: el tono se corrige despues
    sig = max(side / 4.0, 1.5)
    hp = win - cv2.GaussianBlur(win, (0, 0), sig)
    ry, rx = np.nonzero(ring)

    # Copiar solo vale en zonas sin lineas (piel, cielo, pared lisa). Si
    # alrededor hay una direccion dominante (juntas de baldosa, un borde),
    # copiar pega trozos de linea sueltos y LaMa lo resuelve mejor
    J = _structure(win, side)
    lin, tot = _aniso(J, ry + ty0, rx + tx0)
    if lin / tot > COHERENCE_MAX:
        return False
    lin_ref = lin / len(ry)                       # lineas "normales" del entorno
    tgt = hp[ty0:ty1, tx0:tx1][ry, rx]
    lum = win @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    tgt_lum = float(lum[ty0:ty1, tx0:tx1][ring].mean())
    escala = float(np.mean(tgt ** 2)) + 1e-8
    ch, cw = ty1 - ty0, tx1 - tx0
    blend = _blend_mask(core)
    # todo lo que la copia llega a tocar, cola del desvanecido incluida: una
    # linea en la fuente, aunque sea en el borde, asoma como rayita suelta
    cy, cx = np.nonzero(blend[..., 0] > 0.02)

    mejor, coste_min = None, np.inf
    for dist in (1.1, 1.5, 2.0, 2.6, 3.2):
        for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            dy = int(round(np.sin(ang) * dist * side))
            dx = int(round(np.cos(ang) * dist * side))
            sy0, sx0 = ty0 + dy, tx0 + dx
            if sy0 < 0 or sx0 < 0 or sy0 + ch > win.shape[0] or sx0 + cw > win.shape[1]:
                continue
            if busy[sy0 + cy, sx0 + cx].any():    # la fuente no puede estar retocada
                continue
            src = hp[sy0:sy0 + ch, sx0:sx0 + cw]
            coste = float(np.mean((src[ry, rx] - tgt) ** 2)) / escala
            # la fuente no debe traer lineas que el entorno no tiene: copiaba
            # trozos de juntas de baldosa, rayitas sueltas donde no habia nada
            lin_src = _aniso(J, sy0 + cy, sx0 + cx)[0] / len(cy)
            coste += 4.0 * max(np.log((lin_src + 1e-9) / (lin_ref + 1e-9))
                               - np.log(1.5), 0.0)
            # ni ser de una zona de luz muy distinta (otra parte de la cara)
            src_lum = float(lum[sy0:sy0 + ch, sx0:sx0 + cw][ring].mean())
            coste += 2.0 * abs(np.log((src_lum + 0.01) / (tgt_lum + 0.01)))
            coste += 0.04 * dist                  # a igualdad, lo mas cerca
            if coste < coste_min:
                coste_min, mejor = coste, (sy0, sx0)
    if mejor is None:
        return False

    sy0, sx0 = mejor
    tgt_img = win[ty0:ty1, tx0:tx1]
    src_img = win[sy0:sy0 + ch, sx0:sx0 + cw]
    # tono y color: la diferencia en el anillo, extendida suave hacia dentro
    ajuste = _smooth_offset(tgt_img - src_img, ring.astype(np.float32),
                            max(side * 0.35, 2.0))
    parche = src_img + ajuste
    gy0, gx0 = wy0 + ty0, wx0 + tx0
    img[gy0:gy0 + ch, gx0:gx0 + cw] = tgt_img * (1.0 - blend) + parche * blend
    return True


def _inpaint_region(img, mask, y0, y1, x0, x1):
    """Procesa un recorte y lo funde de vuelta en img (in place)."""
    sess = _get_session()
    names = [i.name for i in sess.get_inputs()]
    crop = img[y0:y1, x0:x1]
    mcrop = mask[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]

    # LaMa aprendio con fotos ya reveladas. La base de un RAW esta oscura
    # (piel a ~0,1) y ahi inventaba manchas amarillentas sin grano que el
    # revelado luego aclaraba x3. Se le da la zona "revelada" a ojo (a su
    # blanco y con gamma) y el resultado se devuelve al espacio de la base
    ref = max(float(np.percentile(crop, 99.5)), 1e-3)
    crop_n = np.clip(crop / ref, 0.0, 1.0) ** (1.0 / LAMA_GAMMA)
    crop512 = cv2.resize(crop_n, (512, 512), interpolation=cv2.INTER_AREA)
    mask512 = cv2.resize(mcrop, (512, 512), interpolation=cv2.INTER_NEAREST)
    mask512 = cv2.dilate(mask512, np.ones((5, 5), np.uint8))

    x_img = np.ascontiguousarray(crop512.transpose(2, 0, 1)[None], dtype=np.float32)
    x_mask = (mask512 > 127).astype(np.float32)[None, None]
    out = sess.run(None, {names[0]: x_img, names[1]: x_mask})[0][0]
    out = out.transpose(1, 2, 0)
    if float(out.max()) > 2.0:  # este export devuelve 0..255
        out = out / 255.0
    out = np.clip(out, 0.0, 1.0)

    # en fotos grandes (sobre todo al exportar a resolucion completa) el
    # recorte puede ser mucho mayor a 512: si se estira con interpolacion
    # simple, el parche sale borroso y "inventado" frente al detalle real
    # de alrededor. Se re-amplia con Real-ESRGAN, igual que el borrado
    # generativo, para que la nitidez combine con el resto de la foto.
    out = _sharpen_patch(out, cw, ch)
    out = np.clip(out, 0.0, 1.0) ** LAMA_GAMMA * ref

    # El modelo rellena la ventana ENTERA, no solo lo pintado, asi que fuera
    # de la mascara su respuesta deberia coincidir con la foto. Lo que se
    # desvie ahi es el sesgo que trae el parche (viene de haber pasado por
    # 512x512), y es el mismo sesgo que tiene dentro. Restarlo es lo que
    # evita que el parche se vea como una mancha de otro tono: sin esto, el
    # salto de nivel en el borde se triplicaba al exportar.
    fuera = mcrop == 0
    if int(fuera.sum()) > 100:
        for c in range(out.shape[2]):
            sesgo = (float(np.median(crop[..., c][fuera]))
                     - float(np.median(out[..., c][fuera])))
            out[..., c] += sesgo
        # y ademas el sesgo LOCAL: alrededor del parche mismo (no de toda la
        # ventana) el tono puede seguir desviado, y eso es lo que se ve como
        # mancha mas clara o amarillenta en la piel
        anillo = _ring((mcrop > 0).astype(np.uint8),
                       max(3, max(y1 - y0, x1 - x0) // 24))
        if int(anillo.sum()) > 20:
            my, mx = np.nonzero(mcrop)
            lado = max(int(np.ptp(my)), int(np.ptp(mx)), 8)
            out += _smooth_offset(crop - out, anillo.astype(np.float32),
                                  max(lado * 0.35, 3.0))
        out = np.clip(out, 0.0, 1.0)

    # OJO: el desvanecido del borde va en FRACCION del parche, no en pixeles.
    # Con un valor fijo, la misma foto exportada al doble de resolucion tenia
    # una transicion la mitad de ancha en proporcion, y ahi es donde se veia
    # el contorno del parche que en pantalla no estaba.
    blend = _blend_mask(mcrop)
    img[y0:y1, x0:x1] = crop * (1.0 - blend) + out * blend


def inpaint(img, mask, progress_cb=None):
    """img: float32 RGB 0..1; mask: uint8 (255 = borrar).
    Devuelve una copia con las zonas marcadas rellenadas."""
    if mask.max() == 0:
        return img
    result = img.copy()
    h, w = img.shape[:2]

    # cada zona conexa se procesa con su propio contexto alrededor
    n_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    for i in range(1, n_labels):
        ys, xs = np.where(labels == i)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        region_mask = np.where(labels == i, mask, 0)
        if (max(y1 - y0, x1 - x0) <= PATCH_FRAC * max(h, w)
                and _patch_heal(result, mask, region_mask, y0, y1, x0, x1)):
            pass    # curado copiando piel/textura real de al lado
        elif not model_available():
            # sin modelo descargado: relleno clasico (mejor que nada)
            pad = 24
            _fast_inpaint_region(result, region_mask,
                                 max(y0 - pad, 0), min(y1 + pad + 1, h),
                                 max(x0 - pad, 0), min(x1 + pad + 1, w))
        else:
            # siempre LaMa, tambien en manchas pequenas: el relleno clasico
            # dejaba un borron liso que desentona en piel o zonas con grano
            side = max(y1 - y0, x1 - x0)
            pad = max(int(side * 0.8), 96)
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            half = min(max(side // 2 + pad, 192), max(h, w) // 2)
            ry0, ry1 = max(cy - half, 0), min(cy + half, h)
            rx0, rx1 = max(cx - half, 0), min(cx + half, w)
            _inpaint_region(result, region_mask, ry0, ry1, rx0, rx1)
        if progress_cb:
            progress_cb(i / (n_labels - 1))
    return np.clip(result, 0.0, 1.0)
