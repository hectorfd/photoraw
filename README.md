# PhotoRAW

Editor de fotos RAW hecho a medida, con herramientas IA que corren en tu
propia GPU (NVIDIA RTX 4070 + CUDA). Todo local: ninguna foto sale de tu PC.

## Cómo abrirlo

Doble clic en **PhotoRAW.bat** (en esta carpeta).

## Instalar desde cero (repositorio recién clonado)

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Los modelos de IA **no están en el repositorio** (son ~3,3 GB). Abre
PhotoRAW y pulsa **Modelos de IA** en la barra de herramientas: ahí ves
las 9 IAs del proyecto, cuáles faltan y cuánto pesa cada una, y las bajas
de una en una (o todas de golpe) según lo que vayas a usar. Quedan en
`~/.photoraw/models`. Desde esa misma ventana puedes borrar las que no
uses para recuperar el espacio.

## Lo que ya tiene

### Revelado (pestaña Revelar)
- **Perfil**: interpretación base del RAW — Estándar, Vívido, Retrato,
  Paisaje, Plano (para editar) y Blanco y negro.
- **Ajustes básicos**: exposición, contraste, luces, sombras, blancos,
  negros, temperatura, matiz, textura, borrar neblina, saturación, vitalidad.
- **Curva de tonos**: paramétrica, de puntos y por canal (RGB / R / G / B).
- **Mezclador de color HSL**: 8 bandas (rojo → magenta), cada una con matiz,
  saturación y luminancia.
- **Color de punto**: cuentagotas 💧 — toma una muestra en la foto y cambia
  solo ese color (matiz hasta ±180°, saturación, luminancia y rango).
- **Calibración**: matiz de sombras y tono/saturación de cada primario RGB
  (la matriz conserva los neutros: los grises jamás se tiñen).
- **Detalle**: enfoque (cantidad/radio/detalle/máscara — Alt para ver la
  máscara), reducción de ruido manual (luminancia/color).
- **Efectos**: grano de película (cantidad, tamaño, aspereza).
- Todos los deslizadores de color llevan **pistas degradadas** que indican
  hacia dónde va el ajuste (estilo Lightroom).

### Recorte (pestaña Recorte)
Marco arrastrable con rejilla de tercios, proporciones (1:1, 4:3, 3:2, 16:9,
verticales…), enderezar ±45°, girar 90° y voltear. No destructivo.

### Máscaras (pestaña Máscaras) — ajustes locales
- **Degradado lineal** y **radial**: se dibujan arrastrando sobre la foto y
  después se **editan con tiradores** (mover el centro/extremos, estirar
  ancho y alto); la radial además tiene control de **desvanecido**.
- **Pincel**: pinta a mano la zona, con modo **añadir/quitar** (borrador).
- **Sujeto (IA)** y **Fondo (IA)**: segmentación automática con u2net.
- **Retrato (IA)**: la IA divide las caras en zonas y cada una se vuelve
  una máscara — **Piel, Cabello, Cejas, Ojos, Labios o Dientes** (menú del
  botón). Ejemplos: aclarar solo los dientes, suavizar solo la piel, dar
  color solo a los labios. Si hay varias caras, la máscara las une todas.
- Cada máscara tiene sus propios ajustes (exposición, contraste, luces,
  sombras, temperatura, matiz, saturación) y se puede **invertir**. El velo
  rojo que marca la zona se puede **ocultar** con una casilla y se aparta
  solo mientras arrastras un ajuste.
- **Ruido IA y Rostros IA por máscara**: aplica la reducción de ruido o el
  retoque de rostros solo en la zona de la máscara (p. ej. limpiar solo el
  fondo). La primera vez en cada foto la IA tarda unos segundos; después
  el deslizador responde al instante. También sale así en el JPEG exportado.
- **Refinar a mano**: botones **Añadir / Restar** — pinta con el pincel
  sobre cualquier máscara (incluidas las de IA) para sumarle zonas que le
  faltaron o quitarle lo que sobró.
- **Borrar con IA (rellenar fondo)**: hace desaparecer lo que cubre la
  máscara seleccionada y **reconstruye el fondo con IA generativa**
  (difusión local en tu GPU, ~15-20 s). Para quitar personas y objetos
  grandes de verdad. Ctrl+Z lo deshace; al exportar se re-aplica con la
  misma semilla. Modelo ~2 GB, ya descargado. La resolución de trabajo
  se adapta a la zona y a la VRAM libre (512/640/768 px): con otras apps
  acaparando la GPU baja de resolución en vez de arrastrarse. Consejo:
  máscara ajustada al objeto, y si el fondo mezcla texturas (pasto+agua),
  mejor borrar por partes.

### Herramientas IA (modelos ONNX en tu GPU)
| Herramienta | Modelo | Qué hace |
|---|---|---|
| Reducción de ruido | SCUNet | Limpia el ruido conservando el detalle |
| Retoque de rostros | CodeFormer / GFPGAN | Restaura caras (piel, ojos) |
| Corrector | LaMa | Todos los trazos con IA (~0,2 s; el relleno clásico queda de respaldo si falta el modelo) |
| Máscaras Sujeto/Fondo | u2net | Segmenta a las personas / objeto principal |
| Máscaras de retrato | BiSeNet | Divide las caras en piel, pelo, ojos, labios… |
| Superresolución | Real-ESRGAN x4 | Exporta a 2× o 4× reconstruyendo detalle |
| Borrado generativo | Realistic Vision (difusión) | Reconstruye el fondo tras quitar personas/objetos |

Los modelos se descargan una sola vez (al usarlos, o desde la ventana
**Modelos de IA** de la barra) y quedan en `~/.photoraw/models`. Esa
ventana es también donde ves cuánto ocupan y puedes borrar los que no uses.

### Flujo de trabajo
- **Original** (tecla `O`): antes/después instantáneo.
- **Píldora de progreso** flotante sobre la foto: aro girando + estado de lo
  que trabaja en segundo plano, con porcentaje en ruido IA, corrector y
  borrado generativo.
- **Detener la IA**: botón **Detener IA** en la barra de herramientas —
  apagado en gris cuando no hay nada calculando, **rojo** mientras la IA
  trabaja, así que de un vistazo sabes si hay algo en marcha. También la ✕
  de la píldora flotante o la tecla `Esc`. Corta el trabajo
  en marcha. El aro para al instante y la app vuelve a responder; la GPU
  puede tardar unos segundos más en soltar el paso que ya tenía entre manos
  (una llamada al modelo no se puede interrumpir a medias). Lo que se
  quedó a medio calcular se descarta: un borrado generativo o unos trazos
  del corrector detenidos **no** quedan guardados. Detener **también
  devuelve la tarjeta gráfica**: descarga los modelos que estaban
  residentes (se recargan solos al volver a usarlos). Y si pasas
  5 minutos sin usar la IA, se sueltan igualmente sin que hagas nada.
  **Cerrar la ventana también detiene la IA**: antes el proceso seguía vivo e invisible
  quemando la GPU hasta acabar el trabajo. Después del Detener la
  IA guardada de esa foto queda en pausa y no se relanza sola, hasta que
  vuelvas a pedirla (o cambies de foto).
- **Restaurar foto**: borra toda la edición (con confirmación).
- **Corrector** (tecla `B`): pinta y suelta para borrar; Ctrl+Z deshace.
- **Copiar/Pegar ajustes** (Ctrl+Shift+C/V) a varias fotos, **preajustes**
  con nombre, **insignia de lápiz** en las miniaturas editadas y miniatura
  viva que refleja tu edición.
- **Exportar JPEG** (Ctrl+E) a tamaño original, 2× o 4× con IA.

### Rendimiento
- **Render en dos fases**: borrador reducido mientras arrastras un ajuste
  (~0,2 s) y calidad completa al soltar. El zoom no salta en el cambio.
- **Renders en fila india**: nunca corre más de un render a la vez; si
  mueves un ajuste mientras uno trabaja, solo se atiende el pedido más
  reciente. Así el procesador no se satura al arrastrar.
- **Caché en disco** (`~/.photoraw/cache`, máx. 4 GB, se limpia solo): las
  fotos ya visitadas cargan en ~30 ms y las miniaturas al instante.
- **La IA también se guarda en disco**: ruido IA, rostros IA, corrector,
  borrado generativo y máscaras IA se calculan una sola vez por foto —
  reabrir una foto editada pasa de ~16 s a ~0,2 s.
- **Carril rápido**: decodificar y renderizar tienen sus propios hilos; la
  foto en pantalla nunca hace cola detrás de un trabajo IA largo.
- Curvas y HSL por tablas de consulta; mapas suaves a resolución reducida.

### No destructivo
Los ajustes viven en `.photoraw_edits.json` dentro de cada carpeta de fotos.
Tus RAW originales **nunca** se modifican.

## Formatos compatibles

RAW: CR2, CR3, NEF, ARW, DNG, RAF, RW2, ORF, PEF y más.
También JPEG, PNG, TIFF, WebP y **HEIC/HEIF** (las fotos de iPhone y de
móviles Android recientes), vía `pillow-heif`.

A las fotos que traen perfil de color incrustado (los HEIC y JPG de iPhone
vienen en Display P3) se les convierte el color a sRGB al cargarlas, igual
que hace Windows; sin eso los rojos y verdes salen sobresaturados.

## Pendiente / ideas futuras

- [ ] **Pincel inteligente** (SAM/MobileSAM): tocar un objeto y que la
      máscara se ajuste sola a sus bordes.
- [ ] Máscara de **Cielo** con IA (falta elegir un buen modelo ONNX de
      segmentación de cielo; Sujeto/Fondo ya están).
- [ ] Máscara por **rango de color/luminancia** (matemática, como el color
      de punto pero para máscaras) y rotación de los degradados.
- [x] ~~**Edición generativa** (borrar objetos grandes rellenando con
      difusión, tipo Magic Editor)~~ — hecho: botón «Borrar con IA».
- [ ] Textura/claridad como ajustes locales por máscara (ruido IA y rostros
      IA ya están).
- [ ] Historial de deshacer general (Ctrl+Z hoy solo deshace el corrector).
- [ ] Exportar con perfil de color incrustado y elección de calidad JPEG.
- [ ] Vista de comparación lado a lado (antes | después en pantalla dividida).

## Arquitectura (para el que programe)

```
photoraw/
  loader.py      decodificación RAW (rawpy) e imágenes
  diskcache.py   caché en disco de vistas previas y miniaturas (LRU)
  engine.py      pipeline de revelado (numpy/OpenCV): geometría → perfil →
                 calibración → tonal → curvas → HSL/color punto → máscaras
                 → detalle → grano
  edits.py       persistencia no destructiva (JSON por carpeta)
  presets.py     preajustes con nombre
  ai.py          SCUNet (ruido) + infraestructura ONNX/CUDA compartida
  faces.py       CodeFormer/GFPGAN (rostros) + YuNet (detección)
  heal.py        corrector: LaMa (IA; Telea de respaldo sin modelo)
  masks_ai.py    u2net (segmentación de sujeto)
  face_parse.py  BiSeNet (máscaras de retrato: piel, pelo, ojos...)
  generative.py  borrado generativo (difusión) + afinado Real-ESRGAN
  upscale.py     Real-ESRGAN (superresolución)
  models.py      catálogo de las IAs descargables (AI_MODELS)
  ui/            PySide6: ventana principal, visor con zoom/recorte/máscaras,
                 editor de curvas, ventana de modelos de IA
```



# Aqui te Quedaste CLAUDE

**Hecho (2026-07-07):**
- Nitidez del «Borrar con IA»: el parche se re-amplía con Real-ESRGAN en
  borrados grandes — falta que Héctor lo valide con la foto del pasto seco.
- Fotos editadas lentas al reabrir: los resultados IA (ruido, rostros,
  corrector, borrado, máscaras) ahora se cachean en disco (16 s → 0,2 s) y
  decodificar/renderizar tienen carril propio de hilos. La primera apertura
  de una foto sigue calculando la IA una vez; las siguientes son instantáneas.
- Miniaturas de DNG de iPhone salían volteadas: la miniatura incrustada ahora
  se endereza con la orientación del RAW (convención dcraw de LibRaw).
- El corrector dejaba un borrón liso en trazos pequeños (la nariz de Héctor,
  jaja): eran Telea; ahora TODOS los trazos usan LaMa (~0,2 s con la sesión
  caliente; clave de caché subida a "heal2" para invalidar los borrones
  guardados). Verificado con los trazos reales de IMG_3270.
- Píldora de progreso (BusyChip) flotante sobre el visor: aro girando +
  estado, con porcentaje en ruido IA, corrector y borrado generativo.
- El borrado generativo se quedaba «al 100%» ~3 min: el porcentaje solo
  contaba la difusión, y el afinado ESRGAN posterior se arrastraba porque
  los modelos de difusión residentes acaparaban la VRAM. Arreglado en tres
  frentes: cuDNN a HEURISTIC (sin calibración exhaustiva al primer uso),
  borrado en dos fases que libera la difusión de la VRAM antes de afinar
  (210 s → 12,6 s medidos), y el porcentaje ahora cubre todo el trabajo
  (difusión 85% + afinado 15%, tope visual en 99% hasta acabar).

**Hecho (2026-07-08, madrugada):**
- **DATO CLAVE descubierto**: la RTX 4070 de Héctor es la de **8 GB** (no
  12), y apps de fondo (iCloud Photos, Camo, Kaspersky, Screenpresso) le
  comen VRAM. Cuando la VRAM se desborda, Windows tira de RAM y todo va
  10-100× más lento sin avisar — esa era la raíz de casi toda la lentitud.
- Relleno «tipo pelaje» en el pasto: la difusión trabajaba fija a 512 px.
  Ahora la resolución es adaptativa según la VRAM libre del momento
  (`_pick_size`): 640 px medido en su GPU = 19,8 s limpio; 768 desborda
  (513 s), queda para tarjetas más grandes. Escalera de reintentos
  768→640→512 si falta memoria. Clave de caché → "erase2".
- El techo de calidad restante es el modelo (Realistic Vision 5.1, clase
  SD 1.5): es lo razonable para 8 GB. SDXL/FLUX no entran cómodos.

**Hecho (2026-08-03):**
- **Botón de Detener la IA** (queja de Héctor: cuando el borrado generativo
  se atasca, el aro gira para siempre y no hay forma de pararlo). Tres vías:
  botón **Detener IA** en la barra (`a_stop_ai`, gris/rojo según haya trabajo
  — lo conmuta `_update_busy`, y lleva el atajo `Esc`) y la ✕ de la píldora.
  No encontró la ✕ porque la píldora solo existe mientras la IA trabaja: de
  ahí el botón fijo.
  - `ai.Cancelled` + `AIJob` (clase base de los 6 trabajos de IA en
    `main_window.py`): cada trabajo se queda al crearse con un «testigo»
    (`window.ai_cancel`), y `AIJob.progress()` envuelve el `progress_cb` que
    ya usaban los módulos, así que **los puntos de progreso existentes son
    los puntos de control** — no hizo falta instrumentar los bucles.
  - Al pulsar Detener, la ventana marca el testigo viejo y estrena uno
    limpio (los trabajos nuevos no heredan la cancelación), vacía los
    conjuntos `*_running` y apaga la píldora **sin esperar** al hilo: una
    llamada al modelo ya lanzada en la GPU no se puede interrumpir desde
    Python, y bloquear la interfaz por eso era justo el problema.
  - Cuidado con los `except Exception` que rodean pasos de IA: en
    `generative.py` la escalera de reintentos 768→640→512 y `_sharpen_patch`
    se tragaban la cancelación (reintentaban en vez de abortar). Ambos
    dejan pasar `Cancelled` ahora.
  - Un borrado generativo o unos trazos del corrector detenidos se **quitan**
    del historial (`_drop_erase_ops` / `_drop_heal_strokes`): no llegaron a
    aplicarse, así que no deben guardarse ni reintentarse al reabrir. El
    hilo de trabajo solo deja el aviso (`erase_undo_request`), lo ejecuta el
    hilo de la interfaz. Re-aplicar borrados ya guardados NO los borra
    (`drop_on_cancel=False`).
  - `ai_paused`: tras Detener, `_autorun_ai` / `_autorun_erase` no relanzan
    la IA guardada (si no, el resultado vacío disparaba otra vuelta y volvía
    a empezar). Se levanta al pedir IA a mano o al cambiar de foto.
  - Al abortar un borrado se llama a `generative.release_sessions()`: suelta
    los ~2 GB de VRAM en vez de dejarlos ocupados.
- **PhotoRAW fantasma (el fallo gordo que salió de esto)**: `closeEvent` no
  paraba la IA, y `QThreadPool` **espera** a los QRunnable en curso antes de
  dejar morir el proceso. Resultado: cerrabas la ventana, desaparecía de la
  pantalla y PhotoRAW seguía vivo **sin ninguna ventana** quemando la GPU
  hasta acabar el borrado. Así se acumularon 4 procesos peleándose por los
  8 GB (uno con 590 s de CPU, VRAM al 100 % y solo 499 MiB libres) — lo que
  a su vez hacía que cada paso de difusión tardase muchísimo. `closeEvent`
  ahora marca `ai_cancel` y vacía las colas (`pool.clear()`), así que cerrar
  la ventana corta de verdad. Medido: con un trabajo de 60 s en marcha, el
  proceso muere en 1,5 s; el control sin cancelación sobrevivía a la
  ventana los 8 s enteros del trabajo.
  **Si algo va lentísimo, mira primero cuántos pythonw hay vivos.**
- **Modelos residentes = VRAM ocupada sin estar trabajando.** Los 7 módulos
  guardaban su `InferenceSession` en un global y no la soltaban nunca:
  SCUNet + LaMa + ESRGAN + u2net + rostros + BiSeNet acaban sumando GB
  aunque no se calcule nada (Héctor lo vio en el Administrador de tareas y
  creyó que la tarjeta seguía «quemando»; era memoria reservada, no
  trabajo: 0,0 s de CPU en 6 s y 45 °C). Nuevo `ai.release_all_sessions()`
  (seguro con trabajos en vuelo: solo suelta la referencia del módulo, el
  hilo que la usa mantiene viva la suya) llamado desde Detener y desde un
  temporizador de `IDLE_FREE_MIN` = 5 min sin IA. Efecto secundario bueno:
  con más VRAM libre, `_pick_size` puede elegir 640 px en vez de 512 y el
  borrado sale con mejor textura.
- **Ventana «Modelos de IA»** (botón de la barra, al lado de Detener IA):
  catálogo en `photoraw/models.py` (`AI_MODELS`) + `ui/models_dialog.py`,
  que se dibuja solo a partir de la lista — para añadir una IA basta con
  sumar una entrada. Muestra las 9 IAs con lo que hace cada una, su peso y
  si está instalada; descarga en segundo plano (con barra por fila) y
  botón de eliminar por modelo. Objetivo: el repo no lleva los 3,3 GB, se
  clona, se instalan requirements y cada uno baja lo que use.
  Trampa de layout ya resuelta: las `QLabel` de la fila necesitan
  `setWordWrap(True)` **y** `setMinimumWidth(1)`, o el texto más largo fija
  el ancho mínimo (872 px medidos) y los botones se salen de la ventana.
- **Detener IA se puede pulsar siempre**: en reposo avisa de que no hay
  nada corriendo y suelta los modelos de la GPU. Antes se quedaba en gris
  y desde fuera parecía que el botón «no hacía nada».

**Falta que Héctor pruebe** (reiniciar PhotoRAW primero): repasar el
corrector en la nariz (borrar los trazos feos y volver a pintar), abrir una
foto editada dos veces (la 2.ª debe ser inmediata), miniaturas derechas,
rehacer el «Borrar con IA» del pasto (se recalculará solo a 640 px), y
**detener un borrado generativo a media difusión** (debe parar en unos
segundos y no dejar el borrado guardado).
