# PhotoRAW

Un editor de fotos RAW para escritorio, hecho a la medida de un flujo de
trabajo propio. Revela tus RAW sin modificarlos, te deja ajustar zonas
concretas con máscaras, fusiona bracketings en HDR y trae varias herramientas
de inteligencia artificial que trabajan **con tu propia tarjeta gráfica**.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PySide6](https://img.shields.io/badge/GUI-PySide6-green)
![Local](https://img.shields.io/badge/IA-100%25%20local-orange)

## Instalación

```bash
git clone <este-repositorio>
cd photoraw
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Para abrirlo, doble clic en **PhotoRAW.bat** (o desde la consola,
`.venv\Scripts\python main.py`).

Si tu computadora no tiene una tarjeta NVIDIA, cambia `onnxruntime-gpu` por
`onnxruntime` en `requirements.txt` antes de instalar.

**Los modelos de inteligencia artificial no vienen incluidos**, porque entre
todos pesan unos 3,3 GB. Abre el programa, entra a **Modelos de IA** en la
barra de herramientas y descarga solo los que vayas a usar; desde ahí mismo
puedes borrarlos después. Se guardan en `~/.photoraw/models`. El editor
funciona perfectamente sin ninguno: simplemente se desactivan las herramientas
que los necesitan.

**Qué necesitas.** Python 3.12 y Windows. Una tarjeta NVIDIA con CUDA hace que
la inteligencia artificial vuele, pero no es obligatoria: sin ella todo
funciona igual usando el procesador, solo que más lento. El programa se
desarrolló sobre una RTX 4070 de 8 GB.

## Qué puedes hacer

**Revelar.** Perfiles de color para partir de una base (incluido *RAW sin
perfil*, que no interpreta nada y te deja el archivo tal como salió de la
cámara), exposición y tonos, curvas —paramétrica, por puntos y por canal—,
mezclador de color de 8 bandas, cuentagotas para retocar un color puntual,
calibración, enfoque y reducción de ruido. Cuando te acercas más allá del
100 %, el programa revela los píxeles reales del archivo en lugar de estirar
la vista previa, así que puedes juzgar de verdad el enfoque y el grano.

**Efectos.** Dramático, estado de ánimo, virado, mate, místico, brillo y grano
de película. Todos miden sus radios en proporción al tamaño de la foto, así
que lo que ves en pantalla es exactamente lo que sale al exportar.

**Recortar.** Marco ajustable con proporciones listas, enderezado de ±45°,
giros y volteos.

**Máscaras para ajustar solo una zona.** Degradado lineal, degradado radial,
rectángulo con las esquinas sueltas (moviéndolas una a una lo conviertes en
trapecio, ideal para agarrar una ventana o una puerta que salen en
perspectiva), pincel a mano alzada, y selección automática con IA del sujeto,
del fondo o de partes del rostro (piel, cabello, ojos, labios, dientes). Cada
máscara tiene sus propios ajustes, se puede invertir y se retoca a mano si le
falta o le sobra algo. Dentro de una máscara también puedes aplicar la
reducción de ruido o el retoque de rostros con IA.

**Fusión HDR.** Junta las tomas de un bracketing en una sola foto con detalle
tanto en las sombras como en las luces. El programa encuentra las tandas por
su cuenta, fijándose en la hora de disparo y en la diferencia de exposición.
Puedes elegir entre dos caminos: el *natural*, que va zona por zona quedándose
con la mejor toma de cada una, y el *HDR real*, que reconstruye cuánta luz
recibió la escena partiendo de los datos en bruto del sensor y después la
comprime para que quepa en pantalla. El resultado se guarda como **DNG
lineal** de 16 bits —lo mismo que entrega Lightroom cuando fusiona un HDR— y
aparece en la tira como una foto más, con el balance de blancos todavía por
decidir.

**Herramientas con IA.** Reducción de ruido (SCUNet), retoque de rostros
(CodeFormer y GFPGAN), pincel corrector para quitar manchas y objetos (LaMa),
borrado de cosas grandes rellenando el fondo por difusión, ampliación a 2× y
4× reconstruyendo detalle (Real-ESRGAN) y las máscaras automáticas (u2net y
BiSeNet). Todo corre en tu tarjeta gráfica con ONNX Runtime, y hay un botón
para detener lo que esté trabajando y liberar la tarjeta cuando quieras.

## Formatos que abre

**RAW:** CR2, CR3, NEF, ARW, DNG, RAF, RW2, ORF, PEF y varios más, gracias a
LibRaw. **Otros:** JPEG, PNG, TIFF (incluidos los de 16 bits), WebP y
HEIC/HEIF.

Las fotos que traen un perfil de color incrustado —los HEIC y JPG del iPhone
vienen en Display P3— se convierten a sRGB al abrirlas, para que los colores
se vean como deben.

## Dónde se guarda cada cosa

**Tus RAW nunca se modifican.** Todo lo que revelas queda anotado en un
archivo `.photoraw_edits.json` dentro de la misma carpeta de las fotos, así
que tus ediciones viajan con ellas si mueves la carpeta o la copias a otro
disco.

En `~/.photoraw` se guarda lo que pertenece a la computadora y no a las fotos:
los modelos de IA, la caché de vistas previas y miniaturas (con un límite que
puedes ajustar, 4 GB al principio) y las preferencias del programa, como la
última carpeta en la que estuviste trabajando.

## Cómo está organizado el código

```
photoraw/
  loader.py       lectura de archivos RAW (rawpy) y de imágenes normales
  engine.py       el revelado: geometría → perfil → tonos → curvas → color
                  → máscaras → detalle → efectos
  hdr.py          fusión de bracketings
  dng.py          escritura de DNG lineal
  heal.py         pincel corrector (LaMa)
  generative.py   borrado de objetos por difusión
  ai.py faces.py masks_ai.py face_parse.py upscale.py   modelos de IA
  edits.py presets.py ajustes.py   guardado de ediciones y preferencias
  diskcache.py    caché en disco
  ui/             la interfaz, hecha con PySide6
```

Si quieres el detalle de cada parte, las decisiones técnicas y el diario de
desarrollo, está todo en [NOTAS.md](NOTAS.md).
