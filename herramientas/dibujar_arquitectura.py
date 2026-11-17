"""
dibujar_arquitectura.py — Genera los dos diagramas de arquitectura (SVG).

    arquitectura/arquitectura_prototipo.svg   flujo completo del prototipo
    arquitectura/arquitectura_modelo.svg      el autoencoder convolucional

Los SVG se escriben "a mano" con unas pocas funciones de ayuda, así es fácil
cambiar un texto o mover una caja. Para pasarlos a PNG (diapositivas) se usa
herramientas/svg_a_png.js.

Uso:
    python herramientas/dibujar_arquitectura.py
"""

import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "arquitectura")

# Paleta: tinta neutra + un tono suave por tipo de componente.
TINTA = "#0b0b0b"
TINTA_SUAVE = "#52514e"
FONDO = "#fcfcfb"
LINEA = "#b9b8b2"
AZUL = ("#e6f0fb", "#2a78d6")      # datos
NARANJA = ("#fdebe3", "#eb6834")   # modelos
VERDE = ("#e2f5ee", "#12835b")     # evaluación y resultados
GRIS = ("#f1f0ec", "#8a8983")      # hecho por los autores (se reutiliza)
FUENTE = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"


def texto(x, y, contenido, tam=15, color=TINTA, peso=400, ancla="start", cursiva=False):
    estilo = ' font-style="italic"' if cursiva else ""
    return (f'<text x="{x}" y="{y}" font-size="{tam}" fill="{color}" font-weight="{peso}" '
            f'text-anchor="{ancla}"{estilo}>{contenido}</text>')


def caja(x, y, ancho, alto, titulo, lineas=(), colores=AZUL, punteada=False, tam_titulo=16):
    """Rectángulo redondeado con un título en negrita y renglones debajo."""
    relleno, borde = colores
    trazo = ' stroke-dasharray="7 5"' if punteada else ""
    partes = [f'<rect x="{x}" y="{y}" width="{ancho}" height="{alto}" rx="10" fill="{relleno}" '
              f'stroke="{borde}" stroke-width="1.6"{trazo}/>']
    partes.append(texto(x + 14, y + 26, titulo, tam_titulo, TINTA, 600))
    for i, renglon in enumerate(lineas):
        partes.append(texto(x + 14, y + 48 + 19 * i, renglon, 13.5, TINTA_SUAVE))
    return "\n".join(partes)


def flecha(x1, y1, x2, y2, rotulo=None):
    partes = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{TINTA_SUAVE}" '
              f'stroke-width="1.8" marker-end="url(#punta)"/>']
    if rotulo:
        partes.append(texto((x1 + x2) / 2 + 8, (y1 + y2) / 2 + 4, rotulo, 12.5, TINTA_SUAVE, cursiva=True))
    return "\n".join(partes)


def documento(ancho, alto, cuerpo):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ancho} {alto}" width="{ancho}" height="{alto}" font-family="{FUENTE}">
<defs>
  <marker id="punta" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{TINTA_SUAVE}"/>
  </marker>
</defs>
<rect width="{ancho}" height="{alto}" fill="{FONDO}"/>
{cuerpo}
</svg>
'''


def etapa(y, numero, nombre, archivo, herramienta):
    """Rótulo de la columna izquierda: nombre de la etapa, archivo y herramienta."""
    return "\n".join([
        f'<circle cx="34" cy="{y + 22}" r="15" fill="{TINTA}"/>',
        texto(34, y + 27, numero, 15, "#ffffff", 700, "middle"),
        texto(58, y + 20, nombre, 16, TINTA, 700),
        texto(58, y + 41, archivo, 13, TINTA_SUAVE),
        texto(58, y + 60, herramienta, 13, TINTA_SUAVE, cursiva=True),
    ])


# ---------------------------------------------------------------------------
# DIAGRAMA 1: flujo completo del prototipo
# ---------------------------------------------------------------------------
def arquitectura_prototipo():
    p = [texto(20, 38, "Arquitectura del prototipo de replicación", 24, TINTA, 700),
         texto(20, 62, "Next Day Wildfire Spread (Huot et al., 2022) · de los datos de los autores a la comparación de resultados", 14.5, TINTA_SUAVE)]

    # Etapa 0: lo que hicieron los autores y aquí se reutiliza.
    y = 86
    p.append(etapa(y, "0", "Origen de los datos", "hecho por los autores", "Google Earth Engine"))
    p.append(caja(290, y, 500, 92, "Fuentes satelitales y climáticas (EE. UU., 2012–2020)",
                  ["MODIS (fuego) · VIIRS (vegetación) · SRTM (elevación)",
                   "GRIDMET (clima, sequía, ERC) · GPWv4 (población)"], GRIS, punteada=True))
    p.append(caja(860, y, 700, 92, "Dataset público en Kaggle (se reutiliza tal cual)",
                  ["18 545 muestras de 64 × 64 km · 12 capas de entrada + fuego de mañana",
                   "Ya separado en entrenamiento / validación / prueba (80 / 10 / 10)"], GRIS))
    p.append(flecha(790, y + 46, 858, y + 46))

    # Etapa 1: preparación de datos.
    y = 216
    p.append(etapa(y, "1", "Preparación de datos", "src/datos.py · paso1", "TensorFlow (tf.data), NumPy"))
    ancho, hueco = 302, 17
    cajas = [("Leer TFRecord", ["64 × 64 × 12 + etiqueta", "explorar y verificar"]),
             ("Aumentar", ["volteo y rotación al azar", "(corrige dirección del viento)"]),
             ("Recortar 32 × 32 km", ["ventana al azar: mueve", "el fuego de sitio"]),
             ("Limpiar y normalizar", ["recorte de extremos", "(x − media) / desviación"])]
    for i, (titulo, lineas) in enumerate(cajas):
        x = 290 + i * (ancho + hueco)
        p.append(caja(x, y, ancho, 92, titulo, lineas, AZUL))
        if i:
            p.append(flecha(x - hueco, y + 46, x - 1, y + 46))
    # Del dataset de Kaggle a la lectura (flecha en codo).
    p.append(f'<path d="M 900 178 L 900 197 L 441 197" fill="none" stroke="{TINTA_SUAVE}" stroke-width="1.8"/>')
    p.append(flecha(441, 197, 441, y - 2))
    p.append(texto(930, 202, "Aumentar y recortar se aplican solo al entrenar la red", 12.5, TINTA_SUAVE, cursiva=True))

    # Etapa 2: modelos.
    y = 350
    p.append(etapa(y, "2", "Modelos", "src/modelo.py · paso2", "Keras / TensorFlow"))
    p.append(texto(58, y + 79, "src/modelos_base.py · paso3", 13, TINTA_SUAVE))
    p.append(texto(58, y + 98, "Scikit-learn", 13, TINTA_SUAVE, cursiva=True))
    p.append(caja(290, y, 620, 132, "Red neuronal: autoencoder convolucional",
                  ["Ve la imagen completa (segmentación): 46 480 parámetros",
                   "Pérdida: entropía cruzada con peso 3 para «fuego»",
                   "Adam, tasa 0,0001 · se guarda la mejor época según validación",
                   "Píxeles inciertos (nubes) fuera de la pérdida"], NARANJA))
    p.append(caja(940, y, 620, 132, "Modelos base (píxel a píxel, vecindad 3 × 3 = 108 columnas)",
                  ["Persistencia: «mañana = hoy» (no aprende; es el piso)",
                   "Regresión logística: valores por defecto, W = 7",
                   "Random forest: profundidad 15, W = 5",
                   "Entrenamiento balanceado 1:1 fuego / sin fuego"], NARANJA))
    p.append(flecha(600, 310, 600, y - 2))
    p.append(flecha(1250, 310, 1250, y - 2))

    # Etapa 3: evaluación.
    y = 524
    p.append(etapa(y, "3", "Evaluación", "src/metricas.py", "Keras (AUC), NumPy"))
    p.append(caja(290, y, 1270, 92, "Misma regla para todos los modelos, sobre el conjunto de prueba (1 689 muestras de 64 × 64 km)",
                  ["AUC PR (principal) · precisión · recall — no accuracy: solo el 1,1 % de los píxeles tiene fuego",
                   "Verificación: la persistencia reproduce al decimal los números del artículo (Tablas II y III)"], VERDE))
    p.append(flecha(600, 484, 600, y - 2))
    p.append(flecha(1250, 484, 1250, y - 2))

    # Etapa 4: explicabilidad.
    y = 658
    p.append(etapa(y, "4", "Explicabilidad", "paso3 · paso4", "Scikit-learn, Keras"))
    cajas = [("Ablación de la red", ["re-entrenar sin el fuego previo", "y solo con el fuego previo"]),
             ("Importancia de Gini", ["random forest: qué columnas", "separan mejor fuego / no fuego"]),
             ("Coeficientes y ablación", ["regresión logística: peso de", "cada variable y quitar una a una"])]
    ancho, hueco = 410, 20
    for i, (titulo, lineas) in enumerate(cajas):
        p.append(caja(290 + i * (ancho + hueco), y, ancho, 92, titulo, lineas, VERDE))
    p.append(flecha(925, 618, 925, y - 2))

    # Etapa 5: comparación.
    y = 792
    p.append(etapa(y, "5", "Comparación", "paso5", "Matplotlib"))
    p.append(caja(290, y, 1270, 72, "Artículo frente a replicación",
                  ["Tablas II, III, IV, VI, VII y VIII lado a lado · curvas de entrenamiento · mapas de predicción · diferencias explicadas"],
                  VERDE))
    p.append(flecha(925, 752, 925, y - 2))

    # Leyenda.
    y = 888
    for i, (colores, rotulo, punteada) in enumerate((
            (GRIS, "hecho por los autores (se reutiliza)", False),
            (AZUL, "datos", False), (NARANJA, "modelos", False), (VERDE, "evaluación y resultados", False))):
        x = 290 + (0, 330, 450, 590)[i]
        p.append(f'<rect x="{x}" y="{y}" width="20" height="14" rx="3" fill="{colores[0]}" stroke="{colores[1]}" stroke-width="1.4"/>')
        p.append(texto(x + 28, y + 12, rotulo, 13, TINTA_SUAVE))
    return documento(1580, 918, "\n".join(p))


# ---------------------------------------------------------------------------
# DIAGRAMA 2: el autoencoder convolucional
# ---------------------------------------------------------------------------
def arquitectura_modelo():
    p = [texto(20, 38, "Red neuronal: autoencoder convolucional (Fig. 6 del artículo)", 24, TINTA, 700),
         texto(20, 62, "Entra una «foto» de 12 capas de 32 × 32 km; sale la probabilidad de fuego para mañana en cada km²", 14.5, TINTA_SUAVE)]

    # Cada bloque: (rótulo, detalle, tamaño de salida, lado en píxeles, color).
    # El alto de la caja es proporcional al tamaño espacial: se ve el "reloj de arena".
    bloques = [
        ("Entrada", "variables + fuego de hoy", "32 × 32 × 12", 32, GRIS),
        ("Bloque inicial", "2 conv. 3×3 + atajo", "32 × 32 × 16", 32, AZUL),
        ("ResBlock", "MaxPool 2×2", "16 × 16 × 32", 16, AZUL),
        ("ResBlock", "MaxPool 2×2", "8 × 8 × 32", 8, NARANJA),
        ("Agrandar ×2", "+ ResBlock conv.", "16 × 16 × 32", 16, VERDE),
        ("Agrandar ×2", "+ ResBlock conv.", "32 × 32 × 16", 32, VERDE),
        ("Conv. 1×1", "1 filtro + sigmoide", "32 × 32 × 1", 32, GRIS),
    ]
    centro, ancho, hueco, x0 = 232, 180, 42, 40
    for i, (rotulo, detalle, salida, lado, colores) in enumerate(bloques):
        alto = 70 + lado * 6.5
        x, y = x0 + i * (ancho + hueco), centro - alto / 2
        p.append(f'<rect x="{x}" y="{y}" width="{ancho}" height="{alto}" rx="10" fill="{colores[0]}" stroke="{colores[1]}" stroke-width="1.6"/>')
        p.append(texto(x + ancho / 2, centro - 12, rotulo, 16, TINTA, 600, "middle"))
        p.append(texto(x + ancho / 2, centro + 9, detalle, 13, TINTA_SUAVE, ancla="middle"))
        p.append(texto(x + ancho / 2, centro + 31, salida, 13.5, TINTA, 600, "middle"))
        if i:
            p.append(flecha(x - hueco + 2, centro, x - 2, centro))

    # Llaves de las tres zonas.
    def zona(i_ini, i_fin, rotulo, colores):
        x1 = x0 + i_ini * (ancho + hueco)
        x2 = x0 + i_fin * (ancho + hueco) + ancho
        y = 392
        return "\n".join([
            f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{colores[1]}" stroke-width="3"/>',
            texto((x1 + x2) / 2, y + 22, rotulo, 14.5, TINTA, 600, "middle"),
        ])
    p.append(zona(1, 2, "CODIFICADOR · comprime la imagen", AZUL))
    p.append(zona(3, 3, "CUELLO DE BOTELLA", NARANJA))
    p.append(zona(4, 5, "DECODIFICADOR · reconstruye el mapa a 1 km", VERDE))

    # Detalle del ResBlock.
    y = 450
    p.append(texto(40, y + 22, "Dentro de un ResBlock", 17, TINTA, 700))
    pasos = ["LeakyReLU", "Dropout 0,1", "MaxPool o Conv 3×3", "LeakyReLU", "Dropout 0,1", "Conv 3×3", "Dropout 0,1"]
    xa = 40
    for i, nombre in enumerate(pasos):
        w = 150 if "MaxPool" in nombre else 108
        p.append(f'<rect x="{xa}" y="{y + 40}" width="{w}" height="36" rx="8" fill="#ffffff" stroke="{LINEA}" stroke-width="1.4"/>')
        p.append(texto(xa + w / 2, y + 63, nombre, 13, TINTA, ancla="middle"))
        if i < len(pasos) - 1:
            p.append(flecha(xa + w + 1, y + 58, xa + w + 19, y + 58))
        xa += w + 20
    # Suma y atajo.
    xs = xa + 14
    p.append(flecha(xa - 19, y + 58, xs - 15, y + 58))
    p.append(f'<circle cx="{xs}" cy="{y + 58}" r="13" fill="#ffffff" stroke="{TINTA_SUAVE}" stroke-width="1.6"/>')
    p.append(texto(xs, y + 64, "+", 18, TINTA, 600, "middle"))
    p.append(f'<path d="M 22 {y + 58} L 22 {y + 112} L {xs} {y + 112}" fill="none" stroke="{TINTA_SUAVE}" stroke-width="1.8"/>')
    p.append(flecha(xs, y + 112, xs, y + 73))
    p.append(f'<line x1="22" y1="{y + 58}" x2="38" y2="{y + 58}" stroke="{TINTA_SUAVE}" stroke-width="1.8" marker-end="url(#punta)"/>')
    p.append(f'<rect x="420" y="{y + 95}" width="230" height="34" rx="8" fill="#ffffff" stroke="{LINEA}" stroke-width="1.4"/>')
    p.append(texto(535, y + 117, "atajo: Conv 1×1 (conexión residual)", 13, TINTA, ancla="middle"))

    # Ficha técnica.
    x, y = 1030, 450
    p.append(caja(x, y, 520, 134, "Ficha técnica (Apéndice A-A del artículo)",
                  ["46 480 parámetros · todas las conv. sin sesgo, relleno «same»",
                   "Filtros: 16 al inicio y al final, 32 en el centro",
                   "Pérdida: entropía cruzada, peso 3 para «fuego»",
                   "Adam · tasa de aprendizaje 0,0001 · lote de 32"], GRIS))
    return documento(1580, 606, "\n".join(p))


if __name__ == "__main__":
    os.makedirs(SALIDA, exist_ok=True)
    for nombre, svg in (("arquitectura_prototipo.svg", arquitectura_prototipo()),
                        ("arquitectura_modelo.svg", arquitectura_modelo())):
        with open(os.path.join(SALIDA, nombre), "w", encoding="utf-8") as f:
            f.write(svg)
        print("Escrito", os.path.join(SALIDA, nombre))
