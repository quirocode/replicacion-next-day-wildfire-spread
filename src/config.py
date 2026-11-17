"""
config.py — Configuración central del prototipo de replicación.

Artículo replicado:
    Huot, F., Hu, R. L., Goyal, N., Sankar, T., Ihme, M., & Chen, Y.-F. (2022).
    Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire
    Spreading From Remote-Sensing Data. IEEE TGRS, 60, Art. 4412513.
    https://doi.org/10.1109/TGRS.2022.3192974

Aquí vive todo lo que NO es lógica: nombres de variables, estadísticas de
normalización, hiperparámetros, el presupuesto de cómputo y los resultados que
reporta el artículo (para compararlos con los nuestros al final).

Cada constante indica de dónde sale:
    [ART]  = texto del artículo (sección o tabla)
    [COD]  = código publicado por los autores en GitHub (constants.py, etc.)
    [REP]  = decisión propia de esta replicación (el artículo no lo especifica)
"""

import os

# ---------------------------------------------------------------------------
# 1. RUTAS
# ---------------------------------------------------------------------------
# Carpeta raíz del proyecto (la que contiene a src/).
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Identificador del dataset en Kaggle, publicado por los mismos autores.
DATASET_KAGGLE = "fantineh/next-day-wildfire-spread"

# Presupuesto de cómputo elegido (se explica en la sección 5 de este archivo).
PRESUPUESTO = os.environ.get("WILDFIRE_PRESUPUESTO", "prototipo")

# Carpeta donde se guardan tablas (JSON/CSV), figuras (PNG) y el modelo. Las
# corridas de "prueba" van a otra carpeta para no pisar los resultados reales.
DIR_RESULTADOS = os.path.join(
    RAIZ, "resultados" if PRESUPUESTO != "prueba" else "resultados_prueba"
)
DIR_MODELOS = os.path.join(DIR_RESULTADOS, "modelos")

# ---------------------------------------------------------------------------
# 2. VARIABLES DEL DATASET  [COD: constants.py]
# ---------------------------------------------------------------------------
# Las 12 capas de entrada, en el MISMO orden que usa el código de los autores.
# Cada una es una imagen de 64 x 64 píxeles (1 píxel = 1 km).
VARIABLES_ENTRADA = (
    "elevation",     # elevación del terreno (m)                    — SRTM
    "pdsi",          # índice de sequía de Palmer                   — GRIDMET Drought
    "NDVI",          # índice de vegetación (x 10 000)              — VIIRS
    "pr",            # precipitación diaria (mm)                    — GRIDMET
    "sph",           # humedad específica (kg/kg)                   — GRIDMET
    "th",            # dirección del viento (grados desde el norte) — GRIDMET
    "tmmn",          # temperatura mínima (K)                       — GRIDMET
    "tmmx",          # temperatura máxima (K)                       — GRIDMET
    "vs",            # velocidad del viento (m/s)                   — GRIDMET
    "erc",           # energy release component (índice NFDRS)      — GRIDMET
    "population",    # densidad de población (hab/km²)              — GPWv4
    "PrevFireMask",  # máscara de fuego del día t ("fuego de hoy")  — MODIS
)

# La etiqueta: máscara de fuego del día t + 1 ("fuego de mañana").
VARIABLE_SALIDA = "FireMask"

# Nombres en español para tablas y figuras.
NOMBRES_ES = {
    "elevation": "Elevación",
    "pdsi": "Sequía (PDSI)",
    "NDVI": "Vegetación (NDVI)",
    "pr": "Precipitación",
    "sph": "Humedad",
    "th": "Dirección del viento",
    "tmmn": "Temp. mínima",
    "tmmx": "Temp. máxima",
    "vs": "Velocidad del viento",
    "erc": "ERC",
    "population": "Población",
    "PrevFireMask": "Fuego previo (día t)",
    "FireMask": "Fuego (día t+1)",
}

# Nombre que usa el artículo para cada variable (Tablas IV a VIII).
NOMBRES_ARTICULO = {
    "elevation": "Elevation",
    "pdsi": "Drought",
    "NDVI": "Vegetation",
    "pr": "Precipitation",
    "sph": "Humidity",
    "th": "Wind direction",
    "tmmn": "Min temperature",
    "tmmx": "Max temperature",
    "vs": "Wind speed",
    "erc": "ERC",
    "population": "Population",
    "PrevFireMask": "Previous fire mask",
}

# Estadísticas para limpiar y normalizar cada variable  [COD: constants.py].
# Formato: (recorte_mínimo, recorte_máximo, media, desviación_estándar).
# Los autores las calcularon SOLO con el conjunto de entrenamiento, después de
# recortar (Sección III del artículo). Los recortes son el percentil 0,1 y 99,9
# o un límite físico (p. ej. la precipitación no puede ser negativa).
ESTADISTICAS = {
    "elevation": (0.0, 3141.0, 657.0, 649.0),
    "pdsi": (-6.1, 7.9, 0.0, 2.7),
    "NDVI": (-9821.0, 9996.0, 5158.0, 2467.0),
    "pr": (0.0, 44.5, 1.7, 4.5),
    "sph": (0.0, 1.0, 0.0072, 0.0043),
    "th": (0.0, 360.0, 190.3, 72.6),
    "tmmn": (253.2, 298.9, 281.1, 9.0),
    "tmmx": (253.2, 315.1, 295.2, 9.8),
    "vs": (0.0, 10.0, 3.9, 1.4),
    "erc": (0.0, 106.0, 37.0, 21.0),
    "population": (0.0, 2534.0, 26.0, 155.0),
    # Las máscaras de fuego NO se normalizan: -1 incierto, 0 sin fuego, 1 fuego.
    "PrevFireMask": (-1.0, 1.0, 0.0, 1.0),
    "FireMask": (-1.0, 1.0, 0.0, 1.0),
}

# ---------------------------------------------------------------------------
# 3. TAMAÑOS  [ART: Secciones II y III]
# ---------------------------------------------------------------------------
TAM_ORIGINAL = 64   # cada muestra del dataset mide 64 x 64 km
TAM_RECORTE = 32    # la red se entrena con recortes aleatorios de 32 x 32 km

# ---------------------------------------------------------------------------
# 4. HIPERPARÁMETROS DE LA RED NEURONAL  [ART: Apéndice A-A]
# ---------------------------------------------------------------------------
FILTROS_CODIFICADOR = (16, 32)   # 16 filtros al inicio, 32 en el centro
FILTROS_DECODIFICADOR = (32, 16)  # simétrico al codificador
TASA_DROPOUT = 0.1               # "the dropout rate is 0.1"
TASA_APRENDIZAJE = 1e-4          # "learning rate of 0.0001"
PESO_CLASE_FUEGO = 3.0           # "a weight of 3 in the fire class"
TAM_LOTE = 32                    # [REP] el artículo exploró 32, 64, 128 y 256

# ---------------------------------------------------------------------------
# 5. PRESUPUESTO DE CÓMPUTO  [REP]
# ---------------------------------------------------------------------------
# El artículo entrenó 1000 épocas x 1000 pasos = 1 000 000 de pasos en 4 GPU
# V100. Eso es imposible en un prototipo de clase, así que se reduce. La
# variable de entorno WILDFIRE_PRESUPUESTO permite cambiarlo sin tocar el código:
#     "prueba"    -> segundos, solo para verificar que todo corre
#     "prototipo" -> el que se usó para los resultados de esta replicación
#     "largo"     -> para quien tenga GPU y quiera acercarse más al artículo
# Una época = 468 pasos de 32 muestras = una pasada por las 14 979 muestras de
# entrenamiento. "prototipo" son 28 080 pasos: el 2,8 % de los del artículo.
PRESUPUESTOS = {
    "prueba":    dict(epocas=2,   pasos_por_epoca=20,  pixeles_logistica=40_000,    pixeles_bosque=20_000),
    "prototipo": dict(epocas=60,  pasos_por_epoca=468, pixeles_logistica=1_000_000, pixeles_bosque=1_000_000),
    "largo":     dict(epocas=300, pasos_por_epoca=468, pixeles_logistica=1_000_000, pixeles_bosque=1_000_000),
}

# Semilla para que los experimentos se puedan repetir.
SEMILLA = 123

# [REP] Recortes de 32 x 32 km que se sacan de cada muestra de prueba. El
# artículo usa uno; con tres la medición depende menos del azar del recorte.
REPETICIONES_PRUEBA = 3

# ---------------------------------------------------------------------------
# 6. HIPERPARÁMETROS DE LOS MODELOS BASE  [ART: Apéndice A-B]
# ---------------------------------------------------------------------------
PROFUNDIDAD_BOSQUE = 15      # "we select a maximum depth of 15"
W_LOGISTICA = 7              # factor W del peso de la clase "sin fuego"
W_BOSQUE = 5
# [REP] Tope de píxeles de entrenamiento para los modelos base (mitad fuego,
# mitad sin fuego). El artículo no dice cuántos usó; estos topes mantienen el
# tiempo de cómputo razonable en CPU y dependen del presupuesto elegido.
MAX_PIXELES_LOGISTICA = PRESUPUESTOS[PRESUPUESTO]["pixeles_logistica"]
MAX_PIXELES_BOSQUE = PRESUPUESTOS[PRESUPUESTO]["pixeles_bosque"]

# ---------------------------------------------------------------------------
# 7. RESULTADOS DEL ARTÍCULO (para comparar)  [ART: Tablas II a VIII]
# ---------------------------------------------------------------------------
# Todos en porcentaje: (AUC PR, precisión, recall).
ARTICULO_TABLA_II = {
    "Red neuronal": (28.4, 33.6, 43.1),
    "Random forest": (22.5, 26.3, 46.9),
    "Regresión logística": (19.8, 32.5, 35.3),
    "Persistencia": (11.5, 35.7, 27.3),
}

# Tabla III, filas de "Persistence": resolución más gruesa (etiqueta binaria).
ARTICULO_TABLA_III_PERSISTENCIA = {
    2: (19.5, 45.6, 36.8),
    4: (30.6, 56.0, 46.8),
    8: (41.3, 64.0, 52.5),
}
# Tabla III, filas "Binary" de la red neuronal (referencia, no se replican).
ARTICULO_TABLA_III_RED_BINARIA = {
    2: (39.8, 40.3, 51.9),
    4: (53.5, 48.8, 61.0),
    8: (66.3, 62.1, 63.1),
}

# Tabla IV: red neuronal re-entrenada QUITANDO una variable.
ARTICULO_TABLA_IV = {
    "PrevFireMask": (6.8, 4.6, 0.2),
    "sph": (26.1, 35.4, 35.7),
    "tmmx": (26.8, 35.4, 36.6),
    "vs": (27.2, 33.1, 42.4),
    "NDVI": (27.7, 35.0, 40.0),
    "pr": (28.0, 32.5, 42.0),
    "population": (28.1, 32.8, 43.0),
    "th": (28.3, 32.8, 43.1),
    "tmmn": (28.6, 35.2, 40.9),
    "elevation": (28.8, 33.6, 43.0),
    "pdsi": (28.8, 36.8, 39.0),
    "erc": (28.8, 33.0, 43.2),
}

# Tabla V: red neuronal con el fuego previo + UNA sola variable más.
ARTICULO_TABLA_V = {
    "NDVI": (28.2, 33.0, 41.4),
    "elevation": (27.0, 31.3, 44.4),
    "tmmx": (26.7, 32.7, 43.5),
    "population": (26.6, 32.6, 43.2),
    "pr": (26.6, 34.1, 40.6),
    "tmmn": (26.2, 29.9, 47.2),
    "pdsi": (26.0, 30.8, 44.1),
    "vs": (25.9, 30.3, 45.6),
    "erc": (25.8, 30.7, 45.0),
    "sph": (25.7, 32.4, 41.7),
    "th": (25.5, 30.9, 43.5),
}

# Sección V-B: red neuronal usando SOLO el fuego previo.
ARTICULO_SOLO_FUEGO_PREVIO = (22.9, 29.2, 49.6)

# Tabla VI: las 20 variables más importantes del random forest (Gini).
# El número entre paréntesis es la posición del píxel en la vecindad de 3 x 3
# (1 = arriba-izquierda, 5 = centro, 9 = abajo-derecha).
ARTICULO_TABLA_VI = [
    ("PrevFireMask", 5, 0.092), ("PrevFireMask", 6, 0.074),
    ("PrevFireMask", 4, 0.073), ("PrevFireMask", 2, 0.071),
    ("PrevFireMask", 1, 0.068), ("PrevFireMask", 8, 0.067),
    ("PrevFireMask", 9, 0.061), ("PrevFireMask", 3, 0.052),
    ("PrevFireMask", 7, 0.051), ("erc", 7, 0.017), ("erc", 8, 0.014),
    ("erc", 5, 0.014), ("erc", 6, 0.012), ("erc", 9, 0.010),
    ("erc", 2, 0.008), ("erc", 1, 0.008), ("erc", 4, 0.008),
    ("erc", 3, 0.007), ("elevation", 7, 0.007), ("elevation", 9, 0.005),
]

# Tabla VIII: regresión logística re-entrenada QUITANDO una variable.
ARTICULO_TABLA_VIII = {
    "PrevFireMask": (3.1, 0.0, 0.0),
    "erc": (19.6, 32.4, 35.5),
    "elevation": (19.6, 32.3, 35.5),
    "pdsi": (19.6, 32.4, 35.4),
    "tmmn": (19.6, 32.2, 35.5),
    "th": (19.7, 32.3, 35.7),
    "tmmx": (19.7, 32.3, 35.5),
    "NDVI": (19.8, 32.3, 35.6),
    "pr": (19.8, 32.3, 35.6),
    "population": (19.8, 32.3, 35.6),
    "sph": (19.8, 32.4, 35.6),
    "vs": (20.0, 32.1, 35.7),
}

# Datos generales del dataset según el artículo (Sección II-C).
ARTICULO_DATASET = {
    "muestras": 18545,
    "fuego_crece": 10798,     # 58 %
    "fuego_decrece": 7191,    # 39 %
}
