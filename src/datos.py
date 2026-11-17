"""
datos.py — Lectura y preparación del dataset "Next Day Wildfire Spread".

Reproduce la Sección III del artículo ("Data Preprocessing") y el archivo
dataset.py del código de los autores:

    1. Leer los archivos TFRecord publicados en Kaggle.
    2. Aumentar los datos (solo al entrenar): volteo, rotación y recorte
       aleatorio de 32 x 32 km.
    3. Recortar valores extremos y normalizar cada variable.

Cada muestra del dataset es un cuadrado de 64 x 64 km con 12 capas de entrada
y 1 capa de salida (la máscara de fuego del día siguiente).
"""

import glob
import os

import numpy as np
import tensorflow as tf

from config import (
    DATASET_KAGGLE,
    ESTADISTICAS,
    RAIZ,
    SEMILLA,
    TAM_LOTE,
    TAM_ORIGINAL,
    TAM_RECORTE,
    VARIABLE_SALIDA,
    VARIABLES_ENTRADA,
)

# ---------------------------------------------------------------------------
# 1. LOCALIZAR Y LEER LOS ARCHIVOS
# ---------------------------------------------------------------------------


def localizar_dataset():
    """Devuelve la carpeta que contiene los .tfrecord del dataset.

    Busca, en este orden:
      1. La variable de entorno WILDFIRE_DATOS (si el usuario ya los tiene).
      2. /kaggle/input/... (cuando se corre dentro de un notebook de Kaggle).
      3. La carpeta local datos/ del proyecto.
      4. Si no están, los descarga de Kaggle con kagglehub (dataset público,
         no necesita usuario ni contraseña).
    """
    candidatas = [
        os.environ.get("WILDFIRE_DATOS", ""),
        "/kaggle/input/next-day-wildfire-spread",
        os.path.join(RAIZ, "datos"),
    ]
    for carpeta in candidatas:
        if carpeta and glob.glob(os.path.join(carpeta, "*train*.tfrecord")):
            return carpeta

    import kagglehub  # se importa aquí porque solo hace falta para descargar

    return kagglehub.dataset_download(DATASET_KAGGLE)


def archivos(particion):
    """Lista ordenada de TFRecords de una partición: 'train', 'eval' o 'test'.

    La separación 80/10/10 ya viene hecha por los autores (por semanas al azar
    entre 2012 y 2020), así que se respeta tal cual para poder comparar.
    """
    patron = os.path.join(localizar_dataset(), f"*{particion}*.tfrecord")
    lista = sorted(glob.glob(patron))
    if not lista:
        raise FileNotFoundError(f"No se encontraron archivos con el patrón {patron}")
    return lista


def _parsear(ejemplo_serializado):
    """Convierte un registro binario en dos tensores SIN normalizar.

    Returns:
        x: (64, 64, 12) con las variables de entrada en sus unidades originales.
        y: (64, 64, 1) con la máscara de fuego de mañana: 1 fuego, 0 sin fuego,
           -1 incierto (nubes o dato sin procesar).
    """
    nombres = list(VARIABLES_ENTRADA) + [VARIABLE_SALIDA]
    descripcion = {
        n: tf.io.FixedLenFeature([TAM_ORIGINAL, TAM_ORIGINAL], tf.float32)
        for n in nombres
    }
    campos = tf.io.parse_single_example(ejemplo_serializado, descripcion)
    x = tf.stack([campos[n] for n in VARIABLES_ENTRADA], axis=-1)
    y = campos[VARIABLE_SALIDA][..., tf.newaxis]
    return x, y


def leer_crudo(particion):
    """tf.data.Dataset con las muestras de una partición, sin normalizar."""
    return tf.data.TFRecordDataset(archivos(particion)).map(
        _parsear, num_parallel_calls=tf.data.AUTOTUNE
    )


# ---------------------------------------------------------------------------
# 2. LIMPIEZA Y NORMALIZACIÓN  (Sección III del artículo)
# ---------------------------------------------------------------------------


def _vectores_estadisticos(variables):
    """Arma 4 vectores (mín, máx, media, desv.) alineados con `variables`."""
    tabla = np.array([ESTADISTICAS[v] for v in variables], dtype=np.float32)
    return tabla[:, 0], tabla[:, 1], tabla[:, 2], tabla[:, 3]


def normalizar(x, variables=VARIABLES_ENTRADA):
    """Recorta valores extremos y estandariza cada canal: (x - media) / desv.

    El recorte evita valores físicamente imposibles y rangos enormes que
    desestabilizan el entrenamiento. La máscara de fuego previo no cambia
    porque sus "estadísticas" son (-1, 1, 0, 1). Funciona igual con tensores de
    TensorFlow que con arreglos de NumPy.
    """
    minimo, maximo, media, desv = _vectores_estadisticos(variables)
    x = tf.clip_by_value(x, minimo, maximo)
    return (x - media) / desv


def seleccionar_variables(x, variables):
    """Se queda solo con los canales pedidos (para los estudios de ablación)."""
    indices = [VARIABLES_ENTRADA.index(v) for v in variables]
    return tf.gather(x, indices, axis=-1)


# ---------------------------------------------------------------------------
# 3. AUMENTO DE DATOS  (Secciones III y IV-B del artículo)
# ---------------------------------------------------------------------------

_INDICE_VIENTO = VARIABLES_ENTRADA.index("th")


def _cambiar_direccion_viento(x, funcion):
    """Aplica `funcion` al canal de dirección del viento (en grados)."""
    canales = tf.unstack(x, axis=-1)
    canales[_INDICE_VIENTO] = funcion(canales[_INDICE_VIENTO])
    return tf.stack(canales, axis=-1)


def aumentar(x, y):
    """Volteo y rotación al azar de una muestra SIN normalizar.

    Voltear o girar la imagen también cambia hacia dónde "apunta" el viento,
    así que el canal de dirección (grados en sentido horario desde el norte)
    se corrige con las mismas fórmulas del código de los autores
    (image_utils.py). Por eso se aumenta ANTES de normalizar: las fórmulas
    necesitan los grados reales.

    Entrada y etiqueta se transforman juntas para que sigan alineadas.
    """
    # Volteo izquierda-derecha (50 %): el este pasa a ser oeste.
    if tf.random.uniform(()) < 0.5:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
        x = _cambiar_direccion_viento(x, lambda g: (360.0 - g) % 360.0)

    # Volteo arriba-abajo (50 %): el norte pasa a ser sur.
    if tf.random.uniform(()) < 0.5:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)
        x = _cambiar_direccion_viento(x, lambda g: (180.0 - g) % 360.0)

    # Rotación de 0, 90, 180 o 270 grados en sentido antihorario.
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    y = tf.image.rot90(y, k)
    giro = 90.0 * tf.cast(k, tf.float32)
    x = _cambiar_direccion_viento(x, lambda g: (g - giro) % 360.0)
    return x, y


def recorte_aleatorio(x, y, tam=TAM_RECORTE):
    """Recorta la misma ventana de tam x tam en la entrada y en la etiqueta.

    En el dataset el fuego casi siempre está en el centro del cuadro de
    64 x 64 km. El recorte aleatorio lo mueve de sitio, para que la red no
    aprenda simplemente "el fuego está en el medio" (Sección III).
    """
    canales_x = x.shape[-1]
    junto = tf.concat([x, y], axis=-1)
    junto = tf.image.random_crop(junto, [tam, tam, junto.shape[-1]])
    return junto[..., :canales_x], junto[..., canales_x:]


# ---------------------------------------------------------------------------
# 4. DATASETS LISTOS PARA USAR
# ---------------------------------------------------------------------------

def dataset_entrenamiento(variables=VARIABLES_ENTRADA, lote=TAM_LOTE, semilla=SEMILLA,
                          tam_barajado=2000):
    """Flujo infinito de lotes para entrenar la red neuronal.

    barajar archivos -> leer 4 archivos a la vez -> barajar muestras
        -> aumentar -> recortar -> normalizar -> elegir variables -> lotes

    Los datos se leen del disco en cada pasada (no se guardan en memoria), así
    el entrenamiento ocupa poca RAM. El orden se mezcla dos veces: primero el
    de los 15 archivos y luego el de las muestras, con una ventana de
    `tam_barajado` muestras (igual que el `shuffle_buffer_size` de los autores).
    """

    def preparar(x, y):
        x, y = aumentar(x, y)
        x, y = recorte_aleatorio(x, y)
        x = normalizar(x)
        x = seleccionar_variables(x, variables)
        return x, y

    rutas = tf.data.Dataset.from_tensor_slices(archivos("train"))
    rutas = rutas.shuffle(len(archivos("train")), seed=semilla).repeat()
    ds = rutas.interleave(
        tf.data.TFRecordDataset, cycle_length=4, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(_parsear, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(tam_barajado, seed=semilla)
    ds = ds.map(preparar, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(lote).prefetch(tf.data.AUTOTUNE)


def cargar_numpy(particion):
    """Carga una partición completa en memoria, ya normalizada.

    Se usa para validación y prueba (son pequeñas: ~1 900 y ~1 700 muestras).

    Returns:
        X: (N, 64, 64, 12) float32 normalizado.
        Y: (N, 64, 64, 1) float32 con valores -1, 0 o 1.
    """
    xs, ys = [], []
    for x, y in leer_crudo(particion).batch(512):
        xs.append(normalizar(x).numpy())
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def iterar_numpy(particion, tam_bloque=1024):
    """Recorre una partición por bloques normalizados (sin cargarla entera).

    Lo usan los modelos base para muestrear píxeles del conjunto de
    entrenamiento sin ocupar los ~3 GB que pesa completo.
    """
    for x, y in leer_crudo(particion).batch(tam_bloque):
        yield normalizar(x).numpy(), y.numpy()


def recortes_fijos(X, Y, tam=TAM_RECORTE, repeticiones=1, semilla=SEMILLA):
    """Recortes aleatorios pero REPETIBLES (misma semilla = mismos recortes).

    Los autores evalúan sobre recortes aleatorios de 32 x 32 km (dataset.py,
    modo PREDICT con random_crop). Fijar la semilla permite que todos los
    modelos se midan exactamente sobre los mismos píxeles.

    Args:
        repeticiones: cuántos recortes distintos sacar de cada muestra.

    Returns:
        Xr: (N * repeticiones, tam, tam, C)
        Yr: (N * repeticiones, tam, tam, 1)
    """
    azar = np.random.default_rng(semilla)
    n, alto, ancho, _ = X.shape
    xs, ys = [], []
    for _ in range(repeticiones):
        filas = azar.integers(0, alto - tam + 1, size=n)
        columnas = azar.integers(0, ancho - tam + 1, size=n)
        xs.append(np.stack([X[i, f:f + tam, c:c + tam] for i, (f, c) in enumerate(zip(filas, columnas))]))
        ys.append(np.stack([Y[i, f:f + tam, c:c + tam] for i, (f, c) in enumerate(zip(filas, columnas))]))
    return np.concatenate(xs), np.concatenate(ys)
