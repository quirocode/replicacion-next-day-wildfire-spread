"""
metricas.py — Función de pérdida y métricas, ignorando los píxeles inciertos.

Reproduce models/losses.py y models/metrics.py del código de los autores y la
Sección IV-B del artículo.

Dos ideas clave:

1. DESBALANCE. Solo ~1 % de los píxeles tiene fuego. Una red que siempre diga
   "no hay fuego" acertaría el 99 % de las veces y no serviría para nada. Por
   eso (a) la pérdida castiga más los errores en píxeles con fuego (peso 3) y
   (b) no se usa accuracy sino AUC PR, precisión y recall.

2. ETIQUETAS INCIERTAS. Los píxeles con valor -1 (nubes, dato faltante) no
   cuentan ni para la pérdida ni para las métricas.
"""

import keras
import numpy as np
import tensorflow as tf

from config import PESO_CLASE_FUEGO


def perdida_ponderada_con_mascara(peso_fuego=PESO_CLASE_FUEGO):
    """Entropía cruzada binaria con peso para la clase fuego y máscara.

    Para cada píxel con etiqueta z (0 o 1) y logit x:
        pérdida = -[ peso * z * log(sigmoide(x)) + (1 - z) * log(1 - sigmoide(x)) ]
    Los píxeles inciertos (z = -1) se multiplican por 0.
    """

    def perdida(y_verdad, logits):
        mascara = tf.cast(tf.not_equal(y_verdad, -1.0), tf.float32)
        etiquetas = tf.maximum(y_verdad, 0.0)  # el -1 pasa a 0; igual se anula
        por_pixel = tf.nn.weighted_cross_entropy_with_logits(
            labels=etiquetas, logits=logits, pos_weight=peso_fuego
        )
        return tf.reduce_mean(mascara * por_pixel)

    return perdida


class _ConMascara:
    """Mezcla que adapta una métrica de Keras: aplica sigmoide y máscara."""

    def update_state(self, y_verdad, logits, sample_weight=None):
        probabilidad = tf.sigmoid(logits)
        mascara = tf.cast(tf.not_equal(y_verdad, -1.0), tf.float32)
        super().update_state(
            tf.maximum(y_verdad, 0.0), probabilidad, sample_weight=mascara
        )


class AUCConMascara(_ConMascara, keras.metrics.AUC):
    """AUC que ignora los píxeles inciertos (se usa con curve='PR')."""


class PrecisionConMascara(_ConMascara, keras.metrics.Precision):
    """De los píxeles que el modelo marcó como fuego, ¿cuántos lo eran?"""


class RecallConMascara(_ConMascara, keras.metrics.Recall):
    """De los píxeles que sí tenían fuego, ¿cuántos encontró el modelo?"""


def metricas_de_entrenamiento():
    """Métricas que Keras calcula en cada época (los nombres van al historial)."""
    return [
        AUCConMascara(curve="PR", name="auc_pr"),
        PrecisionConMascara(name="precision"),
        RecallConMascara(name="recall"),
    ]


def calcular_metricas(y_verdad, probabilidad, umbral=0.5):
    """Calcula AUC PR, precisión y recall (en %) para CUALQUIER modelo.

    Se usa la misma métrica de Keras que los autores (tf.keras.metrics.AUC con
    curve='PR' y 200 umbrales), de modo que la red, la regresión logística, el
    random forest y la persistencia se midan con la misma regla.

    Args:
        y_verdad: arreglo con valores -1 (incierto), 0 (sin fuego) o 1 (fuego).
        probabilidad: arreglo de la misma forma con valores entre 0 y 1.
        umbral: desde qué probabilidad se declara "fuego" (0.5, como Keras).

    Returns:
        dict con auc_pr, precision y recall en porcentaje, y los conteos.
    """
    y = np.asarray(y_verdad).reshape(-1)
    p = np.asarray(probabilidad, dtype=np.float32).reshape(-1)
    validos = y != -1  # se descartan los píxeles inciertos
    y, p = y[validos].astype(np.float32), np.clip(p[validos], 0.0, 1.0)

    auc = keras.metrics.AUC(curve="PR")
    for inicio in range(0, len(y), 2_000_000):  # por bloques, para no saturar la RAM
        auc.update_state(y[inicio:inicio + 2_000_000], p[inicio:inicio + 2_000_000])

    marcado = p > umbral
    verdaderos_positivos = float(np.sum(marcado & (y == 1)))
    falsos_positivos = float(np.sum(marcado & (y == 0)))
    falsos_negativos = float(np.sum(~marcado & (y == 1)))
    precision = verdaderos_positivos / max(verdaderos_positivos + falsos_positivos, 1.0)
    recall = verdaderos_positivos / max(verdaderos_positivos + falsos_negativos, 1.0)

    return {
        "auc_pr": 100.0 * float(auc.result()),
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "pixeles_validos": int(len(y)),
        "pixeles_con_fuego": int(np.sum(y == 1)),
    }


def reducir_resolucion(mascara, factor):
    """Pasa una máscara de 1 km a una de `factor` km (etiqueta "binaria").

    Regla del artículo (Sección V-A, Tabla III): un bloque de factor x factor
    píxeles es "fuego" si AL MENOS UNO de sus píxeles lo es. Si todos los
    píxeles del bloque son inciertos, el bloque queda incierto (-1).

    Args:
        mascara: (N, alto, ancho, 1) con valores -1, 0 o 1.

    Returns:
        (N, alto / factor, ancho / factor, 1) con valores -1, 0 o 1.
    """
    n, alto, ancho, _ = mascara.shape
    bloques = mascara.reshape(n, alto // factor, factor, ancho // factor, factor)
    hay_fuego = (bloques == 1).any(axis=(2, 4))
    hay_dato = (bloques != -1).any(axis=(2, 4))
    salida = np.where(hay_fuego, 1.0, np.where(hay_dato, 0.0, -1.0))
    return salida[..., np.newaxis].astype(np.float32)
