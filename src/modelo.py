"""
modelo.py — Autoencoder convolucional para segmentar "fuego de mañana".

Reproduce la Fig. 6 y el Apéndice A-A del artículo. La estructura de capas se
tomó del código de los autores (models/cnn_autoencoder_model.py y
models/model_utils.py) y se reescribió para Keras 3.

Idea general (Fig. 2 del artículo):

    entrada 32x32x12 ──► CODIFICADOR ──► cuello de botella 8x8x32 ──► DECODIFICADOR ──► salida 32x32x1
                         (comprime)                                   (reconstruye)

La entrada es una "foto" de 12 canales (11 variables + el fuego de hoy) y la
salida es un mapa con un número por píxel (logit). Al pasarlo por una sigmoide
se convierte en la probabilidad de que ese km² tenga fuego mañana.
"""

import keras
from keras import layers

from config import (
    FILTROS_CODIFICADOR,
    FILTROS_DECODIFICADOR,
    TASA_DROPOUT,
)

# Pendiente de la LeakyReLU. 0.3 era el valor por defecto de Keras cuando se
# escribió el código original; se fija aquí para que no dependa de la versión.
PENDIENTE_LEAKY = 0.3


def conv(filtros, tam_nucleo=3):
    """Convolución 2D como la definen los autores.

    Núcleo de 3x3, paso 1, relleno 'same' (no cambia el tamaño de la imagen) y
    SIN término de sesgo (use_bias=False en model_utils.py).
    """
    return layers.Conv2D(
        filtros, tam_nucleo, strides=1, padding="same", use_bias=False
    )


def bloque_inicial(x, filtros, dropout):
    """Primer bloque del codificador (parte superior izquierda de la Fig. 6).

    Camino principal:  Conv -> LeakyReLU -> Dropout -> Conv
    Atajo (shortcut):  Conv 1x1 -> Dropout
    Al final se suman. El atajo deja pasar la entrada casi intacta, lo que
    facilita el entrenamiento (conexión residual).
    """
    principal = conv(filtros)(x)
    principal = layers.LeakyReLU(negative_slope=PENDIENTE_LEAKY)(principal)
    principal = layers.Dropout(dropout)(principal)
    principal = conv(filtros)(principal)

    atajo = conv(filtros, tam_nucleo=1)(x)
    atajo = layers.Dropout(dropout)(atajo)
    return layers.Add()([atajo, principal])


def bloque_residual(x, filtros, dropout, reducir):
    """ResBlock de la Fig. 6 (derecha).

    reducir=True  -> variante "Maxpool": baja la resolución a la mitad.
                     Se usa en el codificador y en el cuello de botella.
    reducir=False -> variante "Conv2D": mantiene la resolución.
                     Se usa en el decodificador.

    Camino principal:
        LeakyReLU -> Dropout -> (MaxPool 2x2  o  Conv 3x3)
        LeakyReLU -> Dropout -> Conv 3x3 -> Dropout
    Atajo:
        Conv 1x1 (con paso 2 si se reduce, para que los tamaños coincidan)
    """
    principal = layers.LeakyReLU(negative_slope=PENDIENTE_LEAKY)(x)
    principal = layers.Dropout(dropout)(principal)
    if reducir:
        principal = layers.MaxPooling2D(pool_size=2, strides=2)(principal)
    else:
        principal = conv(filtros)(principal)

    principal = layers.LeakyReLU(negative_slope=PENDIENTE_LEAKY)(principal)
    principal = layers.Dropout(dropout)(principal)
    principal = conv(filtros)(principal)
    principal = layers.Dropout(dropout)(principal)

    atajo = layers.Conv2D(
        filtros, 1, strides=2 if reducir else 1, padding="same", use_bias=False
    )(x)
    return layers.Add()([atajo, principal])


def crear_modelo(
    num_canales,
    filtros_codificador=FILTROS_CODIFICADOR,
    filtros_decodificador=FILTROS_DECODIFICADOR,
    dropout=TASA_DROPOUT,
):
    """Construye el autoencoder convolucional del artículo.

    Args:
        num_canales: número de variables de entrada (12 con todas; menos en
            los experimentos de ablación).
        filtros_codificador: filtros del bloque inicial y de cada ResBlock del
            codificador. (16, 32) = configuración del artículo.
        filtros_decodificador: filtros de cada ResBlock del decodificador.
        dropout: fracción de neuronas que se apagan al entrenar.

    Returns:
        keras.Model que recibe (alto, ancho, num_canales) y devuelve
        (alto, ancho, 1) con LOGITS (sin sigmoide).

    El alto y el ancho se dejan libres (None): como la red solo tiene
    convoluciones, se puede entrenar con recortes de 32x32 y luego evaluar con
    la imagen completa de 64x64. Solo deben ser múltiplos de 4.
    """
    entrada = keras.Input(shape=(None, None, num_canales), name="variables")

    # ---- CODIFICADOR: 32x32 -> 16x16 ------------------------------------
    x = bloque_inicial(entrada, filtros_codificador[0], dropout)
    for filtros in filtros_codificador[1:]:
        x = bloque_residual(x, filtros, dropout, reducir=True)

    # ---- CUELLO DE BOTELLA: 16x16 -> 8x8 --------------------------------
    # Segundo "ResBlock (Maxpool)" de la Fig. 6.
    x = bloque_residual(x, filtros_codificador[-1], dropout, reducir=True)

    # ---- DECODIFICADOR: 8x8 -> 16x16 -> 32x32 ---------------------------
    # Cada etapa agranda la imagen al doble (vecino más cercano) y la refina
    # con un "ResBlock (Conv2D)".
    for filtros in filtros_decodificador:
        x = layers.UpSampling2D(size=2, interpolation="nearest")(x)
        x = bloque_residual(x, filtros, dropout, reducir=False)

    # ---- CAPA DE SALIDA: 1 filtro = 1 logit por píxel -------------------
    salida = conv(1, tam_nucleo=1)(x)
    return keras.Model(entrada, salida, name="autoencoder_convolucional")
