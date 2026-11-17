"""
modelos_base.py — Modelos de comparación: persistencia, regresión logística y
random forest.

Reproduce la Sección IV-A y el Apéndice A-B del artículo.

A diferencia de la red, estos modelos NO ven la imagen completa: clasifican
píxel por píxel. Para darles algo de contexto espacial, cada píxel se describe
con sus 12 variables y las de sus 8 vecinos (un cuadrado de 3 x 3):

        1 2 3
        4 5 6      9 posiciones x 12 variables = 108 columnas
        7 8 9      (la posición 5 es el propio píxel)

Detalles tomados del artículo:
  - Entrenamiento balanceado ~1:1 entre píxeles con y sin fuego.
  - Se excluye el borde de 1 píxel (no tiene los 8 vecinos).
  - Los píxeles inciertos (-1) no se usan.
  - La clase "sin fuego" pesa (N / N_sin_fuego) x W, con W = 7 para la
    regresión logística y W = 5 para el random forest.
  - Random forest con profundidad máxima 15; lo demás, valores por defecto
    de Scikit-learn.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from config import (
    PROFUNDIDAD_BOSQUE,
    SEMILLA,
    VARIABLES_ENTRADA,
)
from datos import iterar_numpy

NUM_VARIABLES = len(VARIABLES_ENTRADA)
_INDICE_FUEGO_PREVIO = VARIABLES_ENTRADA.index("PrevFireMask")

# Desplazamientos (fila, columna) de las 9 posiciones, en el orden 1..9.
_VECINOS = [(df, dc) for df in (-1, 0, 1) for dc in (-1, 0, 1)]


def nombres_columnas():
    """Lista de 108 pares (variable, posición 1..9), en el orden de las columnas."""
    return [(v, pos + 1) for pos in range(9) for v in VARIABLES_ENTRADA]


def columnas_de(variable):
    """Índices de las 9 columnas (una por posición) de una variable."""
    c = VARIABLES_ENTRADA.index(variable)
    return [pos * NUM_VARIABLES + c for pos in range(9)]


# ---------------------------------------------------------------------------
# 1. PERSISTENCIA: "mañana habrá fuego donde hoy hay fuego"
# ---------------------------------------------------------------------------


def predecir_persistencia(X):
    """Devuelve la máscara de fuego de hoy como predicción para mañana.

    No aprende nada: es el piso que cualquier modelo debe superar. Los píxeles
    inciertos de hoy (-1) se cuentan como "sin fuego".
    """
    return np.clip(X[..., _INDICE_FUEGO_PREVIO:_INDICE_FUEGO_PREVIO + 1], 0.0, 1.0)


# ---------------------------------------------------------------------------
# 2. DE IMÁGENES A TABLAS DE PÍXELES
# ---------------------------------------------------------------------------


def vecindades_de_pixeles(X, n, fila, columna):
    """Columnas 3 x 3 de un conjunto de píxeles elegidos.

    Args:
        X: (N, alto, ancho, 12).
        n, fila, columna: índices de los píxeles (deben estar fuera del borde).

    Returns:
        (cantidad de píxeles, 108)
    """
    partes = [X[n, fila + df, columna + dc, :] for df, dc in _VECINOS]
    return np.concatenate(partes, axis=1)


def vecindades_completas(X):
    """Columnas 3 x 3 de TODOS los píxeles interiores de cada imagen.

    Returns:
        (N * (alto - 2) * (ancho - 2), 108)
    """
    _, alto, ancho, _ = X.shape
    partes = [
        X[:, 1 + df:alto - 1 + df, 1 + dc:ancho - 1 + dc, :] for df, dc in _VECINOS
    ]
    return np.concatenate(partes, axis=-1).reshape(-1, 9 * NUM_VARIABLES)


def muestrear_pixeles_entrenamiento(max_pixeles, semilla=SEMILLA):
    """Arma la tabla de entrenamiento balanceada 1:1 a partir de 'train'.

    Recorre el conjunto de entrenamiento por bloques. De cada bloque toma
    todos los píxeles con fuego y la misma cantidad de píxeles sin fuego
    elegidos al azar. Al final, si hay más de `max_pixeles`, se queda con una
    muestra aleatoria que mantiene el balance.

    Returns:
        caracteristicas: (M, 108) float32
        etiquetas: (M,) con 0 (sin fuego) o 1 (fuego)
    """
    azar = np.random.default_rng(semilla)
    con_fuego, sin_fuego = [], []
    for X, Y in iterar_numpy("train", tam_bloque=512):
        sin_borde = Y[:, 1:-1, 1:-1, 0]
        for valor, destino in ((1, con_fuego), (0, sin_fuego)):
            n, f, c = np.nonzero(sin_borde == valor)
            if valor == 0:  # igualar la cantidad de píxeles con fuego del bloque
                cuantos = min(int(np.sum(sin_borde == 1)), len(n))
                elegidos = azar.choice(len(n), size=cuantos, replace=False)
                n, f, c = n[elegidos], f[elegidos], c[elegidos]
            # +1 porque `sin_borde` empieza en la fila/columna 1 de la imagen.
            destino.append(vecindades_de_pixeles(X, n, f + 1, c + 1))

    con_fuego, sin_fuego = np.concatenate(con_fuego), np.concatenate(sin_fuego)
    por_clase = min(len(con_fuego), len(sin_fuego), max_pixeles // 2)
    con_fuego = con_fuego[azar.choice(len(con_fuego), por_clase, replace=False)]
    sin_fuego = sin_fuego[azar.choice(len(sin_fuego), por_clase, replace=False)]

    caracteristicas = np.concatenate([con_fuego, sin_fuego])
    etiquetas = np.concatenate([np.ones(por_clase), np.zeros(por_clase)]).astype(np.int8)
    orden = azar.permutation(len(etiquetas))
    return caracteristicas[orden], etiquetas[orden]


# ---------------------------------------------------------------------------
# 3. ENTRENAR Y PREDECIR
# ---------------------------------------------------------------------------


def pesos_de_clase(etiquetas, w):
    """Peso de la clase "sin fuego" según el Apéndice A-B del artículo.

        peso_sin_fuego = (número de ejemplos / número de ejemplos sin fuego) x W

    Con datos balanceados 1:1 la fracción vale 2, así que el peso es 2 x W.
    Subir W hace al modelo más "prudente": marca menos fuego, con lo que sube
    la precisión y baja el recall.
    """
    fraccion = len(etiquetas) / max(int(np.sum(etiquetas == 0)), 1)
    return {0: fraccion * w, 1: 1.0}


def entrenar_logistica(caracteristicas, etiquetas, w):
    """Regresión logística con los hiperparámetros por defecto de Scikit-learn."""
    modelo = LogisticRegression(class_weight=pesos_de_clase(etiquetas, w))
    return modelo.fit(caracteristicas, etiquetas)


def entrenar_bosque(caracteristicas, etiquetas, w, nucleos=-1):
    """Random forest con profundidad máxima 15 (Apéndice A-B)."""
    modelo = RandomForestClassifier(
        max_depth=PROFUNDIDAD_BOSQUE,
        class_weight=pesos_de_clase(etiquetas, w),
        n_jobs=nucleos,
        random_state=SEMILLA,
    )
    return modelo.fit(caracteristicas, etiquetas)


def predecir_por_pixel(modelo, X, columnas=None, tam_bloque=256):
    """Probabilidad de fuego para cada píxel interior de cada imagen.

    Args:
        modelo: clasificador de Scikit-learn ya entrenado.
        X: (N, alto, ancho, 12).
        columnas: subconjunto de las 108 columnas (para la ablación).

    Returns:
        (N, alto - 2, ancho - 2, 1)
    """
    n, alto, ancho, _ = X.shape
    salidas = []
    for inicio in range(0, n, tam_bloque):  # por bloques, para no saturar la RAM
        tabla = vecindades_completas(X[inicio:inicio + tam_bloque])
        if columnas is not None:
            tabla = tabla[:, columnas]
        salidas.append(modelo.predict_proba(tabla)[:, 1])
    return np.concatenate(salidas).reshape(n, alto - 2, ancho - 2, 1).astype(np.float32)


def interior(Y):
    """Quita el borde de 1 píxel a las etiquetas, igual que a las predicciones."""
    return Y[:, 1:-1, 1:-1, :]
