# %% [markdown]
# ## Paso 3 — Modelos base (Sección IV-A y Apéndices A-B, A-C y A-D)
#
# Se entrenan los tres modelos con los que el artículo compara a la red:
#
# | Modelo | Qué hace | Artículo, Tabla II (AUC PR / precisión / recall) |
# | --- | --- | --- |
# | Persistencia | "mañana = hoy" | 11,5 / 35,7 / 27,3 |
# | Regresión logística | píxel a píxel con vecindad 3 x 3 | 19,8 / 32,5 / 35,3 |
# | Random forest | píxel a píxel con vecindad 3 x 3 | 22,5 / 26,3 / 46,9 |

# %%
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from config import (
    MAX_PIXELES_BOSQUE,
    MAX_PIXELES_LOGISTICA,
    VARIABLES_ENTRADA,
    W_BOSQUE,
    W_LOGISTICA,
)
from datos import cargar_numpy, recortes_fijos
from entrenamiento import guardar_json
from metricas import calcular_metricas, reducir_resolucion
from modelos_base import (
    columnas_de,
    entrenar_bosque,
    entrenar_logistica,
    interior,
    muestrear_pixeles_entrenamiento,
    nombres_columnas,
    predecir_persistencia,
    predecir_por_pixel,
)

# La regresión logística usa los valores por defecto de Scikit-learn (como en el
# artículo), que incluyen un máximo de 100 iteraciones. Algunos ajustes llegan a
# ese tope y Scikit-learn avisa; se silencia el aviso para no llenar la salida.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Conjunto de prueba: las 1 689 muestras completas de 64 x 64 km.
X_prueba, Y_prueba = cargar_numpy("test")

# Conjunto de validación, para elegir el peso W como hace el artículo.
X_val, Y_val = cargar_numpy("eval")

# %% [markdown]
# ### 3.1 Persistencia (y verificación del protocolo de evaluación)
# La persistencia no se entrena, así que su resultado solo depende de los datos
# y de cómo se mide. Si coincide con el artículo, el protocolo de evaluación
# está bien replicado. Se mide sobre la muestra completa de 64 x 64 km y
# también a resolución más gruesa (filas "Persistence" de la Tabla III).

# %%
CANAL_FUEGO = VARIABLES_ENTRADA.index("PrevFireMask")
persistencia = {"1x": calcular_metricas(Y_prueba, predecir_persistencia(X_prueba))}
for factor in (2, 4, 8):
    hoy = reducir_resolucion(X_prueba[..., CANAL_FUEGO, None], factor)
    manana = reducir_resolucion(Y_prueba, factor)
    persistencia[f"{factor}x"] = calcular_metricas(manana, np.clip(hoy, 0, 1))

# Mismo cálculo sobre recortes de 32 x 32 km: el número cambia solo por medir
# sobre otros píxeles (el fuego suele estar hacia el centro de la muestra).
X_rec, Y_rec = recortes_fijos(X_prueba, Y_prueba, repeticiones=3)
persistencia["recortes_32"] = calcular_metricas(Y_rec, predecir_persistencia(X_rec))
del X_rec, Y_rec

guardar_json(persistencia, "metricas_persistencia.json")
for clave, m in persistencia.items():
    print(f"Persistencia {clave:12s} AUC PR {m['auc_pr']:5.1f} | precisión {m['precision']:5.1f} | recall {m['recall']:5.1f}")

# %% [markdown]
# ### 3.2 Tabla de píxeles para entrenar
# Las imágenes se convierten en una tabla: una fila por píxel, 108 columnas.

# %%
inicio = time.time()
tabla, etiquetas = muestrear_pixeles_entrenamiento(MAX_PIXELES_LOGISTICA)
print(f"Tabla de entrenamiento: {tabla.shape[0]:,} píxeles x {tabla.shape[1]} columnas "
      f"({etiquetas.mean():.0%} con fuego) en {time.time() - inicio:.0f} s")

# %% [markdown]
# ### 3.3 Regresión logística
# El artículo prueba W de 1 a 8 y se queda con el mejor AUC PR de validación
# (obtuvo W = 7). Aquí se repite esa búsqueda.

# %%
busqueda_w = {}
for w in range(1, 9):
    candidato = entrenar_logistica(tabla, etiquetas, w)
    busqueda_w[w] = calcular_metricas(interior(Y_val), predecir_por_pixel(candidato, X_val))
    m = busqueda_w[w]
    print(f"W = {w}: validación AUC PR {m['auc_pr']:.2f} | precisión {m['precision']:.1f} | recall {m['recall']:.1f}")
mejor_w = max(busqueda_w, key=lambda w: busqueda_w[w]["auc_pr"])
print(f"Mejor W por AUC PR de validación: {mejor_w} (el artículo eligió {W_LOGISTICA})")

# Para la tabla comparativa se usa el W del artículo (7), de modo que el punto
# de operación (precisión/recall) sea comparable con su Tabla II.
logistica = entrenar_logistica(tabla, etiquetas, W_LOGISTICA)
metricas_logistica = calcular_metricas(interior(Y_prueba), predecir_por_pixel(logistica, X_prueba))
# Como referencia, también con el W que salió mejor en nuestra validación.
metricas_mejor_w = calcular_metricas(
    interior(Y_prueba),
    predecir_por_pixel(entrenar_logistica(tabla, etiquetas, mejor_w), X_prueba),
)
guardar_json(
    {
        "prueba": metricas_logistica,
        "w_usado": W_LOGISTICA,
        "mejor_w_validacion": mejor_w,
        "prueba_con_mejor_w": metricas_mejor_w,
        "busqueda_w_validacion": busqueda_w,
        "pixeles_entrenamiento": int(len(etiquetas)),
        "iteraciones": int(logistica.n_iter_[0]),
    },
    "metricas_logistica.json",
)
print("Regresión logística (prueba):", {k: round(v, 1) for k, v in metricas_logistica.items() if k in ("auc_pr", "precision", "recall")})

# %% [markdown]
# ### 3.4 Coeficientes de la regresión logística (Tabla VII del artículo)
# Como las variables están normalizadas, el tamaño del coeficiente indica
# cuánto pesa cada columna. El signo indica si empuja hacia "fuego" (+) o no (-).

# %%
columnas = nombres_columnas()
orden = np.argsort(-np.abs(logistica.coef_[0]))[:20]
coeficientes = [
    {"variable": columnas[i][0], "posicion": columnas[i][1], "coeficiente": float(logistica.coef_[0][i])}
    for i in orden
]
guardar_json(coeficientes, "coeficientes_logistica.json")
for c in coeficientes[:10]:
    print(f"{c['variable']:14s} ({c['posicion']})  {c['coeficiente']:+.2f}")

# %% [markdown]
# ### 3.5 Ablación de la regresión logística (Tabla VIII del artículo)
# Se vuelve a entrenar 12 veces, quitando cada vez una variable (sus 9 columnas).

# %%
ablacion_logistica = {}
todas = np.arange(tabla.shape[1])
for variable in VARIABLES_ENTRADA:
    quedan = np.setdiff1d(todas, columnas_de(variable))
    sin_variable = entrenar_logistica(tabla[:, quedan], etiquetas, W_LOGISTICA)
    prob = predecir_por_pixel(sin_variable, X_prueba, columnas=quedan)
    ablacion_logistica[variable] = calcular_metricas(interior(Y_prueba), prob)
    m = ablacion_logistica[variable]
    print(f"Sin {variable:14s} AUC PR {m['auc_pr']:5.1f} | precisión {m['precision']:5.1f} | recall {m['recall']:5.1f}")
guardar_json(ablacion_logistica, "ablacion_logistica.json")

# %% [markdown]
# ### 3.6 Random forest
# Se entrena con una muestra más pequeña de la tabla (ver `MAX_PIXELES_BOSQUE`)
# porque es el modelo más lento en CPU. Se usa el W del artículo (5).

# %%
azar = np.random.default_rng(0)
# Muestra balanceada: mitad de filas con fuego y mitad sin fuego.
filas = np.concatenate([
    azar.choice(np.nonzero(etiquetas == clase)[0], MAX_PIXELES_BOSQUE // 2, replace=False)
    for clase in (0, 1)
])
inicio = time.time()
bosque = entrenar_bosque(tabla[filas], etiquetas[filas], W_BOSQUE)
print(f"Random forest entrenado con {len(filas):,} píxeles en {(time.time() - inicio) / 60:.1f} min")

metricas_bosque = calcular_metricas(interior(Y_prueba), predecir_por_pixel(bosque, X_prueba))
guardar_json(
    {"prueba": metricas_bosque, "w_usado": W_BOSQUE, "pixeles_entrenamiento": int(len(filas))},
    "metricas_bosque.json",
)
print("Random forest (prueba):", {k: round(v, 1) for k, v in metricas_bosque.items() if k in ("auc_pr", "precision", "recall")})

# %% [markdown]
# ### 3.7 Importancia de variables del random forest (Tabla VI del artículo)
# La importancia de Gini mide cuánto ayuda cada columna a separar "fuego" de
# "sin fuego" en los árboles. Todas las importancias suman 1.

# %%
orden = np.argsort(-bosque.feature_importances_)
importancia = [
    {"variable": columnas[i][0], "posicion": columnas[i][1], "importancia": float(bosque.feature_importances_[i])}
    for i in orden
]
guardar_json(importancia, "importancia_bosque.json")
for item in importancia[:10]:
    print(f"{item['variable']:14s} ({item['posicion']})  {item['importancia']:.3f}")
