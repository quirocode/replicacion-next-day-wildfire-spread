# %% [markdown]
# ## Paso 2 — Entrenar la red neuronal (Secciones IV-A y IV-B del artículo)
#
# Se entrena el autoencoder convolucional con las 12 variables de entrada y se
# mide en el conjunto de prueba. El resultado se compara con la primera fila de
# la Tabla II del artículo (AUC PR 28,4 %, precisión 33,6 %, recall 43,1 %).

# %%
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import ARTICULO_TABLA_II, DIR_RESULTADOS, VARIABLES_ENTRADA
from datos import cargar_numpy
from entrenamiento import cargar_red, entrenar_red, evaluar_red, guardar_json

NOMBRE = "red_todas"
terminado = os.path.join(DIR_RESULTADOS, f"historial_{NOMBRE}.json")

# Si el entrenamiento ya terminó se reutiliza el modelo (entrenar tarda). Para
# forzar uno nuevo: borrar resultados/modelos/ y resultados/historial_*.json.
# Si quedó a medias (se cortó la sesión), entrenar_red lo retoma donde quedó.
if os.path.exists(terminado):
    print("Se reutiliza el modelo ya entrenado:", NOMBRE)
    modelo = cargar_red(VARIABLES_ENTRADA, NOMBRE)
else:
    modelo, historial = entrenar_red(VARIABLES_ENTRADA, NOMBRE)

# %% [markdown]
# ### Evaluación en el conjunto de prueba
# Los 1 689 ejemplos de prueba nunca se usaron para entrenar ni para elegir el
# modelo. Se mide sobre la muestra completa de 64 x 64 km, que es el protocolo
# con el que la persistencia reproduce exactamente al artículo (paso 3).

# %%
X_prueba, Y_prueba = cargar_numpy("test")
metricas = evaluar_red(modelo, X_prueba, Y_prueba)
guardar_json(metricas, f"metricas_{NOMBRE}.json")

m = metricas["completo_64"]
a = ARTICULO_TABLA_II["Red neuronal"]
print(f"{'':30s}{'AUC PR':>8s}{'Precisión':>11s}{'Recall':>8s}")
print(f"{'Artículo (Tabla II)':30s}{a[0]:8.1f}{a[1]:11.1f}{a[2]:8.1f}")
print(f"{'Esta replicación (64 x 64)':30s}{m['auc_pr']:8.1f}{m['precision']:11.1f}{m['recall']:8.1f}")
r = metricas["recortes_32"]
print(f"{'(sobre recortes de 32 x 32)':30s}{r['auc_pr']:8.1f}{r['precision']:11.1f}{r['recall']:8.1f}")
print(f"Variación del AUC PR entre recortes distintos: ±{r['auc_pr_desviacion']:.2f} puntos")
