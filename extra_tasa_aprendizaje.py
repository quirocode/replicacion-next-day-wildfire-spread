# %% [markdown]
# ## Experimento adicional (NO está en el artículo): una tasa de aprendizaje mayor
#
# El artículo entrena con tasa 0,0001 durante 1 000 000 de pasos. Este prototipo
# solo puede dar 28 080. Una tasa tan baja avanza poco en tan pocos pasos, así
# que aquí se prueba una tasa 10 veces mayor (0,001) con la mitad de las épocas.
#
# Sirve para responder una pregunta concreta: ¿la distancia entre nuestra red y
# la del artículo se debe al presupuesto de entrenamiento? Si con una tasa mayor
# la misma red mejora mucho, la respuesta es sí.
#
# Este resultado NO se usa en la tabla comparativa: ahí va la red entrenada con
# los hiperparámetros del artículo.

# %%
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import DIR_RESULTADOS, PRESUPUESTO, PRESUPUESTOS, VARIABLES_ENTRADA
from datos import cargar_numpy
from entrenamiento import cargar_red, entrenar_red, evaluar_red, guardar_json

NOMBRE = "red_todas_tasa_0001"
EPOCAS = max(1, PRESUPUESTOS[PRESUPUESTO]["epocas"] // 2)

if os.path.exists(os.path.join(DIR_RESULTADOS, f"historial_{NOMBRE}.json")):
    modelo = cargar_red(VARIABLES_ENTRADA, NOMBRE)
else:
    modelo, _ = entrenar_red(VARIABLES_ENTRADA, NOMBRE, epocas=EPOCAS, tasa_aprendizaje=1e-3)

X_prueba, Y_prueba = cargar_numpy("test")
metricas = evaluar_red(modelo, X_prueba, Y_prueba)
guardar_json(metricas, f"metricas_{NOMBRE}.json")
m = metricas["completo_64"]
print(f"Tasa 0,001, {EPOCAS} épocas: AUC PR {m['auc_pr']:.1f} | precisión {m['precision']:.1f} | recall {m['recall']:.1f}")
