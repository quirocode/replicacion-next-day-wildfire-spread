# %% [markdown]
# ## Paso 4 — ¿Qué variables necesita la red? (Sección V-B del artículo)
#
# La red no dice por sí sola qué variables usa. El artículo lo averigua con
# **ablación**: volver a entrenar quitando variables y ver cuánto empeora.
#
# Con el presupuesto de este prototipo se replican los dos experimentos que
# sostienen la conclusión principal del artículo:
#
# | Experimento | Artículo (AUC PR / precisión / recall) |
# | --- | --- |
# | Quitar el fuego previo (Tabla IV, 1.ª fila) | 6,8 / 4,6 / 0,2 |
# | Usar solo el fuego previo (texto de la Sección V-B) | 22,9 / 29,2 / 49,6 |
#
# Las otras 11 filas de la Tabla IV y la Tabla V completa (22 entrenamientos
# más) se pueden lanzar agregando entradas al diccionario `EXPERIMENTOS`.

# %%
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import DIR_RESULTADOS, VARIABLES_ENTRADA
from datos import cargar_numpy
from entrenamiento import cargar_red, entrenar_red, evaluar_red, guardar_json

EXPERIMENTOS = {
    # nombre del experimento -> variables de entrada que SÍ se usan
    "red_sin_fuego_previo": [v for v in VARIABLES_ENTRADA if v != "PrevFireMask"],
    "red_solo_fuego_previo": ["PrevFireMask"],
}

# Opcional: correr solo algunos, p. ej. WILDFIRE_EXPERIMENTOS=red_solo_fuego_previo
elegidos = os.environ.get("WILDFIRE_EXPERIMENTOS", ",".join(EXPERIMENTOS)).split(",")

X_prueba, Y_prueba = cargar_numpy("test")

for nombre in elegidos:
    variables = EXPERIMENTOS[nombre]
    if os.path.exists(os.path.join(DIR_RESULTADOS, f"historial_{nombre}.json")):
        print("Se reutiliza el modelo ya entrenado:", nombre)
        modelo = cargar_red(variables, nombre)
    else:
        # Mismo presupuesto y misma semilla que la red completa: así la única
        # diferencia entre los experimentos son las variables de entrada.
        modelo, _ = entrenar_red(variables, nombre)

    metricas = evaluar_red(modelo, X_prueba, Y_prueba, variables)
    guardar_json(metricas, f"metricas_{nombre}.json")
    m = metricas["completo_64"]
    print(f"{nombre}: AUC PR {m['auc_pr']:.1f} | precisión {m['precision']:.1f} | recall {m['recall']:.1f}")
