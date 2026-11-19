# %% [markdown]
# ## Paso 5 — Comparar con el artículo
#
# Se juntan los resultados de los pasos anteriores y se ponen al lado de los
# que publica el artículo. Salidas:
#
# - `resultados/comparacion_resultados.md`: todas las tablas comparativas.
# - `fig2` a `fig5`: curvas de entrenamiento, predicciones, comparación de la
#   Tabla II e importancia de variables.

# %%
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from config import (
    ARTICULO_SOLO_FUEGO_PREVIO,
    ARTICULO_TABLA_II,
    ARTICULO_TABLA_III_PERSISTENCIA,
    ARTICULO_TABLA_IV,
    ARTICULO_TABLA_VI,
    ARTICULO_TABLA_VIII,
    DIR_RESULTADOS,
    NOMBRES_ES,
    VARIABLES_ENTRADA,
)
from datos import cargar_numpy, recortes_fijos
from entrenamiento import cargar_red, predecir_red
from graficos import (
    figura_comparacion,
    figura_curvas,
    figura_importancia,
    figura_predicciones,
)


def leer(nombre):
    """Lee un JSON de la carpeta de resultados."""
    with open(os.path.join(DIR_RESULTADOS, nombre), encoding="utf-8") as f:
        return json.load(f)


def trio(m):
    """(AUC PR, precisión, recall) de un diccionario de métricas."""
    return (m["auc_pr"], m["precision"], m["recall"])


def num(valor, signo=False, decimales=1):
    """Número con coma decimal (28,4); con signo si es una diferencia."""
    valor = round(valor, decimales) + 0.0  # el + 0.0 evita que aparezca "-0,0"
    formato = f"+.{decimales}f" if signo and valor else f".{decimales}f"
    return format(valor, formato).replace(".", ",")


def miles(entero):
    """Entero con espacio como separador de miles (1 000 000)."""
    return format(entero, ",").replace(",", " ")


def fila(nombre, del_articulo, propio):
    """Una fila de tabla Markdown: artículo, replicación y diferencia."""
    celdas = [nombre]
    for a, p in zip(del_articulo, propio):
        celdas += [num(a), num(p), num(round(p, 1) - a, signo=True)]
    return "| " + " | ".join(celdas) + " |"


ENCABEZADO = (
    "| {} | AUC PR art. | AUC PR rép. | Δ | Prec. art. | Prec. rép. | Δ | Recall art. | Recall rép. | Δ |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)

# %% [markdown]
# ### 5.1 Tabla II — los cuatro modelos

# %%
red = leer("metricas_red_todas.json")
historial = leer("historial_red_todas.json")
r_persistencia = leer("metricas_persistencia.json")
r_logistica = leer("metricas_logistica.json")
r_bosque = leer("metricas_bosque.json")

propios = {
    "Red neuronal": trio(red["completo_64"]),
    "Random forest": trio(r_bosque["prueba"]),
    "Regresión logística": trio(r_logistica["prueba"]),
    "Persistencia": trio(r_persistencia["1x"]),
}

lineas = ["# Comparación de resultados: artículo vs. replicación", ""]
lineas += [
    "Todas las métricas están en porcentaje y se midieron en el conjunto de prueba,",
    "sobre la muestra completa de 64 x 64 km (1 689 muestras).",
    "art. = artículo; rép. = esta replicación; Δ = replicación menos artículo (en puntos).",
    "",
    "## Tabla II — Predicción de la propagación", "",
    ENCABEZADO.format("Modelo"),
]
for modelo, del_articulo in ARTICULO_TABLA_II.items():
    lineas.append(fila(modelo, del_articulo, propios[modelo]))

# Las mismas predicciones medidas sobre recortes aleatorios de 32 x 32 km (el
# tamaño de entrenamiento). El artículo no dice sobre qué píxeles midió cada
# modelo, así que se muestran los dos protocolos.
r = red["recortes_32"]
lineas += ["", "Los mismos modelos medidos sobre recortes aleatorios de 32 x 32 km:", "",
           ENCABEZADO.format("Modelo (recortes de 32 x 32 km)"),
           fila("Red neuronal", ARTICULO_TABLA_II["Red neuronal"], trio(r)),
           fila("Persistencia", ARTICULO_TABLA_II["Persistencia"], trio(r_persistencia["recortes_32"]))]
lineas += [
    "",
    f"- Red neuronal: {miles(historial['pasos_totales'])} pasos de entrenamiento "
    f"({historial['epocas']} épocas x {historial['pasos_por_epoca']} pasos); "
    "el artículo usó 1 000 000 de pasos en 4 GPU V100.",
    f"- Mejor época según validación: {historial['mejor_epoca']} "
    f"(AUC PR de validación {num(historial['mejor_val_auc_pr'])} %).",
    f"- Variación del AUC PR de la red entre tres juegos de recortes distintos: "
    f"±{num(r['auc_pr_desviacion'], decimales=2)} puntos.",
    f"- Regresión logística: W = {r_logistica['w_usado']} (el del artículo), {miles(r_logistica['pixeles_entrenamiento'])} píxeles "
    f"de entrenamiento. El mejor W en nuestra validación fue {r_logistica['mejor_w_validacion']}; con él, en prueba: "
    f"AUC PR {num(r_logistica['prueba_con_mejor_w']['auc_pr'])} %, precisión {num(r_logistica['prueba_con_mejor_w']['precision'])} %, "
    f"recall {num(r_logistica['prueba_con_mejor_w']['recall'])} %.",
    f"- Random forest: W = {r_bosque['w_usado']}; entrenado con {miles(r_bosque['pixeles_entrenamiento'])} píxeles.",
]

figura_comparacion(
    [("Red neuronal\n(64 x 64)", ARTICULO_TABLA_II["Red neuronal"], propios["Red neuronal"]),
     ("Red neuronal\n(recortes 32 x 32)", ARTICULO_TABLA_II["Red neuronal"], trio(r))]
    + [(m.replace(" ", "\n", 1), ARTICULO_TABLA_II[m], propios[m]) for m in ARTICULO_TABLA_II if m != "Red neuronal"]
)

# %% [markdown]
# ### 5.2 Tabla III — persistencia a resolución más gruesa

# %%
lineas += ["", "## Tabla III (filas de persistencia) — Resolución más gruesa", "",
           ENCABEZADO.format("Resolución")]
for factor, del_articulo in ARTICULO_TABLA_III_PERSISTENCIA.items():
    lineas.append(fila(f"{factor}x ({factor} km)", del_articulo, trio(r_persistencia[f"{factor}x"])))

# %% [markdown]
# ### 5.3 Ablación de la red (Tabla IV y Sección V-B)

# %%
experimentos = (
    ("Las 12 variables (Tabla II)", "metricas_red_todas.json", ARTICULO_TABLA_II["Red neuronal"]),
    ("Sin el fuego previo (Tabla IV)", "metricas_red_sin_fuego_previo.json", ARTICULO_TABLA_IV["PrevFireMask"]),
    ("Solo el fuego previo (Sección V-B)", "metricas_red_solo_fuego_previo.json", ARTICULO_SOLO_FUEGO_PREVIO),
)
lineas += ["", "## Ablación de la red neuronal"]
for protocolo, titulo in (("completo_64", "muestra completa de 64 x 64 km"),
                          ("recortes_32", "recortes de 32 x 32 km")):
    lineas += ["", f"Medido sobre {titulo}:", "", ENCABEZADO.format("Variables de entrada")]
    for nombre, archivo, del_articulo in experimentos:
        if os.path.exists(os.path.join(DIR_RESULTADOS, archivo)):
            lineas.append(fila(nombre, del_articulo, trio(leer(archivo)[protocolo])))

# Experimento adicional (no está en el artículo): tasa de aprendizaje 10 veces mayor.
if os.path.exists(os.path.join(DIR_RESULTADOS, "metricas_red_todas_tasa_0001.json")):
    extra = leer("metricas_red_todas_tasa_0001.json")
    h_extra = leer("historial_red_todas_tasa_0001.json")
    lineas += ["", f"Experimento adicional, fuera del artículo: las 12 variables con tasa de aprendizaje 0,001 "
               f"y {h_extra['epocas']} épocas ({miles(h_extra['pasos_totales'])} pasos): "
               f"AUC PR {num(extra['completo_64']['auc_pr'])} / precisión {num(extra['completo_64']['precision'])} / "
               f"recall {num(extra['completo_64']['recall'])} sobre 64 x 64 km; "
               f"{num(extra['recortes_32']['auc_pr'])} / {num(extra['recortes_32']['precision'])} / "
               f"{num(extra['recortes_32']['recall'])} sobre recortes de 32 x 32 km."]

# %% [markdown]
# ### 5.4 Ablación de la regresión logística (Tabla VIII)

# %%
ablacion = leer("ablacion_logistica.json")
lineas += ["", "## Tabla VIII — Regresión logística quitando una variable", "",
           ENCABEZADO.format("Variable quitada")]
for variable, del_articulo in ARTICULO_TABLA_VIII.items():
    lineas.append(fila(NOMBRES_ES[variable], del_articulo, trio(ablacion[variable])))

# %% [markdown]
# ### 5.5 Importancia de variables del random forest (Tabla VI)

# %%
importancia = leer("importancia_bosque.json")
lineas += ["", "## Tabla VI — Random forest: 20 columnas más importantes (Gini)", "",
           "| # | Artículo | Importancia | Replicación | Importancia |",
           "| ---: | --- | ---: | --- | ---: |"]
for i, ((var_a, pos_a, imp_a), propia) in enumerate(zip(ARTICULO_TABLA_VI, importancia[:20]), start=1):
    lineas.append(f"| {i} | {NOMBRES_ES[var_a]} ({pos_a}) | {num(imp_a, decimales=3)} | "
                  f"{NOMBRES_ES[propia['variable']]} ({propia['posicion']}) | {num(propia['importancia'], decimales=3)} |")

# Suma por variable (las 9 posiciones juntas) para verlo en un solo gráfico.
suma_propia = {v: 0.0 for v in VARIABLES_ENTRADA}
for item in importancia:
    suma_propia[item["variable"]] += item["importancia"]
# Del artículo solo se puede sumar lo que publica: su Tabla VI trae las 20
# columnas más importantes. El fuego previo y el ERC aparecen con sus 9
# posiciones; la elevación solo con 2, así que no se incluye en el gráfico.
suma_articulo = {}
for variable in VARIABLES_ENTRADA:
    valores = [valor for v, _, valor in ARTICULO_TABLA_VI if v == variable]
    if len(valores) == 9:
        suma_articulo[variable] = sum(valores)

lineas += ["", "Suma de la importancia por variable (replicación, 9 posiciones):", "",
           "| Variable | Importancia |", "| --- | ---: |"]
for variable in sorted(suma_propia, key=lambda v: -suma_propia[v]):
    lineas.append(f"| {NOMBRES_ES[variable]} | {num(suma_propia[variable], decimales=3)} |")
figura_importancia(suma_articulo, suma_propia)

# %% [markdown]
# ### 5.6 Coeficientes de la regresión logística (Tabla VII)

# %%
coeficientes = leer("coeficientes_logistica.json")
lineas += ["", "## Tabla VII — Regresión logística: 20 coeficientes más grandes (replicación)", "",
           "| # | Columna | Coeficiente |", "| ---: | --- | ---: |"]
for i, c in enumerate(coeficientes, start=1):
    lineas.append(f"| {i} | {NOMBRES_ES[c['variable']]} ({c['posicion']}) | {num(c['coeficiente'], signo=True, decimales=2)} |")

ruta = os.path.join(DIR_RESULTADOS, "comparacion_resultados.md")
with open(ruta, "w", encoding="utf-8") as f:
    f.write("\n".join(lineas) + "\n")
print("\n".join(lineas))

# %% [markdown]
# ### 5.7 Curvas de entrenamiento y ejemplos de predicción (Fig. 3 del artículo)

# %%
figura_curvas(historial["historial"], historial["mejor_epoca"])

modelo = cargar_red(VARIABLES_ENTRADA, "red_todas")
X_prueba, Y_prueba = cargar_numpy("test")
X_rec, Y_rec = recortes_fijos(X_prueba, Y_prueba)
probabilidad = predecir_red(modelo, X_rec)

# Para elegir ejemplos se calcula el F1 de cada muestra (solo las que tienen
# al menos 20 km² de fuego real) y se muestran las 5 mejores y las 5 peores.
marcado = probabilidad[..., 0] > 0.5
real = Y_rec[..., 0] == 1
aciertos = (marcado & real).sum(axis=(1, 2))
f1 = 2 * aciertos / np.maximum(marcado.sum(axis=(1, 2)) + real.sum(axis=(1, 2)), 1)
candidatas = np.nonzero(real.sum(axis=(1, 2)) >= 20)[0]
orden = candidatas[np.argsort(-f1[candidatas])]
# (Se dibujan recortes de 32 x 32 km, como las figuras del artículo.)
figura_predicciones(X_rec, Y_rec, probabilidad, list(orden[:5]) + list(orden[-5:]),
                    titulos=["mejor"] * 5 + ["peor"] * 5)
print("Columnas 1-5: mejores predicciones (F1 más alto). Columnas 6-10: peores.")
