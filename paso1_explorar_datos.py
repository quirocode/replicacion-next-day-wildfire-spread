# %% [markdown]
# ## Paso 1 — Explorar el dataset (Sección II del artículo)
#
# Antes de entrenar nada se comprueba que los datos descargados de Kaggle son
# los que describe el artículo:
#
# | Dato | Artículo |
# | --- | --- |
# | Muestras | 18 545 |
# | El fuego crece de t a t+1 | 58 % (10 798 muestras) |
# | El fuego decrece | 39 % (7 191 muestras) |
# | Tamaño de cada muestra | 64 x 64 km, 12 capas de entrada + 1 de salida |

# %%
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from config import ARTICULO_DATASET, ESTADISTICAS, NOMBRES_ES, VARIABLES_ENTRADA
from datos import leer_crudo
from entrenamiento import guardar_json
from graficos import figura_ejemplos

CANAL_FUEGO = VARIABLES_ENTRADA.index("PrevFireMask")

# %% [markdown]
# ### 1.1 Recorrer las tres particiones
# Se leen los datos SIN normalizar y se acumulan conteos y estadísticas.

# %%
resumen = {}
conteo_etiquetas = {-1: 0, 0: 0, 1: 0}
crece = decrece = igual = centro_con_fuego = sin_fuego_previo = 0
suma = np.zeros(len(VARIABLES_ENTRADA))
suma_cuadrados = np.zeros(len(VARIABLES_ENTRADA))
pixeles_train = 0
minimos = np.full(len(VARIABLES_ENTRADA), np.inf)
maximos = np.full(len(VARIABLES_ENTRADA), -np.inf)
nan_total = 0

for particion in ("train", "eval", "test"):
    muestras = 0
    for x, y in leer_crudo(particion).batch(512):
        x, y = x.numpy(), y.numpy()
        muestras += len(x)
        nan_total += int(np.isnan(x).sum() + np.isnan(y).sum())

        # ¿Cuántos píxeles con fuego hay hoy y mañana en cada muestra?
        fuego_hoy = (x[..., CANAL_FUEGO] == 1).sum(axis=(1, 2))
        fuego_manana = (y[..., 0] == 1).sum(axis=(1, 2))
        crece += int((fuego_manana > fuego_hoy).sum())
        decrece += int((fuego_manana < fuego_hoy).sum())
        igual += int((fuego_manana == fuego_hoy).sum())
        sin_fuego_previo += int((fuego_hoy == 0).sum())

        # ¿El píxel central de la muestra tiene fuego mañana? (ver nota abajo)
        centro_con_fuego += int((y[:, 32, 32, 0] == 1).sum())

        for valor in conteo_etiquetas:
            conteo_etiquetas[valor] += int((y == valor).sum())

        minimos = np.minimum(minimos, x.min(axis=(0, 1, 2)))
        maximos = np.maximum(maximos, x.max(axis=(0, 1, 2)))

        if particion == "train":
            # Media y desviación DESPUÉS de recortar, como hicieron los autores.
            limites = np.array([ESTADISTICAS[v][:2] for v in VARIABLES_ENTRADA])
            recortado = np.clip(x, limites[:, 0], limites[:, 1]).astype(np.float64)
            suma += recortado.sum(axis=(0, 1, 2))
            suma_cuadrados += (recortado ** 2).sum(axis=(0, 1, 2))
            pixeles_train += recortado.shape[0] * recortado.shape[1] * recortado.shape[2]
    resumen[particion] = muestras

total = sum(resumen.values())
media = suma / pixeles_train
desviacion = np.sqrt(suma_cuadrados / pixeles_train - media ** 2)

# %% [markdown]
# ### 1.2 Comparar con lo que dice el artículo

# %%
print(f"Muestras: train {resumen['train']:,} | eval {resumen['eval']:,} | test {resumen['test']:,} "
      f"| total {total:,} (artículo: {ARTICULO_DATASET['muestras']:,})")
print(f"Reparto: {resumen['train'] / total:.1%} / {resumen['eval'] / total:.1%} / {resumen['test'] / total:.1%} (artículo: 8:1:1)")
print(f"El fuego crece:   {crece:,} ({crece / total:.1%})   artículo: {ARTICULO_DATASET['fuego_crece']:,} (58 %)")
print(f"El fuego decrece: {decrece:,} ({decrece / total:.1%})   artículo: {ARTICULO_DATASET['fuego_decrece']:,} (39 %)")
print(f"Queda igual:      {igual:,} ({igual / total:.1%})")
print(f"Muestras sin ningún píxel de fuego hoy: {sin_fuego_previo:,}")
print(f"Valores NaN en todo el dataset: {nan_total:,}")

pixeles = sum(conteo_etiquetas.values())
print("\nEtiquetas (fuego de mañana), sobre todos los píxeles:")
for valor, texto in ((1, "fuego"), (0, "sin fuego"), (-1, "incierto")):
    print(f"  {texto:10s} {conteo_etiquetas[valor] / pixeles:7.2%}")
print(f"\nMuestras cuyo píxel central (32, 32) tiene fuego mañana: {centro_con_fuego / total:.1%}")

print("\nMedia y desviación en 'train' después de recortar (calculado aquí | código de los autores):")
for i, variable in enumerate(VARIABLES_ENTRADA[:-1]):
    _, _, media_autores, desv_autores = ESTADISTICAS[variable]
    print(f"  {NOMBRES_ES[variable]:22s} media {media[i]:10.4f} | {media_autores:<10} desv {desviacion[i]:10.4f} | {desv_autores}")

guardar_json(
    {
        "muestras": resumen,
        "total": total,
        "fuego_crece": crece,
        "fuego_decrece": decrece,
        "fuego_igual": igual,
        "muestras_sin_fuego_previo": sin_fuego_previo,
        "nan": nan_total,
        "fraccion_etiquetas": {str(k): v / pixeles for k, v in conteo_etiquetas.items()},
        "fraccion_centro_con_fuego": centro_con_fuego / total,
        "estadisticas_train": {
            v: {"min": float(minimos[i]), "max": float(maximos[i]),
                "media_recortada": float(media[i]), "desv_recortada": float(desviacion[i])}
            for i, v in enumerate(VARIABLES_ENTRADA)
        },
    },
    "paso1_exploracion.json",
)

# %% [markdown]
# ### 1.3 Ver algunas muestras (equivalente a la Fig. 1 del artículo)

# %%
x, y = next(iter(leer_crudo("test").batch(64)))
figura_ejemplos(x.numpy(), y.numpy(), indices=[3, 12, 27, 41])
