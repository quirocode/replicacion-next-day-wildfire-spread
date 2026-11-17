"""
construir_notebook.py — Arma el notebook a partir de los archivos .py.

El código "de verdad" vive en src/*.py y en los paso*.py. Este script lo copia
dentro de un único notebook autocontenido para Kaggle o Colab, de modo que no
haya dos versiones distintas del mismo código.

Reglas de conversión:
  - Cada módulo de src/ se vuelve una celda de código.
  - En los paso*.py, las marcas "# %%" separan celdas y "# %% [markdown]"
    marca celdas de texto.
  - Se quitan los `from <módulo local> import ...` y el `sys.path.insert(...)`,
    porque en el notebook todo queda definido en el mismo espacio de nombres.

Uso:
    python herramientas/construir_notebook.py
"""

import ast
import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULOS = ["config", "datos", "modelo", "metricas", "entrenamiento", "modelos_base", "graficos"]
PASOS = [
    "paso1_explorar_datos.py",
    "paso2_entrenar_red.py",
    "paso3_modelos_base.py",
    "paso4_ablacion_red.py",
    "paso5_comparar_con_articulo.py",
]
TITULOS_MODULOS = {
    "config": "### Configuración (`src/config.py`)\nNombres de variables, estadísticas, hiperparámetros, presupuesto de cómputo y los resultados del artículo.",
    "datos": "### Datos (`src/datos.py`)\nLectura de los TFRecord, aumento de datos, recorte y normalización.",
    "modelo": "### Modelo (`src/modelo.py`)\nAutoencoder convolucional de la Fig. 6 del artículo.",
    "metricas": "### Pérdida y métricas (`src/metricas.py`)\nEntropía cruzada ponderada y AUC PR / precisión / recall, ignorando píxeles inciertos.",
    "entrenamiento": "### Entrenamiento y evaluación (`src/entrenamiento.py`)",
    "modelos_base": "### Modelos base (`src/modelos_base.py`)\nPersistencia, regresión logística y random forest.",
    "graficos": "### Gráficos (`src/graficos.py`)",
}

INTRODUCCION = """# Replicación de *Next Day Wildfire Spread* (Huot et al., 2022)

**Artículo:** Huot, F., Hu, R. L., Goyal, N., Sankar, T., Ihme, M., & Chen, Y.-F. (2022). Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading From Remote-Sensing Data. *IEEE Transactions on Geoscience and Remote Sensing, 60*, Art. 4412513. https://doi.org/10.1109/TGRS.2022.3192974

**Qué hace este notebook:** repite el flujo de trabajo del artículo con el dataset que publicaron los autores en Kaggle y compara los resultados con los que reportan.

| Paso | Qué se hace | Sección del artículo |
| --- | --- | --- |
| 0 | Preparar el entorno y definir el código | — |
| 1 | Explorar el dataset | II |
| 2 | Entrenar y evaluar la red neuronal | III, IV y V-A |
| 3 | Modelos base: persistencia, regresión logística, random forest | IV-A, Apéndice A |
| 4 | Ablación de la red (¿qué variables necesita?) | V-B |
| 5 | Comparar con el artículo | Tablas II a VIII |

**Dónde correrlo**

- **Kaggle (recomendado):** *Add input* → buscar `next-day-wildfire-spread` (fantineh) → activar *Accelerator: GPU*.
- **Google Colab:** *Entorno de ejecución → Cambiar tipo → GPU*. El dataset se descarga solo (2 GB).

Sin GPU, el notebook completo tarda entre 3 y 4 horas (cada una de las tres redes tomó cerca de 1 hora en CPU). Con GPU es mucho más rápido; no se midió.
"""

PREPARACION = """# El dataset de Kaggle es público: kagglehub lo descarga sin usuario ni clave.
# En Kaggle no hace falta descargar nada si se agregó el dataset como "input".
%pip install -q kagglehub

import os

# Presupuesto de cómputo: "prueba" (segundos), "prototipo" (el de la
# replicación) o "largo" (más épocas, para acercarse al artículo).
os.environ.setdefault("WILDFIRE_PRESUPUESTO", "prototipo")

# En un notebook no existe __file__; se define para que el código que arma las
# rutas del proyecto funcione igual que en los archivos .py.
os.makedirs("src", exist_ok=True)
__file__ = os.path.abspath("src/notebook.py")

import tensorflow as tf
print("TensorFlow", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU") or "no hay (se usará CPU)")
"""


def quitar_importaciones_locales(codigo):
    """Elimina las importaciones entre archivos del proyecto y el sys.path."""
    arbol = ast.parse(codigo)
    lineas = codigo.split("\n")
    borrar = set()
    for nodo in arbol.body:
        es_local = isinstance(nodo, ast.ImportFrom) and nodo.module in MODULOS
        es_ruta = (
            isinstance(nodo, ast.Expr)
            and isinstance(nodo.value, ast.Call)
            and "sys.path.insert" in ast.unparse(nodo.value.func)
        )
        if es_local or es_ruta:
            borrar.update(range(nodo.lineno - 1, nodo.end_lineno))
    resto = [l for i, l in enumerate(lineas) if i not in borrar]
    texto = "\n".join(resto)
    while "\n\n\n\n" in texto:  # compactar huecos que dejan las líneas borradas
        texto = texto.replace("\n\n\n\n", "\n\n\n")
    return texto.strip() + "\n"


def celdas_de_paso(ruta):
    """Convierte un paso*.py con marcas '# %%' en celdas de notebook."""
    with open(ruta, encoding="utf-8") as f:
        bloques = f.read().split("# %%")
    celdas = []
    for bloque in bloques:
        if not bloque.strip():
            continue
        if bloque.startswith(" [markdown]"):
            lineas = bloque[len(" [markdown]"):].strip("\n").split("\n")
            texto = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in lineas)
            celdas.append(new_markdown_cell(texto))
        else:
            codigo = quitar_importaciones_locales(bloque.strip("\n"))
            if codigo.strip():
                celdas.append(new_code_cell(codigo))
    return celdas


def construir():
    celdas = [new_markdown_cell(INTRODUCCION)]
    celdas.append(new_markdown_cell("## Paso 0 — Preparar el entorno y definir el código"))
    celdas.append(new_code_cell(PREPARACION))
    for modulo in MODULOS:
        with open(os.path.join(RAIZ, "src", f"{modulo}.py"), encoding="utf-8") as f:
            codigo = quitar_importaciones_locales(f.read())
        celdas.append(new_markdown_cell(TITULOS_MODULOS[modulo]))
        celdas.append(new_code_cell(codigo))
    for paso in PASOS:
        celdas.extend(celdas_de_paso(os.path.join(RAIZ, paso)))

    cuaderno = new_notebook(cells=celdas)
    cuaderno.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    cuaderno.metadata["language_info"] = {"name": "python"}
    salida = os.path.join(RAIZ, "replicacion_next_day_wildfire_spread.ipynb")
    nbformat.write(cuaderno, salida)
    print(f"Notebook escrito en {salida} ({len(celdas)} celdas)")


if __name__ == "__main__":
    construir()
