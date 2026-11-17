"""
graficos.py — Figuras de la replicación (Matplotlib).

Todas las funciones guardan un PNG en resultados/ y devuelven su ruta.
Las máscaras de fuego usan la convención de colores del artículo:
rojo = fuego, gris = sin fuego, negro = dato incierto.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator

from config import DIR_RESULTADOS, NOMBRES_ES, VARIABLES_ENTRADA

# Colores: una paleta de dos series (artículo / replicación) y tinta neutra.
COLOR_ARTICULO = "#2a78d6"     # azul
COLOR_REPLICA = "#eb6834"      # naranja
COLOR_FONDO = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SUAVE = "#52514e"
REJILLA = "#e4e3df"

# Máscaras: -1 incierto (negro), 0 sin fuego (gris), 1 fuego (rojo).
MAPA_FUEGO = ListedColormap(["#1a1a19", "#c9c8c3", "#e8421c"])
NORMA_FUEGO = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], MAPA_FUEGO.N)

plt.rcParams.update({
    "figure.facecolor": COLOR_FONDO,
    "axes.facecolor": COLOR_FONDO,
    "savefig.facecolor": COLOR_FONDO,
    "font.size": 10,
    "text.color": TINTA,
    "axes.labelcolor": TINTA_SUAVE,
    "xtick.color": TINTA_SUAVE,
    "ytick.color": TINTA_SUAVE,
    "axes.edgecolor": REJILLA,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _con_coma(valor):
    """Número con un decimal y coma decimal (28,4), como se escribe en español."""
    return f"{valor:.1f}".replace(".", ",")


def _guardar(figura, nombre):
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    ruta = os.path.join(DIR_RESULTADOS, nombre)
    figura.savefig(ruta, dpi=160, bbox_inches="tight")
    return ruta


def _mascara(eje, imagen, titulo=None):
    """Dibuja una máscara de fuego sin ejes."""
    eje.imshow(imagen, cmap=MAPA_FUEGO, norm=NORMA_FUEGO, interpolation="nearest")
    eje.set_xticks([])
    eje.set_yticks([])
    for borde in eje.spines.values():
        borde.set_visible(False)
    if titulo:
        eje.set_title(titulo, fontsize=9, color=TINTA)


def figura_ejemplos(X_crudo, Y, indices, nombre="fig1_ejemplos_dataset.png"):
    """Equivalente a la Fig. 1 del artículo: una fila por muestra.

    Columnas: las 11 variables (en sus unidades originales), el fuego de hoy y
    el fuego de mañana.
    """
    variables = [v for v in VARIABLES_ENTRADA if v != "PrevFireMask"]
    columnas = len(variables) + 2
    figura, ejes = plt.subplots(len(indices), columnas, figsize=(1.45 * columnas, 1.75 * len(indices)))
    for fila, i in enumerate(indices):
        for col, variable in enumerate(variables):
            eje = ejes[fila, col]
            imagen = eje.imshow(X_crudo[i, ..., VARIABLES_ENTRADA.index(variable)], cmap="viridis")
            eje.set_xticks([])
            eje.set_yticks([])
            barra = figura.colorbar(imagen, ax=eje, orientation="horizontal", pad=0.04, fraction=0.08)
            barra.ax.tick_params(labelsize=6)
            barra.ax.locator_params(nbins=2)
            if fila == 0:
                eje.set_title(NOMBRES_ES[variable].replace(" ", "\n", 1), fontsize=8, color=TINTA)
        hoy = X_crudo[i, ..., VARIABLES_ENTRADA.index("PrevFireMask")]
        _mascara(ejes[fila, -2], hoy, "Fuego hoy\n(día t)" if fila == 0 else None)
        _mascara(ejes[fila, -1], Y[i, ..., 0], "Fuego mañana\n(día t+1)" if fila == 0 else None)
    figura.suptitle("Muestras del dataset (64 x 64 km, 1 km por píxel). "
                    "Rojo: fuego · gris: sin fuego · negro: incierto", fontsize=10, y=1.0)
    figura.tight_layout()
    return _guardar(figura, nombre)


def figura_curvas(historial, mejor_epoca, nombre="fig2_curvas_entrenamiento.png"):
    """Pérdida y AUC PR por época, en entrenamiento y validación."""
    epocas = np.arange(1, len(historial["loss"]) + 1)
    figura, (eje1, eje2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for eje, clave, titulo in ((eje1, "loss", "Pérdida (entropía cruzada ponderada)"),
                               (eje2, "auc_pr", "AUC PR (%)")):
        escala = 100.0 if clave == "auc_pr" else 1.0
        eje.plot(epocas, escala * np.array(historial[clave]), color=COLOR_ARTICULO, lw=2, label="Entrenamiento")
        eje.plot(epocas, escala * np.array(historial["val_" + clave]), color=COLOR_REPLICA, lw=2, label="Validación")
        eje.axvline(mejor_epoca, color=TINTA_SUAVE, lw=1, ls=":")
        eje.set_title(titulo, loc="left", fontsize=10)
        eje.set_xlabel("Época")
        eje.xaxis.set_major_locator(MaxNLocator(integer=True))
        eje.grid(axis="y", color=REJILLA, lw=0.8)
        eje.legend(frameon=False)
    eje2.annotate(f"mejor época: {mejor_epoca}", xy=(mejor_epoca, eje2.get_ylim()[0]),
                  xytext=(4, 6), textcoords="offset points", fontsize=8, color=TINTA_SUAVE)
    figura.tight_layout()
    return _guardar(figura, nombre)


def figura_predicciones(X, Y, probabilidad, indices, titulos=None, nombre="fig3_predicciones.png"):
    """Equivalente a la Fig. 3 del artículo: hoy, mañana real y mañana predicho.

    Args:
        indices: qué muestras dibujar (una por columna).
        titulos: texto opcional encima de cada columna.
    """
    figura, ejes = plt.subplots(3, len(indices), figsize=(1.5 * len(indices), 4.9))
    rotulos = ("Fuego hoy\n(entrada)", "Fuego mañana\n(real)", "Fuego mañana\n(predicho)")
    canal = VARIABLES_ENTRADA.index("PrevFireMask")
    for col, i in enumerate(indices):
        _mascara(ejes[0, col], X[i, ..., canal], titulos[col] if titulos else None)
        _mascara(ejes[1, col], Y[i, ..., 0])
        _mascara(ejes[2, col], (probabilidad[i, ..., 0] > 0.5).astype(float))
    for fila, rotulo in enumerate(rotulos):
        ejes[fila, 0].set_ylabel(rotulo, fontsize=9, color=TINTA, rotation=0, ha="right", va="center")
    figura.suptitle("Predicciones de la red en el conjunto de prueba (recortes de 32 x 32 km)", fontsize=10)
    figura.tight_layout()
    return _guardar(figura, nombre)


def figura_comparacion(filas, nombre="fig4_comparacion_tabla2.png"):
    """Barras agrupadas: artículo vs. replicación para cada modelo y métrica.

    Args:
        filas: lista de (nombre del modelo, (auc, prec, rec) artículo,
            (auc, prec, rec) replicación). El nombre puede traer saltos de línea.
    """
    metricas = ("AUC PR (%)", "Precisión (%)", "Recall (%)")
    figura, ejes = plt.subplots(1, 3, figsize=(14, 4.0), sharey=False)
    posiciones = np.arange(len(filas))
    ancho = 0.36
    for k, (eje, titulo) in enumerate(zip(ejes, metricas)):
        del_articulo = [f[1][k] for f in filas]
        propios = [f[2][k] for f in filas]
        b1 = eje.bar(posiciones - ancho / 2 - 0.01, del_articulo, ancho, color=COLOR_ARTICULO, label="Artículo")
        b2 = eje.bar(posiciones + ancho / 2 + 0.01, propios, ancho, color=COLOR_REPLICA, label="Replicación")
        eje.bar_label(b1, fmt=_con_coma, fontsize=8, color=TINTA, padding=2)
        eje.bar_label(b2, fmt=_con_coma, fontsize=8, color=TINTA, padding=2)
        eje.set_xticks(posiciones)
        eje.set_xticklabels([f[0] for f in filas], fontsize=8, color=TINTA)
        eje.set_title(titulo, loc="left", fontsize=10)
        eje.grid(axis="y", color=REJILLA, lw=0.8)
        eje.set_axisbelow(True)
        eje.set_ylim(0, max(del_articulo + propios) * 1.18)
    ejes[0].legend(frameon=False, loc="upper right")
    figura.suptitle("Tabla II del artículo frente a esta replicación (conjunto de prueba)", fontsize=11, x=0.01, ha="left")
    figura.tight_layout()
    return _guardar(figura, nombre)


def figura_importancia(del_articulo, propias, nombre="fig5_importancia_variables.png"):
    """Importancia de Gini del random forest sumada por variable.

    Args:
        del_articulo: dict variable -> importancia sumada en las 9 posiciones.
            Solo trae las variables que el artículo publica completas.
        propias: dict variable -> importancia sumada en las 9 posiciones.
    """
    orden = sorted(propias, key=lambda v: propias[v])
    y = np.arange(len(orden))
    alto = 0.36
    figura, eje = plt.subplots(figsize=(8, 5))
    eje.barh(y + alto / 2 + 0.01, [del_articulo.get(v, 0.0) for v in orden], alto,
             color=COLOR_ARTICULO, label="Artículo (solo publica completas estas dos variables)")
    eje.barh(y - alto / 2 - 0.01, [propias[v] for v in orden], alto,
             color=COLOR_REPLICA, label="Replicación")
    eje.set_yticks(y)
    eje.set_yticklabels([NOMBRES_ES[v] for v in orden], color=TINTA)
    eje.set_xlabel("Importancia de Gini (suma de las 9 posiciones de la vecindad)")
    eje.set_title("Random forest: ¿qué variables pesan más?", loc="left", fontsize=11)
    eje.grid(axis="x", color=REJILLA, lw=0.8)
    eje.set_axisbelow(True)
    eje.legend(frameon=False, loc="lower right")
    figura.tight_layout()
    return _guardar(figura, nombre)
