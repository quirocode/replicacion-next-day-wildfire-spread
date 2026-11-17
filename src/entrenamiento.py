"""
entrenamiento.py — Entrenar y evaluar la red neuronal.

Reproduce la Sección IV-B del artículo ("Training Details"):
    - Optimizador Adam con tasa de aprendizaje 0,0001.
    - Pérdida: entropía cruzada ponderada (peso 3 para "fuego").
    - Aumento de datos: recorte, volteo y rotación al azar.
    - Se conserva el modelo con mejor AUC PR en el conjunto de validación.

OJO: los autores publicaron el modelo, la pérdida y las métricas, pero NO el
script de entrenamiento. Este archivo lo reconstruye a partir del texto.
"""

import csv
import json
import os
import time

import keras
import numpy as np

from config import (
    DIR_MODELOS,
    DIR_RESULTADOS,
    PRESUPUESTO,
    PRESUPUESTOS,
    REPETICIONES_PRUEBA,
    SEMILLA,
    TASA_APRENDIZAJE,
    VARIABLES_ENTRADA,
)
from datos import cargar_numpy, dataset_entrenamiento, recortes_fijos
from metricas import (
    calcular_metricas,
    metricas_de_entrenamiento,
    perdida_ponderada_con_mascara,
)
from modelo import crear_modelo


def _canales(variables):
    """Posición de cada variable pedida dentro de los 12 canales."""
    return [VARIABLES_ENTRADA.index(v) for v in variables]


def sigmoide(logits):
    """Convierte logits en probabilidades entre 0 y 1."""
    return 1.0 / (1.0 + np.exp(-logits))


def guardar_json(datos, nombre_archivo):
    """Guarda un diccionario en resultados/<nombre_archivo>."""
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    ruta = os.path.join(DIR_RESULTADOS, nombre_archivo)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)
    return ruta


def entrenar_red(variables=VARIABLES_ENTRADA, nombre="red_todas", epocas=None,
                 pasos_por_epoca=None, tasa_aprendizaje=TASA_APRENDIZAJE, verbose=2):
    """Entrena el autoencoder con las variables indicadas.

    Args:
        variables: qué canales de entrada usar (los 12, o menos en la ablación).
        nombre: prefijo de los archivos que se guardan.
        epocas, pasos_por_epoca: si no se dan, salen del presupuesto elegido en
            config.py.
        tasa_aprendizaje: por defecto, la del artículo (0,0001).

    Returns:
        (modelo con los pesos de su MEJOR época, historial de métricas).
    """
    presupuesto = PRESUPUESTOS[PRESUPUESTO]
    epocas = epocas or presupuesto["epocas"]
    pasos_por_epoca = pasos_por_epoca or presupuesto["pasos_por_epoca"]

    # Misma semilla -> mismos pesos iniciales y mismo orden de los datos.
    keras.utils.set_random_seed(SEMILLA)

    modelo = crear_modelo(num_canales=len(variables))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tasa_aprendizaje),
        loss=perdida_ponderada_con_mascara(),
        metrics=metricas_de_entrenamiento(),
    )

    # Validación: un recorte fijo de 32 x 32 por cada muestra de 'eval'.
    X_val, Y_val = cargar_numpy("eval")
    X_val, Y_val = recortes_fijos(X_val[..., _canales(variables)], Y_val)

    # Tres ayudantes que Keras llama al final de cada época:
    os.makedirs(DIR_MODELOS, exist_ok=True)
    ruta_pesos = os.path.join(DIR_MODELOS, f"{nombre}.weights.h5")
    ruta_registro = os.path.join(DIR_MODELOS, f"{nombre}_epocas.csv")
    mejor_hasta_ahora = max(
        (fila["val_auc_pr"] for fila in _leer_registro(ruta_registro)), default=None
    )
    ayudantes = [
        # 1. Si la sesión se corta (pasa en Colab y Kaggle), al volver a llamar
        #    a esta función el entrenamiento sigue desde la última época
        #    terminada en vez de empezar de cero.
        keras.callbacks.BackupAndRestore(os.path.join(DIR_MODELOS, f"{nombre}_respaldo")),
        # 2. Igual que el BestModelExporter de los autores: solo se guardan los
        #    pesos cuando mejora el AUC PR de validación.
        keras.callbacks.ModelCheckpoint(
            ruta_pesos, monitor="val_auc_pr", mode="max", save_best_only=True,
            save_weights_only=True, initial_value_threshold=mejor_hasta_ahora,
        ),
        # 3. Anota las métricas de cada época en un CSV (sobrevive a un corte).
        keras.callbacks.CSVLogger(ruta_registro, append=True),
    ]

    inicio = time.time()
    modelo.fit(
        dataset_entrenamiento(variables),
        steps_per_epoch=pasos_por_epoca,
        epochs=epocas,
        validation_data=(X_val, Y_val),
        callbacks=ayudantes,
        verbose=verbose,
    )
    minutos = (time.time() - inicio) / 60.0

    # El historial completo sale del CSV (incluye las épocas previas a un corte).
    filas = _leer_registro(ruta_registro)
    historial = {clave: [fila[clave] for fila in filas] for clave in filas[0] if clave != "epoch"}

    # Se recupera la mejor época (no necesariamente la última).
    modelo.load_weights(ruta_pesos)
    mejor_epoca = int(np.argmax(historial["val_auc_pr"])) + 1

    guardar_json(
        {
            "nombre": nombre,
            "variables": list(variables),
            "epocas": epocas,
            "pasos_por_epoca": pasos_por_epoca,
            "pasos_totales": epocas * pasos_por_epoca,
            "tasa_aprendizaje": tasa_aprendizaje,
            "minutos_ultimo_tramo": round(minutos, 1),
            "mejor_epoca": mejor_epoca,
            "mejor_val_auc_pr": 100.0 * float(max(historial["val_auc_pr"])),
            "parametros": int(modelo.count_params()),
            "historial": historial,
        },
        f"historial_{nombre}.json",
    )
    return modelo, historial


def _leer_registro(ruta):
    """Lee el CSV de épocas como lista de diccionarios de números."""
    if not os.path.exists(ruta):
        return []
    with open(ruta, newline="") as f:
        return [{k: float(v) for k, v in fila.items()} for fila in csv.DictReader(f)]


def cargar_red(variables=VARIABLES_ENTRADA, nombre="red_todas"):
    """Reconstruye una red ya entrenada a partir de sus pesos guardados."""
    modelo = crear_modelo(num_canales=len(variables))
    modelo.load_weights(os.path.join(DIR_MODELOS, f"{nombre}.weights.h5"))
    return modelo


def predecir_red(modelo, X, variables=VARIABLES_ENTRADA):
    """Probabilidad de fuego por píxel para un arreglo de muestras."""
    logits = modelo.predict(X[..., _canales(variables)], batch_size=256, verbose=0)
    return sigmoide(logits)


def evaluar_red(modelo, X, Y, variables=VARIABLES_ENTRADA,
                repeticiones=REPETICIONES_PRUEBA):
    """Mide la red sobre un conjunto (normalmente el de prueba).

    Se reportan dos protocolos:
      - "completo_64": la muestra entera de 64 x 64 km. Es el que se compara
        con la Tabla II: con él, la persistencia reproduce exactamente los
        números del artículo (paso 3). La red puede recibir 64 x 64 aunque se
        entrenó con 32 x 32 porque solo tiene capas convolucionales.
      - "recortes_32": recortes aleatorios fijos de 32 x 32 km, el tamaño con
        el que se entrena. Sirve para ver cuánto cambia el número según los
        píxeles sobre los que se mide.
    """
    X_rec, Y_rec = recortes_fijos(X, Y, repeticiones=repeticiones)
    prob_rec = predecir_red(modelo, X_rec, variables)
    en_recortes = calcular_metricas(Y_rec, prob_rec)

    # ¿Cuánto cambia el AUC PR solo por el azar del recorte? (el artículo
    # reporta una variación de 0,5 puntos entre experimentos, Sección V-B).
    n = len(X)
    aucs = [
        calcular_metricas(Y_rec[k * n:(k + 1) * n], prob_rec[k * n:(k + 1) * n])["auc_pr"]
        for k in range(repeticiones)
    ]
    en_recortes["auc_pr_por_repeticion"] = aucs
    en_recortes["auc_pr_desviacion"] = float(np.std(aucs))

    en_completo = calcular_metricas(Y, predecir_red(modelo, X, variables))
    return {"recortes_32": en_recortes, "completo_64": en_completo}
