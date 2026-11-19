# Comparación de resultados: artículo vs. replicación

Todas las métricas están en porcentaje y se midieron en el conjunto de prueba,
sobre la muestra completa de 64 x 64 km (1 689 muestras).
art. = artículo; rép. = esta replicación; Δ = replicación menos artículo (en puntos).

## Tabla II — Predicción de la propagación

| Modelo | AUC PR art. | AUC PR rép. | Δ | Prec. art. | Prec. rép. | Δ | Recall art. | Recall rép. | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Red neuronal | 28,4 | 24,9 | -3,5 | 33,6 | 29,9 | -3,7 | 43,1 | 42,3 | -0,8 |
| Random forest | 22,5 | 21,4 | -1,1 | 26,3 | 27,1 | +0,8 | 46,9 | 45,9 | -1,0 |
| Regresión logística | 19,8 | 20,0 | +0,2 | 32,5 | 36,4 | +3,9 | 35,3 | 28,4 | -6,9 |
| Persistencia | 11,5 | 11,5 | 0,0 | 35,7 | 35,7 | 0,0 | 27,3 | 27,3 | 0,0 |

Los mismos modelos medidos sobre recortes aleatorios de 32 x 32 km:

| Modelo (recortes de 32 x 32 km) | AUC PR art. | AUC PR rép. | Δ | Prec. art. | Prec. rép. | Δ | Recall art. | Recall rép. | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Red neuronal | 28,4 | 28,1 | -0,3 | 33,6 | 33,7 | +0,1 | 43,1 | 41,9 | -1,2 |
| Persistencia | 11,5 | 14,3 | +2,8 | 35,7 | 40,1 | +4,4 | 27,3 | 26,4 | -0,9 |

- Red neuronal: 28 080 pasos de entrenamiento (60 épocas x 468 pasos); el artículo usó 1 000 000 de pasos en 4 GPU V100.
- Mejor época según validación: 49 (AUC PR de validación 21,5 %).
- Variación del AUC PR de la red entre tres juegos de recortes distintos: ±0,02 puntos.
- Regresión logística: W = 7 (el del artículo), 1 000 000 píxeles de entrenamiento. El mejor W en nuestra validación fue 1; con él, en prueba: AUC PR 19,7 %, precisión 21,2 %, recall 45,6 %.
- Random forest: W = 5; entrenado con 1 000 000 píxeles.

## Tabla III (filas de persistencia) — Resolución más gruesa

| Resolución | AUC PR art. | AUC PR rép. | Δ | Prec. art. | Prec. rép. | Δ | Recall art. | Recall rép. | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2x (2 km) | 19,5 | 19,5 | 0,0 | 45,6 | 45,6 | 0,0 | 36,8 | 36,8 | 0,0 |
| 4x (4 km) | 30,6 | 30,6 | 0,0 | 56,0 | 56,0 | 0,0 | 46,8 | 46,8 | 0,0 |
| 8x (8 km) | 41,3 | 41,3 | 0,0 | 64,0 | 64,0 | 0,0 | 52,5 | 52,5 | 0,0 |

## Ablación de la red neuronal

Medido sobre muestra completa de 64 x 64 km:

| Variables de entrada | AUC PR art. | AUC PR rép. | Δ | Prec. art. | Prec. rép. | Δ | Recall art. | Recall rép. | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Las 12 variables (Tabla II) | 28,4 | 24,9 | -3,5 | 33,6 | 29,9 | -3,7 | 43,1 | 42,3 | -0,8 |
| Sin el fuego previo (Tabla IV) | 6,8 | 3,8 | -3,0 | 4,6 | 15,6 | +11,0 | 0,2 | 0,8 | +0,6 |
| Solo el fuego previo (Sección V-B) | 22,9 | 18,0 | -4,9 | 29,2 | 24,9 | -4,3 | 49,6 | 48,5 | -1,1 |

Medido sobre recortes de 32 x 32 km:

| Variables de entrada | AUC PR art. | AUC PR rép. | Δ | Prec. art. | Prec. rép. | Δ | Recall art. | Recall rép. | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Las 12 variables (Tabla II) | 28,4 | 28,1 | -0,3 | 33,6 | 33,7 | +0,1 | 43,1 | 41,9 | -1,2 |
| Sin el fuego previo (Tabla IV) | 6,8 | 6,9 | +0,1 | 4,6 | 15,3 | +10,7 | 0,2 | 0,3 | +0,1 |
| Solo el fuego previo (Sección V-B) | 22,9 | 21,3 | -1,6 | 29,2 | 29,6 | +0,4 | 49,6 | 47,7 | -1,9 |

## Tabla VIII — Regresión logística quitando una variable

| Variable quitada | AUC PR art. | AUC PR rép. | Δ | Prec. art. | Prec. rép. | Δ | Recall art. | Recall rép. | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fuego previo (día t) | 3,1 | 3,2 | +0,1 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 |
| ERC | 19,6 | 19,8 | +0,2 | 32,4 | 36,3 | +3,9 | 35,5 | 28,6 | -6,9 |
| Elevación | 19,6 | 19,9 | +0,3 | 32,3 | 36,4 | +4,1 | 35,5 | 28,5 | -7,0 |
| Sequía (PDSI) | 19,6 | 19,8 | +0,2 | 32,4 | 36,8 | +4,4 | 35,4 | 27,9 | -7,5 |
| Temp. mínima | 19,6 | 19,7 | +0,1 | 32,2 | 36,2 | +4,0 | 35,5 | 28,2 | -7,3 |
| Dirección del viento | 19,7 | 19,8 | +0,1 | 32,3 | 36,2 | +3,9 | 35,7 | 28,4 | -7,3 |
| Temp. máxima | 19,7 | 19,8 | +0,1 | 32,3 | 36,4 | +4,1 | 35,5 | 28,3 | -7,2 |
| Vegetación (NDVI) | 19,8 | 19,9 | +0,1 | 32,3 | 36,5 | +4,2 | 35,6 | 28,6 | -7,0 |
| Precipitación | 19,8 | 19,9 | +0,1 | 32,3 | 36,6 | +4,3 | 35,6 | 28,4 | -7,2 |
| Población | 19,8 | 19,9 | +0,1 | 32,3 | 36,4 | +4,1 | 35,6 | 28,8 | -6,8 |
| Humedad | 19,8 | 19,9 | +0,1 | 32,4 | 36,4 | +4,0 | 35,6 | 28,6 | -7,0 |
| Velocidad del viento | 20,0 | 20,0 | 0,0 | 32,1 | 36,3 | +4,2 | 35,7 | 28,8 | -6,9 |

## Tabla VI — Random forest: 20 columnas más importantes (Gini)

| # | Artículo | Importancia | Replicación | Importancia |
| ---: | --- | ---: | --- | ---: |
| 1 | Fuego previo (día t) (5) | 0,092 | Fuego previo (día t) (5) | 0,099 |
| 2 | Fuego previo (día t) (6) | 0,074 | Fuego previo (día t) (1) | 0,071 |
| 3 | Fuego previo (día t) (4) | 0,073 | Fuego previo (día t) (4) | 0,069 |
| 4 | Fuego previo (día t) (2) | 0,071 | Fuego previo (día t) (7) | 0,067 |
| 5 | Fuego previo (día t) (1) | 0,068 | Fuego previo (día t) (3) | 0,067 |
| 6 | Fuego previo (día t) (8) | 0,067 | Fuego previo (día t) (8) | 0,065 |
| 7 | Fuego previo (día t) (9) | 0,061 | Fuego previo (día t) (9) | 0,065 |
| 8 | Fuego previo (día t) (3) | 0,052 | Fuego previo (día t) (2) | 0,060 |
| 9 | Fuego previo (día t) (7) | 0,051 | Fuego previo (día t) (6) | 0,053 |
| 10 | ERC (7) | 0,017 | ERC (9) | 0,012 |
| 11 | ERC (8) | 0,014 | ERC (6) | 0,011 |
| 12 | ERC (5) | 0,014 | ERC (7) | 0,009 |
| 13 | ERC (6) | 0,012 | ERC (8) | 0,009 |
| 14 | ERC (9) | 0,010 | ERC (1) | 0,009 |
| 15 | ERC (2) | 0,008 | ERC (2) | 0,008 |
| 16 | ERC (1) | 0,008 | ERC (4) | 0,007 |
| 17 | ERC (4) | 0,008 | ERC (3) | 0,007 |
| 18 | ERC (3) | 0,007 | ERC (5) | 0,006 |
| 19 | Elevación (7) | 0,007 | Elevación (8) | 0,006 |
| 20 | Elevación (9) | 0,005 | Elevación (5) | 0,005 |

Suma de la importancia por variable (replicación, 9 posiciones):

| Variable | Importancia |
| --- | ---: |
| Fuego previo (día t) | 0,616 |
| ERC | 0,078 |
| Dirección del viento | 0,040 |
| Elevación | 0,039 |
| Temp. mínima | 0,037 |
| Velocidad del viento | 0,037 |
| Humedad | 0,035 |
| Sequía (PDSI) | 0,033 |
| Vegetación (NDVI) | 0,030 |
| Temp. máxima | 0,029 |
| Población | 0,020 |
| Precipitación | 0,006 |

## Tabla VII — Regresión logística: 20 coeficientes más grandes (replicación)

| # | Columna | Coeficiente |
| ---: | --- | ---: |
| 1 | Fuego previo (día t) (1) | +1,33 |
| 2 | Fuego previo (día t) (9) | +1,25 |
| 3 | Fuego previo (día t) (3) | +1,05 |
| 4 | Fuego previo (día t) (7) | +1,04 |
| 5 | Fuego previo (día t) (4) | +0,83 |
| 6 | Fuego previo (día t) (6) | +0,78 |
| 7 | ERC (9) | +0,34 |
| 8 | ERC (1) | -0,28 |
| 9 | Fuego previo (día t) (5) | +0,26 |
| 10 | Velocidad del viento (9) | +0,21 |
| 11 | Fuego previo (día t) (8) | +0,20 |
| 12 | ERC (8) | +0,19 |
| 13 | Elevación (9) | -0,18 |
| 14 | Elevación (4) | +0,18 |
| 15 | Fuego previo (día t) (2) | +0,18 |
| 16 | Temp. máxima (8) | -0,17 |
| 17 | ERC (2) | -0,16 |
| 18 | Temp. máxima (7) | -0,15 |
| 19 | ERC (6) | +0,15 |
| 20 | Temp. máxima (3) | +0,14 |
