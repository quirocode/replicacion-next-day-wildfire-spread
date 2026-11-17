# Comparación de resultados: artículo vs. replicación

Todas las métricas están en porcentaje y se midieron en el conjunto de prueba,
sobre la muestra completa de 64 x 64 km (1 689 muestras).
art. = artículo; rép. = esta replicación; Δ = replicación menos artículo (en puntos).

## Tabla II — Predicción de la propagación

| Modelo | AUC PR art. | AUC PR rép. | Δ | Prec. art. | Prec. rép. | Δ | Recall art. | Recall rép. | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Red neuronal | 28,4 | 25,1 | -3,3 | 33,6 | 30,0 | -3,6 | 43,1 | 42,7 | -0,4 |
| Random forest | 22,5 | 21,7 | -0,8 | 26,3 | 28,9 | +2,6 | 46,9 | 44,0 | -2,9 |
| Regresión logística | 19,8 | 19,9 | +0,1 | 32,5 | 36,4 | +3,9 | 35,3 | 28,3 | -7,0 |
| Persistencia | 11,5 | 11,5 | 0,0 | 35,7 | 35,7 | 0,0 | 27,3 | 27,3 | 0,0 |

Los mismos modelos medidos sobre recortes aleatorios de 32 x 32 km:

| Modelo (recortes de 32 x 32 km) | AUC PR art. | AUC PR rép. | Δ | Prec. art. | Prec. rép. | Δ | Recall art. | Recall rép. | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Red neuronal | 28,4 | 28,4 | 0,0 | 33,6 | 33,7 | +0,1 | 43,1 | 42,2 | -0,9 |
| Persistencia | 11,5 | 14,3 | +2,8 | 35,7 | 40,1 | +4,4 | 27,3 | 26,4 | -0,9 |

- Red neuronal: 28 080 pasos de entrenamiento (60 épocas x 468 pasos en CPU); el artículo usó 1 000 000 de pasos en 4 GPU V100.
- Mejor época según validación: 58 (AUC PR de validación 21,4 %).
- Variación del AUC PR de la red entre tres juegos de recortes distintos: ±0,12 puntos.
- Regresión logística: W = 7 (el del artículo), 1 000 000 píxeles de entrenamiento. El mejor W en nuestra validación fue 1; con él, en prueba: AUC PR 19,7 %, precisión 21,2 %, recall 45,7 %.
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
| Las 12 variables (Tabla II) | 28,4 | 25,1 | -3,3 | 33,6 | 30,0 | -3,6 | 43,1 | 42,7 | -0,4 |
| Sin el fuego previo (Tabla IV) | 6,8 | 3,8 | -3,0 | 4,6 | 18,2 | +13,6 | 0,2 | 0,6 | +0,4 |
| Solo el fuego previo (Sección V-B) | 22,9 | 18,9 | -4,0 | 29,2 | 21,1 | -8,1 | 49,6 | 53,9 | +4,3 |

Medido sobre recortes de 32 x 32 km:

| Variables de entrada | AUC PR art. | AUC PR rép. | Δ | Prec. art. | Prec. rép. | Δ | Recall art. | Recall rép. | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Las 12 variables (Tabla II) | 28,4 | 28,4 | 0,0 | 33,6 | 33,7 | +0,1 | 43,1 | 42,2 | -0,9 |
| Sin el fuego previo (Tabla IV) | 6,8 | 6,9 | +0,1 | 4,6 | 20,4 | +15,8 | 0,2 | 0,2 | 0,0 |
| Solo el fuego previo (Sección V-B) | 22,9 | 21,3 | -1,6 | 29,2 | 25,8 | -3,4 | 49,6 | 52,0 | +2,4 |

Experimento adicional, fuera del artículo: las 12 variables con tasa de aprendizaje 0,001 y 30 épocas (14 040 pasos): AUC PR 23,7 / precisión 33,8 / recall 32,5 sobre 64 x 64 km; 28,0 / 37,8 / 34,4 sobre recortes de 32 x 32 km.

## Tabla VIII — Regresión logística quitando una variable

| Variable quitada | AUC PR art. | AUC PR rép. | Δ | Prec. art. | Prec. rép. | Δ | Recall art. | Recall rép. | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fuego previo (día t) | 3,1 | 3,1 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 |
| ERC | 19,6 | 19,8 | +0,2 | 32,4 | 36,4 | +4,0 | 35,5 | 28,3 | -7,2 |
| Elevación | 19,6 | 19,9 | +0,3 | 32,3 | 36,4 | +4,1 | 35,5 | 28,4 | -7,1 |
| Sequía (PDSI) | 19,6 | 19,8 | +0,2 | 32,4 | 36,6 | +4,2 | 35,4 | 28,1 | -7,3 |
| Temp. mínima | 19,6 | 19,9 | +0,3 | 32,2 | 36,3 | +4,1 | 35,5 | 28,3 | -7,2 |
| Dirección del viento | 19,7 | 19,8 | +0,1 | 32,3 | 36,2 | +3,9 | 35,7 | 28,6 | -7,1 |
| Temp. máxima | 19,7 | 19,8 | +0,1 | 32,3 | 36,5 | +4,2 | 35,5 | 28,1 | -7,4 |
| Vegetación (NDVI) | 19,8 | 19,9 | +0,1 | 32,3 | 36,6 | +4,3 | 35,6 | 28,5 | -7,1 |
| Precipitación | 19,8 | 19,9 | +0,1 | 32,3 | 36,3 | +4,0 | 35,6 | 28,8 | -6,8 |
| Población | 19,8 | 19,9 | +0,1 | 32,3 | 36,4 | +4,1 | 35,6 | 28,6 | -7,0 |
| Humedad | 19,8 | 19,9 | +0,1 | 32,4 | 36,5 | +4,1 | 35,6 | 28,5 | -7,1 |
| Velocidad del viento | 20,0 | 20,0 | 0,0 | 32,1 | 36,3 | +4,2 | 35,7 | 28,8 | -6,9 |

## Tabla VI — Random forest: 20 columnas más importantes (Gini)

| # | Artículo | Importancia | Replicación | Importancia |
| ---: | --- | ---: | --- | ---: |
| 1 | Fuego previo (día t) (5) | 0,092 | Fuego previo (día t) (1) | 0,081 |
| 2 | Fuego previo (día t) (6) | 0,074 | Fuego previo (día t) (4) | 0,079 |
| 3 | Fuego previo (día t) (4) | 0,073 | Fuego previo (día t) (5) | 0,079 |
| 4 | Fuego previo (día t) (2) | 0,071 | Fuego previo (día t) (7) | 0,068 |
| 5 | Fuego previo (día t) (1) | 0,068 | Fuego previo (día t) (8) | 0,067 |
| 6 | Fuego previo (día t) (8) | 0,067 | Fuego previo (día t) (9) | 0,066 |
| 7 | Fuego previo (día t) (9) | 0,061 | Fuego previo (día t) (2) | 0,064 |
| 8 | Fuego previo (día t) (3) | 0,052 | Fuego previo (día t) (3) | 0,060 |
| 9 | Fuego previo (día t) (7) | 0,051 | Fuego previo (día t) (6) | 0,055 |
| 10 | ERC (7) | 0,017 | ERC (9) | 0,012 |
| 11 | ERC (8) | 0,014 | ERC (6) | 0,010 |
| 12 | ERC (5) | 0,014 | ERC (8) | 0,010 |
| 13 | ERC (6) | 0,012 | ERC (7) | 0,009 |
| 14 | ERC (9) | 0,010 | ERC (1) | 0,008 |
| 15 | ERC (2) | 0,008 | ERC (2) | 0,008 |
| 16 | ERC (1) | 0,008 | ERC (4) | 0,007 |
| 17 | ERC (4) | 0,008 | ERC (3) | 0,007 |
| 18 | ERC (3) | 0,007 | ERC (5) | 0,006 |
| 19 | Elevación (7) | 0,007 | Elevación (5) | 0,005 |
| 20 | Elevación (9) | 0,005 | Dirección del viento (2) | 0,005 |

Suma de la importancia por variable (replicación, 9 posiciones):

| Variable | Importancia |
| --- | ---: |
| Fuego previo (día t) | 0,620 |
| ERC | 0,077 |
| Dirección del viento | 0,041 |
| Elevación | 0,039 |
| Temp. mínima | 0,035 |
| Velocidad del viento | 0,035 |
| Humedad | 0,035 |
| Sequía (PDSI) | 0,032 |
| Vegetación (NDVI) | 0,031 |
| Temp. máxima | 0,029 |
| Población | 0,021 |
| Precipitación | 0,006 |

## Tabla VII — Regresión logística: 20 coeficientes más grandes (replicación)

| # | Columna | Coeficiente |
| ---: | --- | ---: |
| 1 | Fuego previo (día t) (1) | +1,37 |
| 2 | Fuego previo (día t) (9) | +1,29 |
| 3 | Fuego previo (día t) (3) | +1,03 |
| 4 | Fuego previo (día t) (7) | +1,01 |
| 5 | Fuego previo (día t) (4) | +0,84 |
| 6 | Fuego previo (día t) (6) | +0,80 |
| 7 | ERC (9) | +0,29 |
| 8 | ERC (1) | -0,23 |
| 9 | Fuego previo (día t) (5) | +0,22 |
| 10 | Velocidad del viento (9) | +0,20 |
| 11 | Elevación (9) | -0,18 |
| 12 | ERC (8) | +0,17 |
| 13 | Elevación (4) | +0,17 |
| 14 | Fuego previo (día t) (8) | +0,17 |
| 15 | Fuego previo (día t) (2) | +0,16 |
| 16 | Temp. mínima (9) | +0,15 |
| 17 | ERC (6) | +0,14 |
| 18 | Temp. máxima (7) | -0,13 |
| 19 | ERC (2) | -0,13 |
| 20 | Temp. máxima (8) | -0,12 |
