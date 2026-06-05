# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 00:34:52 2026

@author: HUAWEI
"""

# ==========================================================
# PCA PARA SENSORES INDUSTRIALES - PLANTA METAL-TECH
# ==========================================================

# Librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------------------------------------
# 1. GENERACIÓN DEL DATASET
# ----------------------------------------------------------

np.random.seed(42)

n = 200

temp1 = np.random.normal(100, 5, n)
temp2 = temp1 + np.random.normal(0, 1, n)

presion1 = np.random.normal(50, 10, n)

vibracion = (temp1 * 0.5) + np.random.normal(0, 2, n)

datos_sensores = pd.DataFrame({
    'temp_nucleo': temp1,
    'temp_escape': temp2,
    'presion_interna': presion1,
    'vibracion_motor': vibracion,
    'flujo_gas': np.random.normal(20, 2, n),
    'oxigeno': np.random.normal(15, 1, n),
    'co2': temp1 * 0.2 + np.random.normal(0, 0.5, n),
    'humedad': np.random.uniform(30, 40, n),
    'voltaje': np.random.normal(220, 2, n),
    'corriente': np.random.normal(15, 0.5, n),
    'ruido_db': vibracion * 1.2 + np.random.normal(0, 1, n),
    'eficiencia': 100 - (temp1 * 0.1)
})

print("\nPrimeras filas del dataset:")
print(datos_sensores.head())

# ----------------------------------------------------------
# 2. ESTANDARIZACIÓN DE DATOS
# ----------------------------------------------------------

escalador = StandardScaler()

datos_escalados = escalador.fit_transform(datos_sensores)

# ----------------------------------------------------------
# 3. EJECUCIÓN DEL PCA
# ----------------------------------------------------------

pca = PCA()

componentes = pca.fit_transform(datos_escalados)

# ----------------------------------------------------------
# 4. VARIANZA EXPLICADA
# ----------------------------------------------------------

varianza = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza)

tabla_varianza = pd.DataFrame({
    'Componente': np.arange(1, len(varianza)+1),
    'Varianza Explicada (%)': np.round(varianza*100, 2),
    'Varianza Acumulada (%)': np.round(varianza_acumulada*100, 2)
})

print("\nResumen de Varianza:")
print(tabla_varianza)

# Encontrar número de componentes para 85%
n_componentes_85 = np.argmax(varianza_acumulada >= 0.85) + 1

print("\nComponentes necesarios para explicar al menos 85% de la varianza:")
print(n_componentes_85)

# ----------------------------------------------------------
# 5. SCREE PLOT
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    range(1, len(varianza)+1),
    pca.explained_variance_,
    marker='o'
)

plt.axhline(y=1, color='red', linestyle='--')

plt.xlabel("Componentes Principales")
plt.ylabel("Autovalor")
plt.title("Scree Plot - Sensores Industriales")
plt.grid(True)

plt.show()

# ----------------------------------------------------------
# 6. LOADINGS (PESOS DE VARIABLES)
# ----------------------------------------------------------

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(datos_sensores.columns))],
    index=datos_sensores.columns
)

print("\nLoadings de PC1 y PC2:")
print(loadings[['PC1', 'PC2']].round(3))

# Variables más influyentes
print("\nVariables más importantes en PC1:")
print(loadings['PC1'].abs().sort_values(ascending=False).head())

print("\nVariables más importantes en PC2:")
print(loadings['PC2'].abs().sort_values(ascending=False).head())

# ----------------------------------------------------------
# 7. BIPLOT
# ----------------------------------------------------------

plt.figure(figsize=(10,8))

scores = componentes[:, :2]

plt.scatter(
    scores[:,0],
    scores[:,1],
    alpha=0.5
)

for i, variable in enumerate(datos_sensores.columns):

    plt.arrow(
        0,
        0,
        loadings.iloc[i,0]*5,
        loadings.iloc[i,1]*5,
        head_width=0.05
    )

    plt.text(
        loadings.iloc[i,0]*5.2,
        loadings.iloc[i,1]*5.2,
        variable
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Biplot de Sensores Industriales")
plt.grid(True)

plt.show()

# ----------------------------------------------------------
# 8. MATRIZ DE CORRELACIÓN
# ----------------------------------------------------------

correlacion = datos_sensores.corr()

print("\nMatriz de correlación:")
print(correlacion.round(2))

# Detectar sensores redundantes
print("\nPosibles sensores redundantes (|correlación| > 0.85):")

for i in range(len(correlacion.columns)):
    for j in range(i+1, len(correlacion.columns)):

        corr = correlacion.iloc[i,j]

        if abs(corr) > 0.85:
            print(
                correlacion.columns[i],
                "<-->",
                correlacion.columns[j],
                f"(corr={corr:.3f})"
            )

# ----------------------------------------------------------
# 9. CONCLUSIONES AUTOMÁTICAS
# ----------------------------------------------------------

print("\n" + "="*60)
print("CONCLUSIONES")
print("="*60)

print(
    f"\n1. El sistema puede reducirse de 12 sensores "
    f"a aproximadamente {n_componentes_85} componentes "
    f"manteniendo más del 85% de la información."
)

print(
    "\n2. Sensores como temp_nucleo, temp_escape, "
    "co2, vibracion_motor y ruido_db probablemente "
    "presentan redundancia debido a su alta correlación."
)

print(
    "\n3. La reducción dimensional disminuye el tráfico "
    "de datos enviado al PLC, reduce el consumo de ancho "
    "de banda y mejora el procesamiento en tiempo real."
)