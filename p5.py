# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 00:39:28 2026

@author: HUAWEI
"""

# ==========================================
# PCA PARA ANÁLISIS DE TRÁFICO DE RED
# Centro de Datos - Detección de Anomalías
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------------------------
# PASO 1: Generación de Datos Simulados
# ------------------------------------------

np.random.seed(123)

n = 300

datos_red = pd.DataFrame({
    'duracion_ms': np.random.normal(50, 10, n),
    'paquetes_enviados': np.random.normal(100, 20, n),
    'errores_checksum': np.random.poisson(2, n),
    'latencia_avg': np.random.normal(15, 5, n),
    'jitter': np.random.normal(2, 0.5, n),
    'uso_memoria_sw': np.random.normal(40, 10, n),
    'peticiones_http': np.random.normal(200, 50, n)
})

# Crear dependencias (redundancia)
datos_red['bytes_enviados'] = (
    datos_red['paquetes_enviados'] * 1500 +
    np.random.normal(0, 500, n)
)

datos_red['reintentos_tcp'] = (
    datos_red['errores_checksum'] * 1.5 +
    np.random.normal(0, 0.5, n)
)

datos_red['carga_cpu_router'] = (
    datos_red['paquetes_enviados'] * 0.4 +
    datos_red['latencia_avg'] * 0.2
)

# Reordenar columnas
datos_red = datos_red[
    [
        'duracion_ms',
        'paquetes_enviados',
        'bytes_enviados',
        'errores_checksum',
        'reintentos_tcp',
        'latencia_avg',
        'jitter',
        'carga_cpu_router',
        'uso_memoria_sw',
        'peticiones_http'
    ]
]

print("\nPrimeras filas del dataset:")
print(datos_red.head())

# ------------------------------------------
# Correlación entre variables
# ------------------------------------------

print("\nMatriz de correlación:")
print(datos_red.corr().round(2))

# ------------------------------------------
# PASO 2: Estandarización
# ------------------------------------------

scaler = StandardScaler()
datos_escalados = scaler.fit_transform(datos_red)

print("\nDatos estandarizados correctamente.")

# ------------------------------------------
# PASO 3: Aplicación de PCA
# ------------------------------------------

pca = PCA()
componentes = pca.fit_transform(datos_escalados)

# Varianza explicada
varianza = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza)

print("\nVarianza explicada por componente:")
for i, v in enumerate(varianza):
    print(f"PC{i+1}: {v:.4f}")

print("\nVarianza acumulada:")
for i, v in enumerate(varianza_acumulada):
    print(f"PC{i+1}: {v:.4f}")

# ------------------------------------------
# Scree Plot
# ------------------------------------------

plt.figure(figsize=(8,5))
plt.plot(
    range(1, len(varianza)+1),
    varianza,
    marker='o'
)
plt.title("Scree Plot - Tráfico de Red")
plt.xlabel("Componentes Principales")
plt.ylabel("Varianza Explicada")
plt.grid(True)
plt.show()

# ------------------------------------------
# PASO 4: Cargas (Loadings)
# ------------------------------------------

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(datos_red.columns))],
    index=datos_red.columns
)

print("\nCargas de PC1 y PC2:")
print(loadings[['PC1', 'PC2']].round(3))

# ------------------------------------------
# Interpretación automática
# ------------------------------------------

print("\nVariables más importantes en PC1:")
print(loadings['PC1'].abs().sort_values(ascending=False).head())

print("\nVariables más importantes en PC2:")
print(loadings['PC2'].abs().sort_values(ascending=False).head())

# ------------------------------------------
# PASO 5: Biplot
# ------------------------------------------

plt.figure(figsize=(10,8))

# Observaciones
plt.scatter(
    componentes[:, 0],
    componentes[:, 1],
    alpha=0.6
)

# Flechas de variables
for i, variable in enumerate(datos_red.columns):
    plt.arrow(
        0,
        0,
        loadings.iloc[i, 0] * 5,
        loadings.iloc[i, 1] * 5,
        alpha=0.7
    )

    plt.text(
        loadings.iloc[i, 0] * 5.2,
        loadings.iloc[i, 1] * 5.2,
        variable,
        fontsize=9
    )

plt.axhline(0)
plt.axvline(0)

plt.xlabel(
    f"PC1 ({varianza[0]*100:.2f}% varianza)"
)

plt.ylabel(
    f"PC2 ({varianza[1]*100:.2f}% varianza)"
)

plt.title("Mapa de Estado de Red (Biplot PCA)")
plt.grid(True)
plt.show()

# ------------------------------------------
# Detección simple de anomalías
# ------------------------------------------

distancias = np.sqrt(
    componentes[:,0]**2 +
    componentes[:,1]**2
)

umbral = np.percentile(distancias, 95)

anomalias = np.where(distancias > umbral)[0]

print(f"\nNúmero de conexiones anómalas detectadas: {len(anomalias)}")
print("Índices de anomalías:")
print(anomalias)

# Visualización de anomalías
plt.figure(figsize=(10,7))

plt.scatter(
    componentes[:,0],
    componentes[:,1],
    alpha=0.5,
    label='Normal'
)

plt.scatter(
    componentes[anomalias,0],
    componentes[anomalias,1],
    s=80,
    marker='x',
    label='Anomalía'
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Detección de Anomalías usando PCA")
plt.legend()
plt.grid(True)
plt.show()