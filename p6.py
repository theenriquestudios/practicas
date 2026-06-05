# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 01:00:31 2026

@author: HUAWEI
"""

# ==========================================================
# PCA PARA SENSORES DE SMART CITY
# Autor: ChatGPT
# ==========================================================

# Librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================================
# PASO 1: GENERACIÓN DE DATOS URBANOS
# ==========================================================

np.random.seed(456)

n = 400

# Datos base de sensores urbanos
datos_urbanos = pd.DataFrame({
    'temp_ambiente': np.random.normal(28, 4, n),
    'humedad_rel': np.random.normal(60, 10, n),
    'co2_ppm': np.nan,
    'no2_ppb': np.nan,
    'particulas_pm25': np.nan,
    'nivel_ruido_db': np.nan,
    'densidad_vehiculos': np.random.normal(120, 30, n),
    'velocidad_viento': np.random.normal(15, 5, n),
    'radiacion_solar': np.random.normal(800, 100, n),
    'conteo_peatones': np.random.normal(50, 15, n)
})

# Creación de correlaciones urbanas
datos_urbanos['co2_ppm'] = (
    datos_urbanos['densidad_vehiculos'] * 3.5
    + np.random.normal(300, 50, n)
)

datos_urbanos['no2_ppb'] = (
    datos_urbanos['co2_ppm'] * 0.1
    + np.random.normal(5, 2, n)
)

datos_urbanos['particulas_pm25'] = (
    datos_urbanos['co2_ppm'] * 0.05
    + np.random.normal(10, 3, n)
)

datos_urbanos['nivel_ruido_db'] = (
    datos_urbanos['densidad_vehiculos'] * 0.2
    + 50
    + np.random.normal(0, 5, n)
)

print("\nPrimeras filas del dataset:")
print(datos_urbanos.head())

# ==========================================================
# PASO 2: ESTANDARIZACIÓN DE DATOS
# ==========================================================

scaler = StandardScaler()

datos_escalados = scaler.fit_transform(datos_urbanos)

# ==========================================================
# PASO 3: EJECUCIÓN DEL PCA
# ==========================================================

pca = PCA()

componentes = pca.fit_transform(datos_escalados)

# Varianza explicada
varianza = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza)

# Tabla resumen
resumen = pd.DataFrame({
    "Componente": np.arange(1, len(varianza)+1),
    "Varianza Explicada (%)": np.round(varianza * 100, 2),
    "Varianza Acumulada (%)": np.round(varianza_acumulada * 100, 2)
})

print("\nResumen PCA:")
print(resumen)

# ==========================================================
# PASO 4: SCREE PLOT
# ==========================================================

plt.figure(figsize=(8,5))
plt.plot(
    range(1, len(varianza)+1),
    varianza * 100,
    marker='o'
)

plt.title("Scree Plot: Sensores Smart City")
plt.xlabel("Componentes Principales")
plt.ylabel("Varianza Explicada (%)")
plt.grid(True)
plt.show()

# ==========================================================
# PASO 5: ANÁLISIS DE VARIANZA ACUMULADA
# ==========================================================

plt.figure(figsize=(8,5))
plt.plot(
    range(1, len(varianza_acumulada)+1),
    varianza_acumulada * 100,
    marker='s'
)

plt.axhline(
    y=85,
    linestyle='--',
    label='85% Varianza'
)

plt.title("Varianza Acumulada")
plt.xlabel("Número de Componentes")
plt.ylabel("Varianza Acumulada (%)")
plt.legend()
plt.grid(True)
plt.show()

# Número mínimo de componentes para explicar 85%
componentes_85 = np.argmax(varianza_acumulada >= 0.85) + 1

print(
    f"\nComponentes necesarios para explicar al menos "
    f"el 85% de la varianza: {componentes_85}"
)

# ==========================================================
# PASO 6: LOADINGS (PESOS DE VARIABLES)
# ==========================================================

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(datos_urbanos.columns))],
    index=datos_urbanos.columns
)

print("\nLoadings de PC1 y PC2:")
print(loadings[['PC1', 'PC2']].round(3))

# ==========================================================
# PASO 7: INTERPRETACIÓN AUTOMÁTICA
# ==========================================================

print("\nVariables con mayor peso en PC1:")
print(loadings["PC1"].abs().sort_values(ascending=False).head())

print("\nVariables con mayor peso en PC2:")
print(loadings["PC2"].abs().sort_values(ascending=False).head())

# ==========================================================
# PASO 8: BIPLOT
# ==========================================================

plt.figure(figsize=(10,8))

# Puntos (observaciones)
plt.scatter(
    componentes[:,0],
    componentes[:,1],
    alpha=0.5
)

# Escalado visual de flechas
escala = 4

for i, variable in enumerate(datos_urbanos.columns):
    plt.arrow(
        0,
        0,
        loadings.iloc[i,0] * escala,
        loadings.iloc[i,1] * escala,
        head_width=0.05
    )

    plt.text(
        loadings.iloc[i,0] * escala * 1.15,
        loadings.iloc[i,1] * escala * 1.15,
        variable,
        fontsize=9
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Biplot: Estado Urbano")
plt.grid(True)
plt.show()

# ==========================================================
# PASO 9: REDUCCIÓN DE DIMENSIONALIDAD
# ==========================================================

pca_reducido = PCA(n_components=2)

datos_reducidos = pca_reducido.fit_transform(datos_escalados)

df_reducido = pd.DataFrame(
    datos_reducidos,
    columns=["Actividad_Vehicular", "Factor_Climatico"]
)

print("\nDatos reducidos a dos macro-indicadores:")
print(df_reducido.head())

# ==========================================================
# CONCLUSIONES AUTOMÁTICAS
# ==========================================================

ahorro = (1 - 2/10) * 100

print("\n================ RESULTADOS =================")
print(f"Sensores originales: 10")
print(f"Componentes finales: 2")
print(f"Reducción de dimensiones: {ahorro:.0f}%")
print(
    f"Varianza explicada por los dos primeros componentes: "
    f"{pca_reducido.explained_variance_ratio_.sum()*100:.2f}%"
)
print("============================================")