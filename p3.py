# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:20:00 2026

@author: HUAWEI
"""

# =====================================================
# PCA EN SENSORES INDUSTRIALES - METAL-TECH
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------------------------------
# FASE 1: GENERACIÓN DEL DATASET
# -----------------------------------------------------

np.random.seed(42)

n_muestras = 500

# Variables base para crear correlaciones
temperatura_base = np.random.normal(800, 40, n_muestras)
presion_base = np.random.normal(100, 10, n_muestras)
vibracion_base = np.random.normal(20, 3, n_muestras)

# Sensores correlacionados
datos = pd.DataFrame({
    'Temp_1': temperatura_base + np.random.normal(0, 5, n_muestras),
    'Temp_2': temperatura_base + np.random.normal(0, 4, n_muestras),
    'Temp_3': temperatura_base + np.random.normal(0, 6, n_muestras),

    'Presion_1': presion_base + np.random.normal(0, 2, n_muestras),
    'Presion_2': presion_base + np.random.normal(0, 2, n_muestras),
    'Presion_3': presion_base + np.random.normal(0, 3, n_muestras),

    'Vibracion_1': vibracion_base + np.random.normal(0, 1, n_muestras),
    'Vibracion_2': vibracion_base + np.random.normal(0, 1, n_muestras),

    'Flujo_Gas': temperatura_base * 0.05 + np.random.normal(50, 3, n_muestras),
    'CO2': temperatura_base * 0.03 + np.random.normal(20, 2, n_muestras),

    'Nivel_Agua': np.random.normal(60, 8, n_muestras),
    'Consumo_Energia': temperatura_base * 0.08 +
                        presion_base * 0.5 +
                        np.random.normal(0, 5, n_muestras)
})

print("\nPrimeras filas del dataset:")
print(datos.head())

# -----------------------------------------------------
# MATRIZ DE CORRELACIÓN
# -----------------------------------------------------

correlacion = datos.corr()

print("\nMatriz de correlación:")
print(correlacion.round(2))

plt.figure(figsize=(10,8))
plt.imshow(correlacion, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Correlación')
plt.xticks(range(len(datos.columns)), datos.columns, rotation=90)
plt.yticks(range(len(datos.columns)), datos.columns)
plt.title("Matriz de Correlación de Sensores")
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# FASE 2: ESTANDARIZACIÓN
# -----------------------------------------------------

scaler = StandardScaler()
datos_escalados = scaler.fit_transform(datos)

# -----------------------------------------------------
# PCA
# -----------------------------------------------------

pca = PCA()
componentes = pca.fit_transform(datos_escalados)

# Resumen de varianza explicada
varianza = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza)

resumen = pd.DataFrame({
    'Componente': range(1, len(varianza)+1),
    'Varianza Explicada': varianza,
    'Varianza Acumulada': varianza_acumulada
})

print("\nResumen PCA:")
print(resumen)

# -----------------------------------------------------
# FASE 3: SELECCIÓN DE COMPONENTES
# -----------------------------------------------------

n_componentes = np.argmax(varianza_acumulada >= 0.85) + 1

print("\nNúmero de componentes necesarios para alcanzar 85% de varianza:")
print(n_componentes)

# -----------------------------------------------------
# SCREE PLOT
# -----------------------------------------------------

plt.figure(figsize=(8,5))
plt.plot(range(1, len(varianza)+1),
         varianza,
         marker='o',
         linestyle='--')

plt.xlabel("Componente Principal")
plt.ylabel("Varianza Explicada")
plt.title("Scree Plot")
plt.grid(True)
plt.show()

# -----------------------------------------------------
# LOADINGS DEL PRIMER COMPONENTE
# -----------------------------------------------------

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(datos.columns))],
    index=datos.columns
)

print("\nLoadings del primer componente principal (PC1):")
print(loadings['PC1'].sort_values(key=abs, ascending=False))

# Top sensores más influyentes
print("\nSensores con mayor peso en PC1:")
print(loadings['PC1'].abs().sort_values(ascending=False).head(5))

# -----------------------------------------------------
# PCA REDUCIDO AL 85%
# -----------------------------------------------------

pca_reducido = PCA(n_components=n_componentes)
datos_pca = pca_reducido.fit_transform(datos_escalados)

print(f"\nDimensión original: {datos.shape[1]} variables")
print(f"Dimensión reducida: {n_componentes} componentes")

# -----------------------------------------------------
# FASE 4: BIPLOT
# -----------------------------------------------------

plt.figure(figsize=(10,8))

# Observaciones
plt.scatter(
    componentes[:, 0],
    componentes[:, 1],
    alpha=0.5
)

# Vectores de variables
for i, variable in enumerate(datos.columns):
    plt.arrow(
        0, 0,
        pca.components_[0, i] * 5,
        pca.components_[1, i] * 5,
        head_width=0.05
    )

    plt.text(
        pca.components_[0, i] * 5.2,
        pca.components_[1, i] * 5.2,
        variable
    )

plt.xlabel(
    f'PC1 ({varianza[0]*100:.2f}% varianza)'
)
plt.ylabel(
    f'PC2 ({varianza[1]*100:.2f}% varianza)'
)

plt.title("Biplot PCA - Sensores Metal-Tech")
plt.grid(True)
plt.show()

# -----------------------------------------------------
# INTERPRETACIÓN AUTOMÁTICA
# -----------------------------------------------------

print("\nINTERPRETACIÓN:")

if n_componentes < datos.shape[1]:
    reduccion = (1 - n_componentes/datos.shape[1]) * 100
    print(f"Se logró una reducción del {reduccion:.2f}% "
          f"en la dimensionalidad.")

print("Los sensores con cargas altas en PC1 son los que más")
print("contribuyen al comportamiento global del sistema.")
print("Sensores altamente correlacionados pueden considerarse")
print("redundantes para monitoreo en tiempo real.")