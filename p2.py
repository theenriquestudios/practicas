# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:11:33 2026

@author: HUAWEI
"""

# ==========================================
# Global-Logistics ISC
# Análisis Estadístico de Envíos Nacionales
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind

# --------------------------------------------------
# FASE 0: Generación del Dataset (600 registros)
# --------------------------------------------------

np.random.seed(42)

n = 600

# Modalidades de transporte
tipo_transporte = np.random.choice(
    ['Terrestre', 'Aéreo'],
    size=n,
    p=[0.65, 0.35]
)

# Condiciones climáticas
clima = np.random.choice(
    ['Normal', 'Lluvia', 'Tormenta'],
    size=n,
    p=[0.6, 0.3, 0.1]
)

# Distancias simuladas
distancia_km = np.random.randint(50, 2500, n)

# Introducir valores faltantes (NA)
indices_na = np.random.choice(n, size=30, replace=False)
distancia_km = distancia_km.astype(float)
distancia_km[indices_na] = np.nan

# Tiempo de entrega según modalidad
tiempo_entrega = []

for i in range(n):

    dist = distancia_km[i]

    # Usar distancia temporal para simulación
    if np.isnan(dist):
        dist = np.random.randint(50, 2500)

    if tipo_transporte[i] == 'Terrestre':
        tiempo = dist / 60 + np.random.normal(5, 2)
    else:
        tiempo = dist / 500 + np.random.normal(2, 0.8)

    # Efecto del clima
    if clima[i] == 'Lluvia':
        tiempo += np.random.uniform(1, 4)
    elif clima[i] == 'Tormenta':
        tiempo += np.random.uniform(3, 8)

    tiempo_entrega.append(tiempo)

tiempo_entrega = np.array(tiempo_entrega)

# Costos simulados
costo = []

for i in range(n):
    dist = distancia_km[i]

    if np.isnan(dist):
        dist = 1000

    if tipo_transporte[i] == 'Terrestre':
        costo.append(dist * 0.80 + np.random.normal(50, 20))
    else:
        costo.append(dist * 1.50 + np.random.normal(100, 40))

# Crear DataFrame
df = pd.DataFrame({
    'tipo_transporte': tipo_transporte,
    'clima': clima,
    'distancia_km': distancia_km,
    'tiempo_entrega_hrs': tiempo_entrega,
    'costo_envio': costo
})

print("\nPrimeros registros:")
print(df.head())

# --------------------------------------------------
# FASE 1: Calidad de Datos
# --------------------------------------------------

print("\n==============================")
print("FASE 1: CALIDAD DE DATOS")
print("==============================")

# 1. Imputación por mediana
mediana_distancia = df['distancia_km'].median()

df['distancia_km'].fillna(
    mediana_distancia,
    inplace=True
)

print(f"\nMediana utilizada: {mediana_distancia:.2f} km")

# 2. Estadísticos descriptivos
media_tiempo = df['tiempo_entrega_hrs'].mean()
desv_tiempo = df['tiempo_entrega_hrs'].std()

print(f"\nMedia tiempo entrega: {media_tiempo:.2f} hrs")
print(f"Desviación estándar: {desv_tiempo:.2f} hrs")

# --------------------------------------------------
# FASE 2: Correlación
# --------------------------------------------------

print("\n==============================")
print("FASE 2: CORRELACIÓN")
print("==============================")

corr, p_valor = pearsonr(
    df['distancia_km'],
    df['tiempo_entrega_hrs']
)

print(f"\nCoeficiente Pearson: {corr:.4f}")
print(f"p-value: {p_valor:.8f}")

# --------------------------------------------------
# FASE 3: Comparación de Modalidades
# --------------------------------------------------

print("\n==============================")
print("FASE 3: COMPARACIÓN")
print("==============================")

agrupado = df.groupby(
    'tipo_transporte'
).agg({
    'tiempo_entrega_hrs': 'mean',
    'costo_envio': 'mean'
})

print("\nPromedios por modalidad:")
print(agrupado)

# T-Test
terrestre = df[
    df['tipo_transporte'] == 'Terrestre'
]['tiempo_entrega_hrs']

aereo = df[
    df['tipo_transporte'] == 'Aéreo'
]['tiempo_entrega_hrs']

t_stat, p_ttest = ttest_ind(
    terrestre,
    aereo,
    equal_var=False
)

print("\nResultados T-Test")
print(f"Estadístico t: {t_stat:.4f}")
print(f"p-value: {p_ttest:.8f}")

if p_ttest < 0.05:
    print("Existe diferencia significativa entre modalidades.")
else:
    print("No existe diferencia significativa.")

# --------------------------------------------------
# FASE 4: Visualizaciones
# --------------------------------------------------

sns.set_style("whitegrid")

# Gráfico de dispersión
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='distancia_km',
    y='tiempo_entrega_hrs',
    hue='tipo_transporte'
)

plt.title(
    'Distancia vs Tiempo de Entrega'
)

plt.xlabel('Distancia (km)')
plt.ylabel('Tiempo de Entrega (hrs)')

plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x='tipo_transporte',
    y='tiempo_entrega_hrs'
)

plt.title(
    'Comparación de Tiempos de Entrega'
)

plt.xlabel('Modalidad')
plt.ylabel('Tiempo de Entrega (hrs)')

plt.tight_layout()
plt.show()

# --------------------------------------------------
# CONCLUSIÓN AUTOMÁTICA
# --------------------------------------------------

print("\n==============================")
print("RECOMENDACIÓN")
print("==============================")

promedio_terrestre = terrestre.mean()
promedio_aereo = aereo.mean()

print(f"\nTiempo promedio terrestre: {promedio_terrestre:.2f} hrs")
print(f"Tiempo promedio aéreo: {promedio_aereo:.2f} hrs")

if p_ttest < 0.05 and promedio_aereo < promedio_terrestre:
    print(
        "\nRECOMENDACIÓN: "
        "Los datos sugieren migrar una mayor proporción "
        "de envíos al transporte aéreo para reducir los "
        "tiempos de entrega y mejorar la satisfacción del cliente."
    )
else:
    print(
        "\nRECOMENDACIÓN: "
        "No existe evidencia estadística suficiente para justificar "
        "una migración masiva al transporte aéreo."
    )