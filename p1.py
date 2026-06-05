# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:43:37 2026

@author: HUAWEI
"""

# ==========================================
# TechManufacture - Análisis Estadístico
# ==========================================

# Librerías
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind

# ------------------------------------------
# Generación del dataset sintético
# ------------------------------------------

np.random.seed(42)

n = 500

turnos = np.random.choice(
    ["Matutino", "Vespertino"],
    size=n,
    p=[0.5, 0.5]
)

temperatura = np.random.normal(
    loc=75,
    scale=5,
    size=n
)

presion = np.random.normal(
    loc=100,
    scale=8,
    size=n
)

# Tasa de error influenciada por temperatura y presión
tasa_error = (
    0.2 * (temperatura - 75)
    + 0.05 * (presion - 100)
    + np.random.normal(0, 1.5, n)
)

# Crear DataFrame
df = pd.DataFrame({
    "turno": turnos,
    "temperatura": temperatura,
    "presion": presion,
    "tasa_error": tasa_error
})

# ------------------------------------------
# Introducir valores faltantes (NA)
# ------------------------------------------

indices_na = np.random.choice(df.index, size=15, replace=False)
df.loc[indices_na, "temperatura"] = np.nan

print("Primeras filas del dataset:")
print(df.head())

# ==========================================
# FASE A: ANÁLISIS EXPLORATORIO (EDA)
# ==========================================

print("\n" + "="*50)
print("FASE A: EDA")
print("="*50)

# 1. Media, mediana y desviación estándar
media_temp = df["temperatura"].mean()
mediana_temp = df["temperatura"].median()
desv_temp = df["temperatura"].std()

print(f"\nMedia temperatura: {media_temp:.2f}")
print(f"Mediana temperatura: {mediana_temp:.2f}")
print(f"Desviación estándar: {desv_temp:.2f}")

# 2. Identificar valores faltantes
print("\nValores faltantes:")
print(df.isna().sum())

# Limpieza mediante imputación con la media
df["temperatura"] = df["temperatura"].fillna(
    df["temperatura"].mean()
)

print("\nValores faltantes después de limpiar:")
print(df.isna().sum())

# 3. Correlación de Pearson
corr, p_value = pearsonr(
    df["temperatura"],
    df["tasa_error"]
)

print("\nCorrelación de Pearson:")
print(f"r = {corr:.4f}")
print(f"p-value = {p_value:.4f}")

# ==========================================
# FASE B: AGRUPACIÓN Y SEGMENTACIÓN
# ==========================================

print("\n" + "="*50)
print("FASE B: AGRUPACIÓN")
print("="*50)

# 1. Comparar tasa de error promedio por turno
promedio_error = (
    df.groupby("turno")["tasa_error"]
    .mean()
    .reset_index()
)

print("\nPromedio de tasa de error por turno:")
print(promedio_error)

# 2. Diferencia significativa en temperatura
temp_matutino = df[df["turno"] == "Matutino"]["temperatura"]
temp_vespertino = df[df["turno"] == "Vespertino"]["temperatura"]

t_stat, p_temp = ttest_ind(
    temp_matutino,
    temp_vespertino,
    equal_var=False
)

print("\nPrueba t para temperatura entre turnos")
print(f"Estadístico t = {t_stat:.4f}")
print(f"p-value = {p_temp:.4f}")

if p_temp < 0.05:
    print("Existe diferencia significativa entre turnos.")
else:
    print("No existe diferencia significativa entre turnos.")

# ==========================================
# FASE C: VISUALIZACIÓN
# ==========================================

sns.set_style("whitegrid")

# ------------------------------------------
# Boxplot
# ------------------------------------------

plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="turno",
    y="tasa_error"
)

plt.title("Distribución de la Tasa de Error por Turno")
plt.xlabel("Turno")
plt.ylabel("Tasa de Error")
plt.show()

# ------------------------------------------
# Scatter Plot + Regresión
# ------------------------------------------

plt.figure(figsize=(8,5))

sns.regplot(
    data=df,
    x="temperatura",
    y="tasa_error",
    scatter_kws={"alpha":0.6}
)

plt.title("Relación entre Temperatura y Tasa de Error")
plt.xlabel("Temperatura")
plt.ylabel("Tasa de Error")
plt.show()

# ==========================================
# CONCLUSIÓN AUTOMÁTICA
# ==========================================

print("\n" + "="*50)
print("CONCLUSIÓN")
print("="*50)

if corr > 0.3:
    print(
        "Se observa una correlación positiva moderada o fuerte "
        "entre la temperatura y la tasa de error. "
        "La empresa debería considerar invertir en sistemas "
        "de enfriamiento para reducir defectos."
    )
else:
    print(
        "La relación entre temperatura y errores es débil. "
        "Se recomienda investigar otros factores además de "
        "la temperatura antes de realizar inversiones."
    )