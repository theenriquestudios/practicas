# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 01:05:10 2026

@author: HUAWEI
"""

# ============================================================
# PROYECTO: DataSystems S.A.
# Regresión, Clasificación y Clustering de Empleados
# ============================================================

# Librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PASO 1: GENERACIÓN DEL DATASET
# ============================================================

np.random.seed(789)

n = 500

df_empleados = pd.DataFrame({
    'experiencia': np.random.normal(5, 2, n),
    'certificaciones': np.random.poisson(3, n),
    'habilidades_sociales': np.random.uniform(1, 10, n),
    'remoto': np.random.choice([0, 1], n)
})

# Variable objetivo continua (Salario)
df_empleados['salario'] = (
    25000
    + df_empleados['experiencia'] * 5000
    + df_empleados['certificaciones'] * 2000
    + np.random.normal(0, 3000, n)
)

# Variable objetivo binaria (Retención)
prob = 1 / (
    1 + np.exp(
        -(-2
          + 0.0001 * df_empleados['salario']
          + 0.2 * df_empleados['habilidades_sociales'])
    )
)

df_empleados['retencion'] = np.where(
    np.random.rand(n) < prob,
    1,
    0
)

print("\nPrimeras filas del dataset:")
print(df_empleados.head())

# ============================================================
# PASO 2: REGRESIÓN LINEAL
# PREDICCIÓN DEL SALARIO
# ============================================================

X_reg = df_empleados[['experiencia', 'certificaciones']]
y_reg = df_empleados['salario']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=123
)

modelo_salario = LinearRegression()

modelo_salario.fit(X_train_reg, y_train_reg)

pred_salario = modelo_salario.predict(X_test_reg)

print("\n==============================")
print("MODELO DE REGRESIÓN LINEAL")
print("==============================")

print(f"Intercepto: {modelo_salario.intercept_:.2f}")

for variable, coef in zip(X_reg.columns, modelo_salario.coef_):
    print(f"{variable}: {coef:.2f}")

rmse = np.sqrt(mean_squared_error(y_test_reg, pred_salario))
r2 = r2_score(y_test_reg, pred_salario)

print(f"\nRMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# ============================================================
# PASO 3: CLASIFICACIÓN KNN
# PREDICCIÓN DE RETENCIÓN
# ============================================================

X_clf = df_empleados[
    ['experiencia', 'habilidades_sociales', 'salario']
]

y_clf = df_empleados['retencion']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf,
    y_clf,
    test_size=0.20,
    random_state=123
)

# Escalamiento + KNN
pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

pipeline_knn.fit(X_train_clf, y_train_clf)

pred_knn = pipeline_knn.predict(X_test_clf)

accuracy = accuracy_score(y_test_clf, pred_knn)

print("\n==============================")
print("MODELO KNN - RETENCIÓN")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test_clf, pred_knn))

print("\nReporte de Clasificación:")
print(classification_report(y_test_clf, pred_knn))

# Comparación de distintos valores de K

print("\nAccuracy para distintos valores de K")

for k in range(1, 11):

    modelo_temp = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])

    modelo_temp.fit(X_train_clf, y_train_clf)

    pred_temp = modelo_temp.predict(X_test_clf)

    acc = accuracy_score(y_test_clf, pred_temp)

    print(f"K = {k}: {acc:.4f}")

# ============================================================
# PASO 4: CLUSTERING K-MEANS
# ============================================================

datos_cluster = df_empleados[
    ['experiencia', 'salario']
]

scaler = StandardScaler()

datos_cluster_escalados = scaler.fit_transform(datos_cluster)

kmeans = KMeans(
    n_clusters=3,
    random_state=456,
    n_init=10
)

clusters = kmeans.fit_predict(datos_cluster_escalados)

df_empleados['cluster'] = clusters.astype(str)

print("\n==============================")
print("TAMAÑO DE LOS CLUSTERS")
print("==============================")

print(df_empleados['cluster'].value_counts())

# ============================================================
# VISUALIZACIÓN
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df_empleados,
    x='experiencia',
    y='salario',
    hue='cluster',
    palette='Set1'
)

plt.title('Segmentación de Empleados por Perfil')
plt.xlabel('Experiencia (años)')
plt.ylabel('Salario')
plt.grid(True)

plt.show()

# ============================================================
# EJEMPLO DE PREDICCIÓN DE SALARIO
# ============================================================

nuevo_empleado = pd.DataFrame({
    'experiencia': [7],
    'certificaciones': [4]
})

salario_estimado = modelo_salario.predict(nuevo_empleado)

print("\n==============================")
print("PREDICCIÓN DE SALARIO")
print("==============================")

print(f"Salario estimado: ${salario_estimado[0]:,.2f}")