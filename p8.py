# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 01:08:35 2026

@author: HUAWEI
"""

# ==========================================================
# PROYECTO: DETECCIÓN DE ATAQUES EN TRÁFICO DE RED
# Regresión Logística + KNN + K-Means
# ==========================================================

# Librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ==========================================================
# PASO 1: SIMULACIÓN DE TRÁFICO DE RED
# ==========================================================

np.random.seed(444)

n = 400

df_red = pd.DataFrame({
    'latencia': np.random.normal(20, 5, n),
    'intentos_fallidos': np.random.poisson(1, n),
    'tamano_paquete': np.random.normal(500, 100, n)
})

# Ataque si intentos > 2 o latencia > 35
df_red['es_ataque'] = np.where(
    (df_red['intentos_fallidos'] > 2) |
    (df_red['latencia'] > 35),
    1,
    0
)

print("Primeras filas:")
print(df_red.head())

print("\nDistribución de clases:")
print(df_red['es_ataque'].value_counts())

# ==========================================================
# EXPLORACIÓN DE DATOS
# ==========================================================

print("\nEstadísticas descriptivas:")
print(df_red.describe())

sns.pairplot(
    df_red,
    hue='es_ataque',
    diag_kind='hist'
)
plt.show()

# ==========================================================
# PREPARACIÓN DE DATOS
# ==========================================================

X = df_red.drop('es_ataque', axis=1)
y = df_red['es_ataque']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Escalamiento
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# PASO 2: REGRESIÓN LOGÍSTICA
# ==========================================================

modelo_log = LogisticRegression(max_iter=1000)

modelo_log.fit(X_train_scaled, y_train)

pred_log = modelo_log.predict(X_test_scaled)

print("\n==============================")
print("REGRESIÓN LOGÍSTICA")
print("==============================")

print("\nAccuracy:")
print(accuracy_score(y_test, pred_log))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, pred_log))

print("\nReporte de clasificación:")
print(classification_report(y_test, pred_log))

# Coeficientes
coeficientes = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo_log.coef_[0]
})

print("\nImportancia de variables:")
print(coeficientes.sort_values(
    by="Coeficiente",
    ascending=False
))

# ==========================================================
# INTERPRETACIÓN DOCENTE
# ==========================================================
#
# Un coeficiente positivo indica que al aumentar la variable
# incrementa la probabilidad de ataque.
#
# Si "intentos_fallidos" posee el coeficiente más alto,
# significa que es un indicador crítico de intrusión.
#
# ==========================================================

# ==========================================================
# PASO 3: CLASIFICACIÓN CON KNN
# ==========================================================

print("\n==============================")
print("KNN")
print("==============================")

k_values = range(1, 21)
accuracy_scores = []

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        knn,
        X_train_scaled,
        y_train,
        cv=5,
        scoring='accuracy'
    )

    accuracy_scores.append(scores.mean())

# Mejor K
best_k = k_values[np.argmax(accuracy_scores)]

print(f"\nMejor valor de K: {best_k}")
print(f"Accuracy CV: {max(accuracy_scores):.4f}")

# Entrenamiento final
modelo_knn = KNeighborsClassifier(
    n_neighbors=best_k
)

modelo_knn.fit(X_train_scaled, y_train)

pred_knn = modelo_knn.predict(X_test_scaled)

print("\nAccuracy en prueba:")
print(accuracy_score(y_test, pred_knn))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, pred_knn))

# Gráfico Accuracy vs K
plt.figure(figsize=(8,5))
plt.plot(k_values, accuracy_scores, marker='o')
plt.xlabel("Número de Vecinos (K)")
plt.ylabel("Accuracy")
plt.title("Selección del Mejor K")
plt.grid(True)
plt.show()

# ==========================================================
# INTERPRETACIÓN DOCENTE
# ==========================================================
#
# El gráfico muestra cómo cambia la precisión al modificar K.
#
# K pequeño:
#   - Mayor sensibilidad
#   - Riesgo de sobreajuste
#
# K grande:
#   - Modelo más estable
#   - Puede ignorar ataques reales
#
# El mejor K es aquel que maximiza la Accuracy.
#
# ==========================================================

# ==========================================================
# PASO 4: CLUSTERING K-MEANS
# ==========================================================

print("\n==============================")
print("K-MEANS")
print("==============================")

X_cluster = StandardScaler().fit_transform(
    df_red.drop('es_ataque', axis=1)
)

kmeans = KMeans(
    n_clusters=2,
    random_state=111,
    n_init=10
)

clusters = kmeans.fit_predict(X_cluster)

df_red["cluster"] = clusters

# Comparación con la realidad
comparacion = pd.crosstab(
    df_red['es_ataque'],
    df_red['cluster'],
    rownames=['Real'],
    colnames=['Cluster']
)

print("\nTabla de comparación:")
print(comparacion)

# ==========================================================
# VISUALIZACIÓN DE CLUSTERS
# ==========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df_red,
    x='latencia',
    y='intentos_fallidos',
    hue='cluster',
    palette='Set1'
)

plt.title("Clusters Descubiertos por K-Means")
plt.show()

# ==========================================================
# INTERPRETACIÓN DOCENTE
# ==========================================================
#
# Si un cluster contiene principalmente ataques
# y el otro conexiones seguras, significa que:
#
# - Existe una firma digital clara del comportamiento
#   malicioso.
#
# - El algoritmo puede detectar patrones anómalos
#   incluso sin conocer previamente la etiqueta.
#
# - Esto demuestra el potencial de los métodos
#   no supervisados para ciberseguridad.
#
# ==========================================================

# ==========================================================
# RESUMEN FINAL
# ==========================================================

print("\n=========== RESUMEN ===========")

print(f"Accuracy Regresión Logística: "
      f"{accuracy_score(y_test, pred_log):.4f}")

print(f"Accuracy KNN: "
      f"{accuracy_score(y_test, pred_knn):.4f}")

print(f"Mejor K encontrado: {best_k}")

print("\nProceso completado correctamente.")