import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

# ============================================================
# 1. Cargar el Dataset
# ============================================================

data = {
    'Superficie_m2': [60, 80, 100, 120, 140, 160, 180, 200],
    'Habitaciones': [1, 2, 2, 3, 3, 4, 4, 5],
    'Precio_kUSD': [110, 145, 170, 210, 230, 275, 295, 340]
}

df = pd.DataFrame(data)

# Variables independientes
X = df[['Superficie_m2', 'Habitaciones']]

# Variable dependiente
y = df['Precio_kUSD']


# ============================================================
# 2. Entrenar la Regresión Lineal Múltiple
# ============================================================
model = LinearRegression()
model.fit(X, y)

# ============================================================
# 3. Realizar Predicciones
# ============================================================
y_pred = model.predict(X)

# ============================================================
# 4. Calcular los errores individuales
# ============================================================

# Error de cada predicción:
# Error = Valor real - Valor predicho

error = y - y_pred

# Error absoluto:
# Permite conocer cuánto se equivocó el modelo
# sin importar si se pasó por encima o por debajo.

error_absoluto = np.abs(error)

# ============================================================
# 5. Crear tabla de resultados
# ============================================================

resultados = pd.DataFrame({
    'Superficie_m2': df['Superficie_m2'],
    'Habitaciones': df['Habitaciones'],
    'Precio_Real_kUSD': y,
    'Precio_Predicho_kUSD': y_pred,
    'Error_kUSD': error,
    'Error_Absoluto_kUSD': error_absoluto
})


# Redondear valores para facilitar la lectura

resultados['Precio_Predicho_kUSD'] = \
    resultados['Precio_Predicho_kUSD'].round(2)

resultados['Error_kUSD'] = \
    resultados['Error_kUSD'].round(2)

resultados['Error_Absoluto_kUSD'] = \
    resultados['Error_Absoluto_kUSD'].round(2)

# ============================================================
# 6. Calcular métricas
# ============================================================
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

# ============================================================
# 7. Mostrar coeficientes
# ============================================================
print("\n==============================================")
print("COEFICIENTES DEL MODELO")
print("==============================================")
print(f"Intersección (b0): {model.intercept_:.2f}")
print(
    f"Coeficiente Superficie (b1): "
    f"{model.coef_[0]:.2f}"
)
print(
    f"Coeficiente Habitaciones (b2): "
    f"{model.coef_[1]:.2f}"
)

# ============================================================
# 8. Mostrar métricas
# ============================================================
print("\n==============================================")
print("MÉTRICAS DE EVALUACIÓN")
print("==============================================")
print(
    f"R² (Coeficiente de Determinación): "
    f"{r2:.4f}"
)
print(
    f"MAE (Error Absoluto Medio): "
    f"{mae:.2f} kUSD"
)
print(
    f"MSE (Error Cuadrático Medio): "
    f"{mse:.2f}"
)
print(
    f"RMSE (Raíz del Error Cuadrático Medio): "
    f"{rmse:.2f} kUSD"
)

# ============================================================
# 9. MOSTRAR TABLA DE VALORES Y ERRORES
# ============================================================

print("\n==============================================")
print("VALORES REALES, PREDICCIONES Y ERRORES")
print("==============================================")
print(resultados.to_string(index=False))

# ============================================================
# 10. Visualizaciones
# ============================================================
plt.figure(figsize=(12, 5))

# ------------------------------------------------------------
# Gráfico 1:
# Valores reales vs. valores predichos
# ------------------------------------------------------------

plt.subplot(1, 2, 1)
plt.scatter(
    y,
    y_pred,
    color='blue',
    edgecolor='k',
    s=70
)
plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    'r--',
    lw=2,
    label='Ajuste Perfecto'
)
plt.title('Valores Reales vs. Predicciones')
plt.xlabel('Precio Real ($k USD)')
plt.ylabel('Precio Predicho ($k USD)')
plt.legend()
plt.grid(
    True,
    linestyle='--',
    alpha=0.6
)

# ------------------------------------------------------------
# Gráfico 2:
# Residuos
# ------------------------------------------------------------
plt.subplot(1, 2, 2)
plt.axhline(
    y=0,
    color='r',
    linestyle='--',
    lw=2
)
plt.scatter(
    y_pred,
    error,
    color='purple',
    edgecolor='k',
    s=70
)
plt.title('Gráfico de Residuos (Errores)')
plt.xlabel('Precio Predicho ($k USD)')
plt.ylabel('Error (Real - Predicho)')
plt.grid(
    True,
    linestyle='--',
    alpha=0.6
)

plt.tight_layout()
plt.show()
