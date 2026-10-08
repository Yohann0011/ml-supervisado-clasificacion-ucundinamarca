import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Creación de un dataset sintético
np.random.seed(42)
n_samples = 100

area_m2 = np.random.uniform(40, 200, n_samples)          # Entre 40 y 200 m2
habitaciones = np.random.randint(1, 6, n_samples)        # Entre 1 y 5 habitaciones

# Precio real = 30,000 (base) + 2,500*(m2) + 15,000*(hab) + ruido aleatorio
precio_real = 30000 + (2500 * area_m2) + (15000 * habitaciones) + np.random.normal(0, 15000, n_samples)

df = pd.DataFrame({
    'Area_m2': area_m2,
    'Habitaciones': habitaciones,
    'Precio': precio_real
})

# 2. Separación de variables (X e y)
X = df[['Area_m2', 'Habitaciones']]
y = df['Precio']

# 3. Entrenamiento del modelo
modelo = LinearRegression()
modelo.fit(X, y)

# 4. Extracción de coeficientes
beta_0 = modelo.intercept_
beta_1, beta_2 = modelo.coef_

print(f"Intercepto (Beta 0): ${beta_0:,.2f}")
print(f"Coeficiente Área (Beta 1): ${beta_1:,.2f} por m²")
print(f"Coeficiente Habitaciones (Beta 2): ${beta_2:,.2f} por habitación")

# 5. Predicción para una casa de 85 m² y 3 habitaciones
nueva_casa = np.array([[85, 3]])
precio_predicho = modelo.predict(nueva_casa)
print(f"\nPrecio estimado para 85m² y 3 hab: ${precio_predicho[0]:,.2f}")
