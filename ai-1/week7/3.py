"""
ÖRNEK: Polinomsal Regresyon
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Regresyon analizi için
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Normalde iris veri seti gibi iki boyutlu bir veri seti gerekir.
x = [1, 4, 5, 6, 9, 12, 18, 25, 33, 42, 55, 65]
y = [11, 31, 44, 55, 64, 76, 87, 91, 100, 123, 136, 145]

# include_bias, sabit terimin eklenip eklenmeyeceğini belirler.
poly = PolynomialFeatures(degree=3, include_bias=False)

# Verileri modelin kullanabileceği biçime dönüştür
x_poly = poly.fit_transform(x)

# Modeli oluştur
model = LinearRegression()

# Modeli, x'in dönüştürülmüş değerleriyle eğit
model.fit(x_poly, y)

# Tahmin yap
y_tahmin = model.predict(x_poly)

# Hataları hesapla
r2 = r2_score(y, y_tahmin)
mae = mean_absolute_error(y, y_tahmin)
mse = mean_squared_error(y, y_tahmin)
## RMSE hesaplama (MSE'nin karekökü)
rmse = np.square(mse)

print("r2 değer: ", r2)

plt.figure(figsize=(5, 6))
plt.scatter(y, y_tahmin)
