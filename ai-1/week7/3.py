"""
ÖRNEK: Polinomsal Regresyon
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

x = [1, 4, 5, 6, 9, 12, 18, 25, 33, 42, 55, 65]
y = [11, 31, 44, 55, 64, 76, 87, 91, 100, 123, 136, 145]

# x dizisini 2 boyutlu sütun vektörüne çevirme
x_2d = np.array(x).reshape(-1, 1)

# 1. Polinom özellik dönüştürücü (3. derece)
poly = PolynomialFeatures(degree=3, include_bias=False)
x_poly = poly.fit_transform(x_2d)
# Çıktı (x_poly ilk 2 satır):
# [[ 1.,  1.,   1.]   -> (x¹, x², x³)
#  [ 4., 16.,  64.]]

# 2. Modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(x_poly, y)

# 3. Tahmin ve Hata Analizi
y_tahmin = model.predict(x_poly)

r2 = r2_score(y, y_tahmin)
mae = mean_absolute_error(y, y_tahmin)
mse = mean_squared_error(y, y_tahmin)
rmse = np.sqrt(mse)  # MSE'nin karekökü

print("r2 değer: ", r2)
# Çıktı: 0.9963... (Eğrisel model veriye %99.6 uyum sağladı)

plt.figure(figsize=(5, 6))
plt.scatter(y, y_tahmin)
plt.show()
