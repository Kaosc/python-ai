import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = [1, 4, 5, 6, 9, 12, 18, 25, 33, 42, 55, 65]
y = [11, 31, 44, 55, 64, 76, 87, 91, 100, 123, 136, 145]

# 1. Tek satırda Doğrusal Regresyon (y = slope * x + intercept)
slope, intercept, r, p, std_err = stats.linregress(x, y)
# Çıktı: slope ≈ 1.88 (Eğim), intercept ≈ 32.16 (Kesim noktası)

def yontem(x):
    return slope * x + intercept

hesapla = list(map(yontem, x))

plt.scatter(x, y, label="Gerçek Değerler")
plt.plot(x, hesapla, color="red", label="Regresyon Doğrusu")
plt.show()

# 2. Veriye 3. dereceden polinom uydurma (y = ax³ + bx² + cx + d)
polinom = np.poly1d(np.polyfit(x, y, 3))
print(polinom)