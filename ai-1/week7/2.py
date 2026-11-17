"""
ÖRNEK: Lineer Regresyon
İstatistiksel Lineer Regresyon ve Polinoma Geçiş
"""

# Tek değişkenli doğrusal regresyon katsayılarını scipy.stats ile tek satırda bulmayı ve
# verinin düz bir çizgiye uymadığı durumlarda polinom denklemine geçmeyi gösterir.

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = [1, 4, 5, 6, 9, 12, 18, 25, 33, 42, 55, 65]
y = [11, 31, 44, 55, 64, 76, 87, 91, 100, 123, 136, 145]

# Doğrusal regresyon denklemi (y = slope * x + intercept)
slope, intercept, r, p, std_err = stats.linregress(x, y)
# Örnek Çıktı: slope ≈ 1.88 (Eğim), intercept ≈ 32.16 (Kesim noktası)


def yontem(x):
    return slope * x + intercept


hesapla = list(map(yontem, x))

plt.scatter(x, y, label="Gerçek Değerler")
plt.plot(x, hesapla, color="red", label="Regresyon Doğrusu")
plt.show()

# Veriye 3. dereceden polinom uydurma
polinom = np.poly1d(np.polyfit(x, y, 3))
