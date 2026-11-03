"""
ÖRNEK: Lineer Regresyon
"""

import numpy as np
import matplotlib.pyplot as plt

# Kümeleme işlemi için
from scipy import stats

# Doğrusal regresyon denklemi: ax + b
# a için slope, b için intercept kullanılır.

x = [1, 4, 5, 6, 9, 12, 18, 25, 33, 42, 55, 65]
y = [11, 31, 44, 55, 64, 76, 87, 91, 100, 123, 136, 145]

# -0.3,   19
slope, intercept, r, p, std_err = stats.linregress(x, y)


def yontem(x):
    return slope * x + intercept


hesapla = list(map(yontem, x))

plt.scatter(x, y, label="Gerçek Değerler")

# Regresyon doğrusunu göster
# Noktalar doğruya çok uzaksa lineer denklem bu veriye uygun değildir.
# Bu durumda tahmin için başka bir yöntem kullanılmalıdır.
plt.plot(x, hesapla, color="red", label="Regrasyon Doğrusu")


plt.show()

"""
ÖRNEK: Polinomsal Regresyon
"""

# Dereceyi biz belirliyoruz; burada derece 3 olarak seçilmiştir.
polinom = np.poly1d(np.polyfit(x, y, 3))
