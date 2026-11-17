"""
Sentetik XOR Verisi Üretme ve Görselleştirme
"""

# Doğrusal olarak tek bir çizgiyle ayrılamayan veri dağılımlarını (XOR mantık kapısı problemini)
# göstermek için rastgele sayılarla yapay bir veri seti üretilir ve sınıflara ayrılır.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# 200 satır, 2 sütundan oluşan normal dağılımlı rastgele veri matrisi
X_xor = np.random.randn(200, 2)
# Örnek Çıktı (X_xor ilk 2 satır):
# [[ 1.62434536 -0.61175641]
#  [-0.52817175 -1.07296862]]

# XOR mantıksal koşulu: İki koşuldan yalnızca biri True ise True döner
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 1)
# Çıktı: [True, False, False, ...]

# Boolean (True/False) değerlerini sayısal sınıflara (1 ve 0) dönüştürme
y_xor = np.where(y_xor, 1, 0)
# Çıktı: [1, 0, 0, ...]

# Sınıf 1 noktalarını çizdirme
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c="royalblue", marker="s", label="sınıf 1", edgecolor="black")

# Sınıf 0 noktalarını çizdirme
plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], c="royalblue", marker="s", label="sınıf 2", edgecolor="black")

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
