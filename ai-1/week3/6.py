import numpy as np
from scipy.interpolate import interp1d

# ======================================================================
# Aralık İçi ve Aralık Dışı Tahmin (İnterpolasyon ve Doğrusal Tahmin)
# ======================================================================

# Verilen veri seti üzerinden, veride olmayan bir x değeri için y karşılığını tahmin etmek için kullanılır.

x = [1, 2, 4]
y = [10, 17, 21]

# ====================================
# 1. Aralık İçi Tahmin (Interpolation)
# ====================================

f = interp1d(x, y)
sonuc_ici = f(4)  # x=4 için y karşılığını getirir
print("Aralık İçi Tahmin (x=4):", sonuc_ici)
# Çıktı: 21.0
# x veri setinde aynı indeksli bir değer olduğu için direkt olarak y karşılığı döndürülür.

# ==================================================
# 2. Aralık Dışı Tahmin (Lineer Regresyon / Polyfit)
# ==================================================

n, b = np.polyfit(x, y, 1)  # 1. derece denklem (y = n*x + b) katsayılarını çıkarır
print(f"Eğim (n): {n}")
print(f"Kesim Noktası (b): {b}")
# Çıktı:
# Eğim (n): 3.571428571428571,
# Kesim Noktası (b): 7.285714285714287

# Veri setinde olmayan x = 5 için tahmin yürütme
tahmin_x = 5
sonuc_disi = n * tahmin_x + b
print("Aralık Dışı Tahmin (x=5):", sonuc_disi)
# Çıktı: 25.14285714285714

# ==================================================
# Örnek Aralık Dışı Tahmin (x=6)
# ==================================================

l1 = [2, 6, 18]  # 3'ün katı şeklinde bir veri seti
l2 = [1, 3, 9]  # 3'ün karesi şeklinde bir veri seti

tahmin_x2 = 54 # (18 * 3)

# 3'ün katı şeklinde bir veri seti için tahmin
n1, b1 = np.polyfit(l1, l2, 1)
sonuc_disi2 = n1 * tahmin_x2 + b1
print("Aralık Dışı Tahmin (x=54):", sonuc_disi2)
# Çıktı: 27.0
