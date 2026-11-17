import pandas as pd
import numpy as np
import statistics as stat
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================================================================
# ===================================================================

a = [1, 2, 3]
print(stat.mean(a))

sales = np.random.randint(10, 50, size=(4, 3))
salesDataFrame = pd.DataFrame(sales)
firstDayMean = stat.mean(sales[0])

rdList = np.linspace(0, 10, 100)

# ===================================================================
# Normalizations - IQR, Z-SCOR, LOG, L2
# ===================================================================

clicks = np.array(([5, 12, 15, 18, 22, 500]))

q75, q25 = np.percentile(clicks, [75, 25])
aykiri = q75 - q25
medyan = np.median(clicks)
x = (clicks - medyan) / aykiri
# [-1.39393939 -0.54545455 -0.18181818  0.18181818  0.66666667 58.60606061]

z = (clicks - np.mean(clicks)) / np.std(clicks)
# [-0.49894552 -0.46028185 -0.4437117  -0.42714155 -0.40504803  2.23512865]

log = np.log1p(clicks)
# [1.79175947 2.56494936 2.77258872 2.94443898 3.13549422 6.2166061 ]

l2 = np.linalg.norm(clicks, keepdims=True)
normal = clicks / l2
# [0.00997605 0.02394251 0.02992814 0.03591377 0.0438946  0.99760463]

# ===================================================================
# Errors MAE, RMSE, MAPE, R2
# ===================================================================

gercek = [200, 350, 500, 150, 800]
tahmin = [210, 330, 480, 160, 750]

# MAE - Mean Absolute Error
mae = mean_absolute_error(gercek, tahmin)
# 22.0

# MSE - Mean Squared Error
rmse = np.sqrt(mean_squared_error(gercek, tahmin))
# 26.457513110645905

# MAPE - Mean Absolute Percentage Error
gercek = np.array(gercek)
tahmin = np.array(tahmin)
hata_orani = np.abs((gercek - tahmin) / gercek)
mape = np.mean(hata_orani) * 100
# 5.52 - %5.52

# R2 - R Squared
r2 = r2_score(gercek, tahmin)
# 0.9872727272727273

# ===================================================================
# Matris Operations & Lineer
# ===================================================================

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

matris_carpimi = x @ y
matris_transpozu = x.T  # Transpoz, bir matrisin satır ve sütunlarının yer değiştirmesidir
matris_determinant = np.linalg.det(x)  # Tersi var mı yok mu. Sıfır ise tersi yoktur.
matris_tersi = np.linalg.inv(x)  # Ters matris / Bölme işlemi.

# Linear Denklem Sistemi Çözümü (Ax = b)
# 2x + 1y = 5
# 1x + 3y = 6
a = np.array([[2, 1], [1, 3]])
b = np.array([5, 6])
denklem_cozumu = np.linalg.solve(a, b)
# Çıktı: [1.8 1.4] - (x = 1.8, y = 1.4)

# ===================================================================
# T Test
# ===================================================================

import scipy.stats as sa

a = [10, 12, 11, 13, 12]
b = [20, 21, 22, 23, 24]

# İki Bağımsız Örneklem T-Testi
sonuc = sa.ttest_ind(a, b)

print("t değeri:", sonuc.statistic)
print("p değeri:", sonuc.pvalue)
# Çıktı:
# t değeri: -12.649110640673518
# p değeri: 4.887823377545672e-06  (Yani 0.00000488...)

# ===================================================================
# Denklem Minimizasyonu / Optimizasyon
# ===================================================================

from scipy.optimize import minimize_scalar


def f(x):
    return x**2 - 4 * x + 5


sonuc = minimize_scalar(f)
# Çıktı:
# fun: 1.0  ---> Fonksiyonun ulaşabildiği MINIMUM y değeri (En az hata)
#   x: 2.0  ---> Fonksiyonu minimum yapan x değeri (En ideal ayar)

# ===================================================================
# Belirli İntegral Hesaplama
# ===================================================================

from scipy.integrate import quad


def f(x):
    return x**2


# 0 alt limit, 2 üst limit: ∫₀² x² dx
sonuc, hata = quad(f, 0, 2)
# Çıktı:
# sonuc: 2.666666666666667  (Yani 8/3)
# hata:  2.960594732333751e-14 (Hesaplama hassasiyeti)

# ===================================================================
# Tepe Noktası Bulma
# ===================================================================

from scipy.signal import find_peaks
import matplotlib.pyplot as plt
   
veri = [1, 2, 3, 4, 15, 2, 4, 1]

tepe_indeksleri, bilgiler = find_peaks(veri)
# Çıktı: [4 6]

plt.plot(veri)
plt.plot(tepe_indeksleri, np.array(veri)[tepe_indeksleri], "x")
plt.show()

# ===================================================================
# Aralık İçi Tahmin - İnterpolasyon
# ===================================================================

from scipy.interpolate import interp1d

x = [1, 2, 4]
y = [10, 17, 21]

f = interp1d(x, y)

# Bilinen bir nokta (x=4) veya aradaki eksik bir nokta (örneğin x=3) için tahmin:
print("x=4 için:", f(4))  # Çıktı: 21.0
print("x=3 için:", f(3))  # Çıktı: 19.0 (17 ile 21'in tam ortasını doğrusal olarak birleştirir)

# ===================================================================
# Aralık Dışı Tahmin - DoğrusalTahmin
# ===================================================================

import numpy as np

x = [1, 2, 4]
y = [10, 17, 21]

# 1. derece doğru denklemi (y = n*x + b) katsayılarını bulma
n, b = np.polyfit(x, y, 1)
# Çıktı: Eğim (n): 3.57, Kesim Noktası (b): 7.29

# Veri setinin dışındaki x = 5 için gelecek tahmini
tahmin_x = 5
sonuc_disi = n * tahmin_x + b
# Çıktı: 25.14