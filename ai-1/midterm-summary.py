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

# ===================================================================
# Entropi - ID3 - C4.5
# ===================================================================

veri = pd.read_csv("E:\\dev\\python-ai\\ai-1\\week4\\iris.csv")

ozellikler = veri.iloc[:, :-1]
hedef_sinif = veri.iloc[:, -1]


def entropiHesap(veri_kumesi):
    # counts = Hangi veriden kaç adet var
    _, counts = np.unique(veri_kumesi, return_counts=True)

    # adet / tüm veri = olasılık
    olasiliklar = counts / len(veri_kumesi)

    entropi = -np.sum(olasiliklar * np.log2(olasiliklar + 1e-9))
    print(entropi)
    return entropi


# ID3
def bilgiKazanciHesapla(ozellik_kolon, hedef, esik_degeri):
    entropi = entropiHesap(hedef)

    # Taç yaprak uzunluğu <= 2.5 cm mi?
    sol = ozellik_kolon <= esik_degeri  # Koşulu sağlayanlar - Sol dal
    sag = ozellik_kolon >= esik_degeri  # Koşulu Sağlamayanlar - Sağ da

    toplam_veri_len = len(hedef)
    sol_boyut = np.sum(sol)
    sag_boyut = np.sum(sag)

    # Dallanma olmadıysa yani tüm veri tek taraftaysa dallanma başarısızdır.
    if sol_boyut == 0 or sag_boyut == 0:
        return 0

    # Alt dalların ağırlık ortalamaları
    alt_entropi = (sol_boyut / toplam_veri_len) * entropiHesap(hedef[sol]) + (sag_boyut / toplam_veri_len) * entropiHesap(hedef[sag])

    # Kazanç
    return entropi - alt_entropi


# C4.5
def kazanc_orani_hesapla(ozellik_kolon, hedef, esik_degeri):
    kazanc = bilgiKazanciHesapla(ozellik_kolon, hedef, esik_degeri)

    sol = ozellik_kolon <= esik_degeri
    sag = ozellik_kolon >= esik_degeri

    toplam_veri = len(hedef)
    sol_boyut = np.sum(sol)
    sag_boyut = np.sum(sag)

    if sol_boyut == 0 or sag_boyut == 0:
        return 0

    p_sol = sol_boyut / toplam_veri
    p_sag = sag_boyut / toplam_veri

    bolunme_bilgisi = -((p_sol * np.log2(p_sol + 1e-9)) + p_sag * np.log2(p_sag + 1e-9))

    if bolunme_bilgisi == 0:
        return 0

    return kazanc / bolunme_bilgisi


# Iris veri setinin 3. sütununu (petal_length) bağımsız değişken olarak seçiyoruz
ornek_ozellik = ozellikler.iloc[:, 2]
ornek_esik_degeri = 2.5  # Dallanma sınırı

id3 = bilgiKazanciHesapla(ornek_ozellik, hedef_sinif, ornek_esik_degeri)
# Çıktı: ~0.918 (Veriyi 2.5 sınırından böldüğümüzde yüksek bilgi kazancı elde ederiz)

c45 = kazanc_orani_hesapla(ornek_ozellik, hedef_sinif, ornek_esik_degeri)
# Çıktı: ~0.987 (Bölünme bilgisi ile düzeltilmiş karar skoru)

# ===================================================================
# Veri Temizleme ve Manuel Doğrusal Regrasyon
# Dolar verisi ile gelecek dolar fiyatlarını tahmin etme
# ===================================================================

df = pd.read_csv("E:\\dev\\python-ai\\ai-1\\week5\\dolar.csv")

for sutun in ("Şimdi", "Açılış", "Yüksek", "Düşük"):
    df[sutun] = df[sutun].astype(str).str.replace(".", "").str.replace(",", ".").astype(float)

df["Tarih"] = pd.to_datetime(df["Tarih"], dayfirst=True)
df = df.sort_values("Tarih").reset_index(drop=True)

# İlk günden itibaren geçen gün sayısı: [0, 1, 2, 5, ...]
df["Gün"] = (df["Tarih"] - df["Tarih"].min()).dt.days

x = df["Gün"].to_numpy(dtype=float)
y = df["Şimdi"].to_numpy(dtype=float)

# Veriyi sırayı bozmadan %80 Eğitim, %20 Test olarak bölme
egitimsayisi = int(len(x) * 0.8)

x_train, y_train = x[:egitimsayisi], y[:egitimsayisi]  # Geçmiş %80 (Öğrenme)
x_test, y_test = x[egitimsayisi:], y[egitimsayisi:]  # Gelecek %20 (Sınav)

n = len(x_train)
toplamx = np.sum(x_train)  # ∑x
toplamy = np.sum(y_train)  # ∑y
toplamxy = np.sum(x_train * y_train)  # ∑(x * y)
toplamx2 = np.sum(x_train**2)  # ∑(x²)

a = (n * toplamxy - toplamx * toplamy) / (n * toplamx2 - toplamx**2)
b = (toplamy - a * toplamx) / n

# Test verisi üzerinde tahmin üretme
y_tahmin = a * x_test + b
# [45.93161646 45.95127034 46.01023201 46.02988589 46.04953978 46.06919367
#  46.08884756 46.14780922 46.1674631  46.18711699 46.20677088 46.22642477
# ...... 48.38835238, 48.40800627 48.42766016]

# ===================================================================
# Hiyerarşik Kümeleme ve Dendrogram (Gözetimsiz Öğrenme)
# ===================================================================

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage

iris = load_iris()

x = iris.data[:, 0] # 1. sütun (sepal_length)
y = iris.data[:, 1] # 2. sütun (sepal_width)

plt.scatter(x, y)

# 1. Hiyerarşik Kümeleme Matrisi (Linkage)
data = iris.data
linkage_data = linkage(data, method="ward", metric="euclidean")

# 2. Dendrogram (Ağaç Şeması) Çizimi
plt.figure(figsize=(12,6))
dendrogram(linkage_data)

plt.title("Kümeleme Grafiği")
plt.xlabel("Çiçek Örnekleri")
plt.ylabel("Küme Mesafeleri")
plt.show()

# ===================================================================
# İstatistiksel Doğrusal Regresyon (linregress) ve Polinoma Geçiş
# ===================================================================

# 2. Veriye 3. dereceden polinom uydurma (y = ax³ + bx² + cx + d)
polinom = np.poly1d(np.polyfit(x, y, 3))

x = [1, 4, 5, 6, 9, 12, 18, 25, 33, 42, 55, 65]
y = [11, 31, 44, 55, 64, 76, 87, 91, 100, 123, 136, 145]

# 1. Tek satırda Doğrusal Regresyon (y = slope * x + intercept)
slope, intercept, r, p, std_err = stat.linregress(x, y)
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