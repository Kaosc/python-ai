# AI-1 Code

# week2

## week2/main.py

```python
import pandas as pd
import numpy as np
import statistics as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =======================================
# PAGE 1
# =======================================
print("\n=======================================\n")

a = [1, 2, 3]
st.mean(a)  # 2.0
print("Mean: ", st.mean(a))  # 2.0

a = np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9]))
dataFrame = pd.DataFrame(a)
print("DataFrame: ", dataFrame)


matris = np.random.randint(1, 10, size=(3, 4))  # 3 satır 4 sütun
print("Matris: ", matris)

# =======================================
# PAGE 2
# =======================================
print("\n=======================================\n")


list = np.array([1, 34, 45, 65, 1000, 7, 1])

# Aykırı veri: 1000.
# Aykırı veri temizleme

# İki çeyrek arası (Çeyrek (Q)). Dizi ilk paramtere. q -> aykırı değer yapısı
q75, q25 = np.percentile(list, [75, 25])
aykiri = q75 - q25
medyan = np.median(list)

print("Medyan", medyan)  # 34

# Dar aralığa çekme - normalizasyon

# Standart Normalizasyon - Medyan
x = (list - medyan) / aykiri
print("Normalize:", x)  # [-0.64705882, 0. , 0.21568627, 0.60784314, 18.94117647, -0.52941176, -0.64705882]

# Z skor ile normalizasyon
zskor = (list - np.mean(list)) / np.std(list)
print("Z skor: ", zskor)

l2 = np.linalg.norm(list, keepdims=True)
normal = list / l2
print("Normal: ", normal)

# Verilerin logaritması
right = np.log1p(list)
print("Right: ", right)

# =======================================
# PAGE 3
# =======================================
print("\n=======================================\n")


# -5 ile 5 arasında 100 tane sayı (float)
list = np.linspace(-5, 5, 50)
print("List: ", list)

# =======================================
# PAGE 4
# =======================================
print("\n=======================================\n")

# Hata payı bulma
import pandas as pd
import numpy as np
import statistics as st

gercek = [12, 123, 34, 56, 7, 88, 9] # Veri seti
tahmin = [11, 110, 25, 58, 9, 92, 11] # Tahmin edilen değerler (modelin çıktısı)

# 1 MAE hesaplama
mae = mean_absolute_error(gercek, tahmin)
print("mae: ", mae)  # 4.714285714285714 = %4

# R2 hesap
r2 = r2_score(gercek, tahmin)
print("r2: ", r2)  # 0.9767034068136272 = %97

# =======================================
# !! below codes auto generated !!
# =======================================

# RMSE hesap
rmse = np.sqrt(mean_squared_error(gercek, tahmin))
print("rmse: ", rmse)  # 7.0710678118654755 = %7

# MAPE hesap
mape = np.mean(np.abs((np.array(gercek) - np.array(tahmin)) / np.array(gercek))) * 100
print("mape: ", mape)  # 7.071067811865475
```

# week3

## week3/1.py

```python
import pandas as pd
import numpy as np 

import scipy.linalg as la
import scipy.stats as sa

# Matris tanımlama. (Kare olmak zorunda, 2x2, 3x3 etc.)

matris = [
    [1,2,5],
    [3,4,6],
    [6,3,7]
  ]

# Matris determinat hesaplama.
res = la.det(matris)

# Matrisin satırlarını sütunlara çeviri. Tersini alma.
res2 = la.inv(matris)

print(res)


# örnek denklemi matrise çevirerek çözüm

# 2x + y = 5 --> y = 5 - 2x
# x + 3y = 6 --> y = (6 - x) / 2

# Katsayılar matrisi x değişken matrisi = y

# Katsayılar matirisi
a = [[2,1], [1,3]] 

# Değişkenler matrisi
b = [5, 6]

# Katsayılar ve değiken matrisi çarpımı bulma
# - Birinci matrisin satırı ile 2. matrisin sütunu eşit değilse çarpma yapılamaz
res3 = la.solve(a, b) # // [1.8, 1.4] 1.si= X, 2.si = Y
print(res3)


```

## week3/2.py

```python
import pandas as pd
import numpy as np 

import scipy.linalg as la
import scipy.stats as sa

a = [10, 12, 11, 13, 12]
b = [20, 21, 22, 23, 24]

# T testi yapma. İki veri arasındaki değerler
sonuc = sa.ttest_ind(a, b)

# 1.si t değeri, 2.si p değeri (anlamlılık oranı)
print("t değeri: ", sonuc.statistic)
print("p değeri: ", sonuc.pvalue)


```

## week3/3.py

```python
import pandas as pd
import numpy as np 

import scipy.linalg as la
import scipy.stats as sa

from scipy.optimize import minimize_scalar

# Denklem optimizasyonu ile denklem minumum değeri bulma.

def f(x) :
    return x**2-4*x+5

sonuc = minimize_scalar(f)

print(sonuc)

"""
 message: 
          Optimization terminated successfully;
          The returned value satisfies the termination criteria
          (using xtol = 1.48e-08 )
 success: True
     fun: 1.0 <---- Minimum fonksiyon değeri
       x: 2.0 <---- minimum x değeri
     nit: 4
    nfev: 8
"""
```

## week3/4.py

```python
import pandas as pd
import numpy as np 

import scipy.linalg as la
import scipy.stats as sa

from scipy.optimize import minimize_scalar
from scipy.integrate import quad # intergral

# Denklem optimizasyonu ile denklem minumum değeri bulma.

def f(x) :
    return x**2

# 0 alt limit, 2 üst limit.
sonuc = quad(f, 0, 2)

print(sonuc)

"""
1. değer integral değeri
2. değer hata oranı (hata değeri neredeyse 0)
(2.666666666666667, 2.960594732333751e-14)
"""
```

## week3/5.py

```python
import pandas as pd
import numpy as np 

import scipy.linalg as la
import scipy.stats as sa

from scipy.signal import find_peaks
from scipy.integrate import quad # intergral
import matplotlib.pyplot as plt 

# Denklem optimizasyonu ile denklem minumum değeri bulma.

veri = [1,2,3,4,15,2,4,1]

tepe, bilgiler = find_peaks(veri)

print(tepe) # [4 6]

# Verideki uzunluk değeri
x = range(len(veri))

plt.plot(x, veri, marker="o", label="veriler")

plt.show()
```

## week3/6.py

```python
import pandas as pd
import numpy as np 

import scipy.linalg as la
import scipy.stats as sa

from scipy.signal import find_peaks
from scipy.integrate import quad # intergral
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d

# Denklem optimizasyonu ile denklem minumum değeri bulma.

x = [1,2,4]
y = [10, 17, 21]

# Tahmin (Bir aralık içi tahmin)
f = interp1d(x, y)

# sonuc = f(1) --> 10
# sonuc = f(2) --> 17
sonuc = f(4) # --> 21.0

print(sonuc) # 21.0

# Aralık dışı tahmin işlemi

# Doğrusallık oluşturarak olmayan bir x verisini tahmin etme 
# örneğin 5 verisi x de yok dolayısı ile karşılığı yok.
# 3. parametre denklem parametresi (1. derece, 2. derece)
# n = x'in katsayısı, b = hata payı
# 1. derece bir denklem dönderir.
# doğrusal regresyon

n, b = np.polyfit(x, y, 1)

# tahmin etmek istediğimiz değer
tahmin = 5

sonuc2 = n * tahmin + b

print(sonuc2) # 25.14285714285714
```

## week3/7.py

```python
import pandas as pd
import numpy as np 

# 2 Matrisin çarpımı

x = np.array([[1,2], [3,4]])
y = np.array([[5,6], [7,8]])

# @ işareti matrislerin çarpımı için
c = x @ y

print(c)

"""
[[19 22]
 [43 50]]
"""

# Matrisi ters çevirme (satırla sütun yer değiştirir)

ters = x.T
print(ters)

"""
[[1 3]
 [2 4]]
"""

# 

inverse = np.la.inv
```

# week4

## week4/main.py

```python
import pandas as pd
import numpy as np

veri = pd.read_csv("E:\\dev\\python-ai\\ai-1\\week4\\iris.csv")

# ==================================================
# Veri önişleme
# ==================================================

# veri setinde bağımlı ve bağımsız değişkenleri birbirinden
# ayırmak gerekir. iloc bu amaçla kullanılır.

# veri.iloc[:] --> ?
# veri.iloc[0,0] --> 1. satır 1. sütun
# veri.iloc[::] --> Tüm satır ve sütunlar (genelde ilk yapılan işlem)
# veri.iloc[:,0] -->

# a = veri.iloc[:,0] --> 1. Sütunun tüm satırları
# a = veri.iloc[0,:] --> 1. Satırın tüm sütunları

# x = veri.iloc[:,:-1] --> Tüm satırlar ve Son sütun hariç tüm sütunlar (arr.length -1)

x = veri.iloc[:, :-1]  # --> Bağımlı değişkenler
y = veri.iloc[:, -1]  # --> Son sütunu al (bağımsız değişken)

print(x.head())  # --> İlk 5 değerin özetini verir
"""
   sepal_length  sepal_width  petal_length  petal_width
0           5.1          3.5           1.4          0.2
1           4.9          3.0           1.4          0.2
2           4.7          3.2           1.3          0.2
3           4.6          3.1           1.5          0.2
4           5.0          3.6           1.4          0.2
"""

print(x.shape)  # --> Satır, sütun sayılarını verir # (150, 4)

# ==================================================
# Sınıflandırma - Karar ağaçları modelleri uygulama | C4.5
# ==================================================

# Özellikleri ayırma
ozellikler = veri.iloc[:, :-1]

# Bağımsız değişkeni ayırma
hedef_sinif = veri.iloc[:, -1]

# entropi hesaplama


def entropiHesap(veri_kumesi):
    siniflar, sayilar = np.unique(veri_kumesi, return_counts=True)

    olasiliklar = sayilar / len(veri_kumesi)
    # 1e-9 ==> bazi sayilar sıfır değer alıp sonsuz değere girebilir,
    # Bunun için çok küçük bir değer eklenir.
    entropi = -np.sum(olasiliklar * np.log2(olasiliklar + 1e-9))

    return entropi


def bilgiKazanciHesapla(ozellik_kolon, hedef, esik_degeri):

    # Ana entropiyi hesapla
    parent_entropi = entropiHesap(hedef)

    # Karar ağacının dallandırılması
    sol = ozellik_kolon <= esik_degeri  # sol dallanma
    sag = ozellik_kolon >= esik_degeri  # sag dallanma

    toplam_veri_len = len(hedef)

    sol_boyut = np.sum(sol)
    sag_boyut = np.sum(sag)

    if sol_boyut == 0 or sag_boyut == 0:  # Dallanmamış veri
        return 0

    # Alt entropi hesaplama
    child_entropi = (sol_boyut / toplam_veri_len) * entropiHesap(hedef[sol]) + (sag_boyut / toplam_veri_len) * entropiHesap(hedef[sag])

    bilgi_kazanci = parent_entropi - child_entropi

    return bilgi_kazanci


def kazanc_orani_hesapla(ozellik_kolon, hedef, esik_degeri):

    kazanc = bilgiKazanciHesapla(ozellik_kolon, hedef, esik_degeri)

    sol = ozellik_kolon <= esik_degeri  # sol dallanma
    sag = ozellik_kolon >= esik_degeri  # sag dallanma

    toplam_veri = len(hedef)

    sol_boyut = np.sum(sol)
    sag_boyut = np.sum(sag)

    if sol_boyut == 0 or sag_boyut == 0:  # Dallanmamış veri
        return 0

    p_sol = sol_boyut / toplam_veri
    p_sag = sag_boyut / toplam_veri

    bolunme_bilgisi = -(p_sol * np.log2(p_sol) + 1e-9) + (p_sag * np.log2(p_sag) + 1e-9)

    if bolunme_bilgisi == 0:
        return 0

    return kazanc / bolunme_bilgisi


ornek_ozellik = ozellikler.iloc[:, 2]
ornek_esik_degeri = 2.5  # Kendimiz belirliyoruz

id3 = bilgiKazanciHesapla(ornek_ozellik, hedef_sinif, ornek_esik_degeri)
c45 = kazanc_orani_hesapla(ornek_ozellik, hedef_sinif, ornek_esik_degeri)
```

# week5

## week5/1.py

```python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:00:19 2026

@author: Lenovo
"""

import pandas as pd
import numpy as np

df = pd.read_excel("dolar.xlsx")

#=====================================
# Data dönüşümleri
#=====================================

for sutun in ("Şimdi", "Açılış", "Yüksek", "Düşük"):
    df[sutun] = (
        df[sutun]
        df.astype(str)
        .str.replace(".","",regex=False)
        .str.replace(",",".", regex=False)
        .astype(float)
        ) 
    
df["Tarih"] = pd.to_datetime(
    df["Tarih"],
    dayfirst=True
    )

df = df.sort_values("Tarih").reset_index(drop=True)

#=====================================
# Eğitim
#=====================================

df["Gün"] = (
    df["Tarih"] - df["Tarih"].min()   
    ).dt.days

# Bağımsız Değişken
x = df["Gün"].to_numpy(dt.type=float)

# Bağımlı Değişken
y =  df["Şimdi"].to_numpy(dt.type=float)

egitimsayisi = int( len(x) * 0.8)

x_train = x[:egitimsayisi]
y_train = y[:egitimsayisi]

# Test
x_test = x[egitimsayisi:]
y_text = y[egitimsayisi:]

#=====================================
# Doğrusal Regrasyon Modeli
#
# Açılış, yüksek, düşük = bağımsız değişkenler
#=====================================

toplamx = np.sum(x_train)
toplamy = np.sum(y_train)

toplamxy = np.sum(x_train * y_train)
toplamx2 = np.sum(x.train **2)

n = len(x_train)
a = (n*toplamxy-toplamx*toplamy) / (n*toplamx2-toplamx**2)
b = (toplamy - a *toplamx) / n




```

# week7

## week7/1.py

```python
"""
GRAFİK ÇİZME
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Kümeleme işlemi için
from scipy.cluster.hierarchy import dendrogram, linkage

iris = load_iris()

x = iris.data[:, 0]  # İlk sütundaki tüm elemanlar
y = iris.data[:, 1]

# Grafik

plt.scatter(x, y)
# plt.show()

# Verilerin nerede yoğunlaştığını ve ortak merkez noktalarının nerede
# bulunduğunu görmek için hiyerarşik kümeleme grafiği oluşturulur.
data = iris.data
linkage_data = linkage(data, method="ward", metric="euclidean")
# Grafiği çiz
plt.figure(figsize=(12, 6))
dendrogram(linkage_data)

plt.title("Kümeleme Grafiği")
plt.xlabel("Çiçek Örnekleri")
plt.ylabel("Küme Mesafeleri")
plt.show()
```

## week7/2.py

```python
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
```

## week7/3.py

```python
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
```

## week7/4.py

```python
"""
ÖRNEK: KNN SINIFLANDIRMASI
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

from sklearn.metrics import confusion_matrix  # Regresyon analizi için

iris = load_iris()

x = iris.data  # Tüm bağımsız değişkenler
y = iris.target  # Sınıf etiketleri

print("Bağımsız değişkenler:", iris.data)
print("Sınıflar:", iris.target)

# Bağımlı ve bağımsız değişkenleri eğitim ve test kümelerine ayır
from sklearn.model_selection import train_test_split

# Verinin %20'sini test için ayır. random_state, rastgelelik değeridir.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=17)

# Modeli oluştur

model = KNeighborsClassifier()

# Modeli eğit

model.fit(x_train, y_train)

# Tahmin
y_tahmin = model.predict(x_test)

# Doğruluk oranını değerlendirmek için hata matrisi oluştur
hata_matrisi = confusion_matrix(y_test, y_tahmin)

print(hata_matrisi)

"""
# Sınıf 1, sınıf 2 ve sınıf 3 için örnek hata matrisi:
[[ 7       0        0]   s1 # Sıfır olmayan değerler yanlış sınıflandırmayı gösterir.
 [ 0       11       0]   s2
 [ 0       2        10]]  s3 # Örneğin 2 yanlış, 10 doğru sınıflandırmayı gösterir.

Hata oluşursa modelin değiştirilmesi gerekebilir. Bunun nedeni verilerin
birbirine çok yakın olması veya verilerde hata bulunması olabilir.

"""

import seaborn as sea

# Hata matrisini düzenlemek ve sınıf adlarını eklemek için kullanılır.

index = ["setosa", "vericolor", "virginica"]
cols = ["setosa", "vericolor", "virginica"]

hata_goster = pd.DataFrame(hata_matrisi, cols, index)

plt.figure(figsize=(10, 6))

sea.heatmap(hata_goster)
```

# week8

## week8/1.py

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:,0]>0, X_xor[:,1]>1)

y_xor = np.where(y_xor,1,0) #elimizde ki değerleri sayısal sınıflara dönüştürür
plt.scatter(
    X_xor[y_xor == 1,0],
    X_xor[y_xor == 1,1],
    c = "royalblue",
    marker = "s",
    label = "sınıf 1",
    edgecolor = "black"
    
    )

plt.scatter(
    X_xor[y_xor == 0,0],
    X_xor[y_xor == 0,1],
    c = "royalblue",
    marker = "s",
    label = "sınıf 2",
    edgecolor = "black"
    
    )

plt.xlim([-3,3])
plt.ylim([-3,3])
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")
plt.legend(loc = "best")
plt.tight_layout()
plt.show()
```

## week8/2.py

```python
# svm

import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import datasets
from matplotlib.colors import ListedColormap

# veri seti yükle
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# test size = 0.2
# train size = 0.8

# Ağı eğitmek içi gerekli olan standartitasyon işelmerli
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

#

# eğitim verilerini dikeyde/sütunda birleştir.
x_combined_std = np.vstack(x_train_std, x_test_std)

# yatayda
y_combined_Std = np.hstack(y_train, y_test)

# karar bölgelerini çizen fonksyion


def karar(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "grav", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(x1_min, x1_max, resolution), np.meshgrid(x2_min, x2_max, resolution)

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # sondaki t [x1, x2] şeklindeki listeyi
    # [x1,
    # x2,]
    # şekline çeviri.
    Z = Z.reshape(xx1.shape)


# model # rbf = gauss yöntem # lineer = doğrusal,
svm = SVC(kernel="linear", random_state=1)

# Modelin test aşamasında kullanılacak standart aşamasını gösterir
svm.fit(x_train_std, y_train)
```

