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
