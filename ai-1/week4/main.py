import pandas as pd
import numpy as np

# ==================================================
# 1. Veri Okuma ve iloc ile Dilimleme (Ön İşleme)
# ==================================================

veri = pd.read_csv("E:\\dev\\python-ai\\ai-1\\week4\\iris.csv")

# iloc Kulanım Kuralları
# veri.iloc[0,0]   --> 1. satır 1. sütun
# veri.iloc[:,0]   --> 1. sütunun tüm satırları
# veri.iloc[0,:]   --> 1. satırın tüm sütunları
# veri.iloc[:,:-1] --> Tüm satırlar ve son sütun hariç tüm sütunlar

ozellikler = veri.iloc[:, :-1]  # Bağımsız Değişkenler (Sepal/Petal boyutları) 4 sütun - X bağımsız değişkenler
hedef_sinif = veri.iloc[:, -1]  # Bağımlı Değişken (Çiçek türü) 1 sütun - Y bağımlı değişkenler

print(ozellikler.head())
# Çıktı: İlk 5 satır özet tablosu
print(ozellikler.shape)
# Çıktı: (150, 4) -> 150 örnek, 4 öznitelik

# ==================================================
# 2. Manuel Entropi Hesaplama
# ==================================================


def entropiHesap(veri_kumesi):
    # Veri kümesindeki sınıfları ve kaçar adet olduklarını bulur
    siniflar, sayilar = np.unique(veri_kumesi, return_counts=True)

    # Olasılık hesaplama (p = eleman sayısı / toplam sayı)
    olasiliklar = sayilar / len(veri_kumesi)

    # 1e-9: Olasılık sıfır olduğunda log2(0) tanımsızlık hatası vermesin diye eklenir
    entropi = -np.sum(olasiliklar * np.log2(olasiliklar + 1e-9))

    return entropi


# ==================================================
# 3. ID3 Algoritması (Bilgi Kazancı / Information Gain)
# ==================================================


def bilgiKazanciHesapla(ozellik_kolon, hedef, esik_degeri):
    # Ana düğümün (Parent) entropisi
    parent_entropi = entropiHesap(hedef)

    # Verinin belirlenen eşik değerine göre sol ve sağ dallara ayrılması
    sol = ozellik_kolon <= esik_degeri
    sag = ozellik_kolon >= esik_degeri

    toplam_veri_len = len(hedef)
    sol_boyut = np.sum(sol)
    sag_boyut = np.sum(sag)

    # Dallanma gerçekleşmediyse kazanç 0'dır
    if sol_boyut == 0 or sag_boyut == 0:
        return 0

    # Alt dalların ağırlıklı ortalama entropisi
    child_entropi = (sol_boyut / toplam_veri_len) * entropiHesap(hedef[sol]) + (sag_boyut / toplam_veri_len) * entropiHesap(hedef[sag])

    # Bilgi Kazancı = İlk Entropi - Bölünme Sonrası Entropi
    bilgi_kazanci = parent_entropi - child_entropi

    return bilgi_kazanci


# ==================================================
# 4. C4.5 Algoritması (Kazanç Oranı / Gain Ratio)
# ==================================================


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

    # Bölünme Bilgisi (Split Info) hesaplama
    bolunme_bilgisi = -(p_sol * np.log2(p_sol) + 1e-9) + (p_sag * np.log2(p_sag) + 1e-9)

    if bolunme_bilgisi == 0:
        return 0

    return kazanc / bolunme_bilgisi


# ==================================================
# 5. Modelin Çalıştırılması ve Örnek Test
# ==================================================

# Iris veri setinin 3. sütununu (petal_length) bağımsız değişken olarak seçiyoruz
ornek_ozellik = ozellikler.iloc[:, 2]
ornek_esik_degeri = 2.5  # Dallanma sınırı

id3 = bilgiKazanciHesapla(ornek_ozellik, hedef_sinif, ornek_esik_degeri)
c45 = kazanc_orani_hesapla(ornek_ozellik, hedef_sinif, ornek_esik_degeri)

print("ID3 Bilgi Kazancı:", id3)
# Çıktı: ~0.918 (Veriyi 2.5 sınırından böldüğümüzde yüksek bilgi kazancı elde ederiz)

print("C4.5 Kazanç Oranı:", c45)
# Çıktı: ~0.987 (Bölünme bilgisi ile düzeltilmiş karar skoru)
