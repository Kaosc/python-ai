from scipy.optimize import minimize_scalar

# ==============================================
# 1. Denklem Optimizasyonu
# ==============================================

# Yapay zeka modelleri eğitilirken Kayıp Fonksiyonu (Loss Function) minimum yapılmaya çalışılır.
# Modelin en az hatayı yaptığı parametreyi bulmak için kullanılır.

def f(x):
    return x**2 - 4*x + 5

sonuc = minimize_scalar(f)
print("Optimizasyon Sonucu:\n", sonuc)
# Çıktı:
#      fun: 1.0  <--- Fonksiyonun ulaşabildiği MINIMUM y değeri
#        x: 2.0  <--- Fonksiyonu minimum yapan x değeri

# ==============================================
# 2. İntegral Hesaplama
# ==============================================

# Sürekli olasılık dağılımlarının altında kalan alanı bulmak veya sınıflandırma modellerinin başarısını ölçen 
# ROC-AUC (Eğri Altında Kalan Alan) skorunu hesaplamak için tercih edilir

from scipy.integrate import quad

def f(x):
    return x**2

# 0 alt limit, 2 üst limit: ∫₀² x² dx

sonuc, hata = quad(f, 0, 2)
print("İntegral Değeri:", sonuc)
print("Tahmini Hata:", hata)
# Çıktı:
# İntegral Değeri: 2.666666666666667  (yani 8/3)
# Tahmini Hata: 2.960594732333751e-14

# ==============================================
# 3. Tepe Noktası (Peak) Bulma
# ==============================================

# Zaman serisi analizi, borsa verileri, ses analizi veya EKG sinyal verilerinde 
# anlık sıçrama veya yerel tepe noktalarını otomatik bulmak için find_peaks kullanılır.

from scipy.signal import find_peaks
import matplotlib.pyplot as plt

veri = [1, 2, 3, 4, 15, 2, 4, 1]

tepe_indeksleri, bilgiler = find_peaks(veri)
print("Tepe Noktası İndeksleri:", tepe_indeksleri)
# Çıktı: [4 6]  (4. indeksteki '15' ve 6. indeksteki '4' tepe noktalarıdır)

plt.plot(range(len(veri)), veri, marker="o", label="veriler")
plt.show()