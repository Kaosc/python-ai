import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent / "dolar.csv")

# Örnek Ham Veri (dolar.xlsx içi):
# Tarih         Şimdi     Açılış    Yüksek    Düşük
# 20.10.2026    "34,25"   "34,10"   "34,30"   "34,05"
# 19.10.2026    "34,10"   "34,00"   "34,15"   "33,95"

# 1. Metin formatındaki fiyatları ondalıklı sayıya (float) çevirme
for sutun in ("Şimdi", "Açılış", "Yüksek", "Düşük"):
    df[sutun] = (
        df[sutun]
        .astype(str)
        .str.replace(".", "", regex=False)  # Binlik noktasını sil ("1.250,50" -> "1250,50")
        .str.replace(",", ".", regex=False)  # Virgülü noktaya çevir ("1250,50" -> "1250.50")
        .astype(float)  # Sayı tipine dönüştür (1250.5)
    )

# 2. Metin formatındaki tarihi gerçek Tarih (Datetime) objesine çevirme
df["Tarih"] = pd.to_datetime(df["Tarih"], dayfirst=True)

# 3. Veriyi eskiden yeniye doğru sıralayıp satır indekslerini sıfırlama
df = df.sort_values("Tarih").reset_index(drop=True)

# Dönüşüm Sonrası Çıktı:
#        Tarih  Şimdi  Açılış  Yüksek  Düşük
# 0 2026-10-19  34.10   34.00   34.15  33.95
# 1 2026-10-20  34.25   34.10   34.30  34.05

# =====================================
# Eğitim
# =====================================

# En eski tarihten itibaren geçen gün sayısını hesaplama
df["Gün"] = (df["Tarih"] - df["Tarih"].min()).dt.days
# Örnek Çıktı (df["Gün"]): [0, 1, 2, 5, 6, ...]

# Bağımsız Değişken (X: Geçen Gün) ve Bağımlı Değişken (Y: Dolar Fiyatı)
x = df["Gün"].to_numpy(dtype=float)
y = df["Şimdi"].to_numpy(dtype=float)

# Verinin %80'ini Eğitim, %20'sini Test olarak manuel ayırma
egitimsayisi = int(len(x) * 0.8)

x_train = x[:egitimsayisi]  # İlk %80'lik dilim (Eğitim girdisi)
y_train = y[:egitimsayisi]  # İlk %80'lik dilim (Eğitim hedefi)

x_test = x[egitimsayisi:]  # Son %20'lik dilim (Test girdisi)
y_test = y[egitimsayisi:]  # Son %20'lik dilim (Test hedefi)

# =====================================
# Doğrusal Regrasyon Modeli
#
# Açılış, yüksek, düşük = bağımsız değişkenler
# =====================================

# Formül bileşenlerinin NumPy ile hesaplanması
toplamx = np.sum(x_train)  # ∑x
toplamy = np.sum(y_train)  # ∑y
toplamxy = np.sum(x_train * y_train)  # ∑(x * y)
toplamx2 = np.sum(x_train**2)  # ∑(x²)

n = len(x_train)  # Eğitim verisi eleman sayısı

# Eğim (a) ve Sabit Terim (b) hesabı
a = (n * toplamxy - toplamx * toplamy) / (n * toplamx2 - toplamx**2)
b = (toplamy - a * toplamx) / n

print(f"Doğru Denklemi: y = {a:.4f}x + {b:.4f}")
# Örnek Çıktı: Doğru Denklemi: y = 0.0452x + 33.8500
# (Anlamı: Dolar başlangıçta 33.85 TL iken her 1 günde ortalama 0.0452 TL artmaktadır)

# Test verisi üzerinde tahmin üretme (Kodun devamı mantığı):
y_tahmin = a * x_test + b
