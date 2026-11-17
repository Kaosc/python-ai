import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Modeli eğittik, tahminler yaptı. Peki modelimiz ne kadar başarılı? Bunu test etmek için gerçek verilerle modelin tahmin ettiği verileri karşılaştırırız.

gercek = [12, 123, 34, 56, 7, 88, 9]  # Veri seti
tahmin = [11, 110, 25, 58, 9, 92, 11]  # Tahmin edilen değerler (modelin çıktısı)

# ========================================================
# 1. MAE (Ortalama Mutlak Hata) - Model ortalama olarak kaç birim şaşırdı
# ========================================================

mae = mean_absolute_error(gercek, tahmin)
# Çıktı: 4.7142 (Ortalama 4.7 birim hata yapmış)

# ========================================================
# 2. R2 Skoru - Modelin ne kadar iyi öğrendiğini gösterir
# ========================================================

# R2 Skoru ise 1'e ne kadar yakınsa modelin o kadar iyi öğrendiğini kanıtlar (%97 muazzam bir başarıdır)
r2 = r2_score(gercek, tahmin)
# Çıktı: 0.9767 (%97 başarı oranı)

# ========================================================
# 3. RMSE (Kök Ortalama Kare Hata)
# ========================================================

# mean_squared_error metodu, gerçek ile tahmin arasındaki farkı (hatayı) bulur ve bunun karesini alır.
# Sonra tüm bu karelerin ortalamasını hesaplar (buna MSE denir).
# np.sqrt (Karekök) metodu ise bu karesi alınmış sonucun karekökünü alarak birimi tekrar verinin orijinal birimine (örneğin fiyat, kilo, derece) döndürür.

# Hataların karesi alındığı için, modelin yaptığı büyük hatalar çok daha ağır cezalandırılır.
# Yani model genelde iyi tahmin yapıp 1-2 veride çok fena çuvalladıysa, RMSE değeri fırlayarak sana "modelde ciddi bir dengesizlik var" uyarısı verir.
# MAE (Mutlak Hata) büyük hataları bu kadar sert belli etmez.
rmse = np.sqrt(mean_squared_error(gercek, tahmin))
# Çıktı: 5.099 (Ortalama 5 birim hata yapmış)

# ========================================================
# 4. MAPE ( Ortalama Mutlak Yüzde Hata)
# ========================================================

# - (np.array(gercek) - np.array(tahmin)): Önce aradaki farkı (hatayı) bulur.
# - / np.array(gercek): Hatayı gerçek değere böler. Böylece hatanın oransal büyüklüğünü bulur. (Örneğin 100 liralık üründe 10 lira hata yapmak %10'luk bir hatadır).
# - np.abs(...): Bulunan oranların mutlak değerini alır ki eksi ve artı hatalar birbirini götürüp sıfırlamasın.
# - np.mean(...): Tüm bu oranların aritmetik ortalamasını alır.
# - * 100: Çıkan ondalıklı küsuratı (örneğin 0.085) yüzdeye (8.5) çevirir.
mape = np.mean(np.abs((np.array(gercek) - np.array(tahmin)) / np.array(gercek))) * 100
# Çıktı: 8.5 (%8.5 hata payı)

# RMSE veya MAE sana birim cinsinden sonuç verir (Örn: "Ortalama 5 birim hata yaptık").
# Ama patrona veya sunuma çıkıp "5 birim hata yaptık" dersen "Bu iyi bir şey mi?" diye sorarlar.
# MAPE bunu herkesin anlayacağı dile, yani yüzdeye çevirir.
# "%8.5 hata payımız var" demek, modelin %91.5 oranında işe yaradığını yöneticilere net şekilde gösterir
