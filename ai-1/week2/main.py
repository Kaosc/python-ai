import pandas
import numpy as np
import statistics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n=================== PAGE 1 ===================\n")

a = [1, 2, 3]
# Verilen dizinin aritmetik ortalamasını
statistics.mean(a)
# output: 2

# Veriyi satır ve indeks numaraları olan, sütunlarına isim verilebilen iki boyutlu bir Excel tablosuna çevirir.
# Model eğitimi her zaman DataFrame veya NumPy matrisleri üzerinden yapılır.
a = np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9]))
dataFrame = pandas.DataFrame(a)
# output:
#     0  1  2
#  0  1  2  3
#  1  4  5  6
#  2  7  8  9

# Algoritmaları test etmek için elimizde her zaman hazır bir CSV dosyası olmayabilir.
# size=(3,4) diyerek 3 satır 4 sütunluk, 1 ile 10 arası sayılardan oluşan "yapay" bir veri seti (matris) üretip algoritmamızı bunun üzerinde deneriz.
matris = np.random.randint(1, 10, size=(3, 4))  # 3 satır 4 sütun
# output:
# [[6 9 2 6]
#  [9 9 2 3]
#  [6 2 8 1]]

# Eşit Aralıkta Veri Üretme: 1 ile 10 arasında 50 tane sayı üretir ve bunları eşit aralıklarla -5 ile 5 arasına dağıtır.
randomList = np.linspace(-5, 5, 50)
# [-5.         -4.79591837 -4.59183673 -4.3877551  -4.18367347 -3.97959184
#  -3.7755102  -3.57142857 -3.36734694 -3.16326531 -2.95918367 -2.75510204
#  -2.55102041 -2.34693878 -2.14285714 -1.93877551 -1.73469388 -1.53061224
#  -1.32653061 -1.12244898 -0.91836735 -0.71428571 -0.51020408 -0.30612245
#  -0.10204082  0.10204082  0.30612245  0.51020408  0.71428571  0.91836735
#   1.12244898  1.32653061  1.53061224  1.73469388  1.93877551  2.14285714
#   2.34693878  2.55102041  2.75510204  2.95918367  3.16326531  3.36734694
#   3.57142857  3.7755102   3.97959184  4.18367347  4.3877551   4.59183673
#   4.79591837  5.        ]

print("\n=================== PAGE 2 ===================\n")

# ========================= Aykırı Veri Temizleme ========================

# Eğer veride diğerlerine göre çok uçuk bir sayı varsa (dizideki 1000 gibi), yapay zeka modelinin "kafası karışır" ve ağırlığı o sayıya vererek yanlış öğrenir.
# Tüm sayıları birbirine yakın bir skalaya (örneğin 0 ile 1 arasına) çekmeye "Normalizasyon" denir.

# Aykırı veri: 1000
list = np.array([1, 34, 45, 65, 1000, 7, 1])

q75, q25 = np.percentile(list, [75, 25])
aykiri = q75 - q25  # Çeyrekler arası açıklık (IQR)
medyan = np.median(list)  # Ortanca değer: 34

# Standart Normalizasyon - Medyan
x = (list - medyan) / aykiri
# output: [-0.647, 0., 0.215, 0.607, 18.941, -0.529, -0.647]

# 1000 sayısı ortalamayı çok yükselteceği için, ortalama yerine verinin tam ortasındaki sayıyı (medyan) baz alarak ölçekleme yaparız.
# Görüldüğü üzere 34 değeri medyan olduğu için 0 oldu, diğerleri onun etrafında şekillendi.
# 1000 değeri hala çok büyük (18.941), yani aykırı olduğunu net şekilde izole ettik.

# Diğer alternatif normalizasyon yöntemleri:

# Z-skor standart sapmayı kullanır
zskor = (list - np.mean(list)) / np.std(list)
# Çıktı: [-0.435, -0.342, -0.311, -0.254,  2.385, -0.418, -0.435]

# L2 normu, vektör uzunluğuna bölme.
l2 = np.linalg.norm(list, keepdims=True)
normal = list / l2
# Çıktı: [0.0009, 0.033, 0.044, 0.064, 0.997, 0.006, 0.0009]

# Logaritma ise 10, 100, 1000 gibi üstel artan verileri doğrusallaştırıp aralığı daraltmak için çok etkilidir.
right = np.log1p(list)
# Çıktı: [0.69, 3.55, 3.82, 4.18, 6.90, 2.07, 0.69]

# Hangisinin kullanılacağı verinin karakterine (çok mu dağınık, çok mu uçuk) göre seçilir.

print("\n=================== PAGE 3 ====================\n")

# Modeli eğittik, tahminler yaptı. Peki modelimiz ne kadar başarılı? Bunu test etmek için gerçek verilerle modelin tahmin ettiği verileri karşılaştırırız.

gercek = [12, 123, 34, 56, 7, 88, 9]  # Veri seti
tahmin = [11, 110, 25, 58, 9, 92, 11]  # Tahmin edilen değerler (modelin çıktısı)

# ========================================================
# MAE (Ortalama Mutlak Hata) - Model ortalama olarak kaç birim şaşırdı
# ========================================================

mae = mean_absolute_error(gercek, tahmin)
# Çıktı: 4.7142 (Ortalama 4.7 birim hata yapmış)

# ========================================================
# R2 Skoru (R-Squared Score) - Modelin ne kadar iyi öğrendiğini gösterir
# ========================================================

# R2 Skoru ise 1'e ne kadar yakınsa modelin o kadar iyi öğrendiğini kanıtlar (%97 muazzam bir başarıdır)
r2 = r2_score(gercek, tahmin)
# Çıktı: 0.9767 (%97 başarı oranı)

# ========================================================
# RMSE (Root Mean Squared Error - Kök Ortalama Kare Hata)
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
# MAPE (Mean Absolute Percentage Error - Ortalama Mutlak Yüzde Hata)
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
