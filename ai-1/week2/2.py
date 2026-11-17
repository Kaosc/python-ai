import numpy as np
# ========================= Aykırı Veri Temizleme ========================

# Eğer veride diğerlerine göre çok uçuk bir sayı varsa (dizideki 1000 gibi), 
# yapay zeka modelinin "kafası karışır" ve ağırlığı o sayıya vererek yanlış öğrenir.
# Tüm sayıları birbirine yakın bir skalaya (örneğin 0 ile 1 arasına) çekmeye "Normalizasyon" denir.

"""
1. Medyan ve IQR (Çeyrekler Arası Açıklık) ile Ölçekleme (Aykırı Değere Dayanıklı)
"""

# 1000 sayısı ortalamayı çok yükselteceği için, ortalama yerine verinin tam ortasındaki sayıyı (medyan) baz alarak ölçekleme yaparız.
# Görüldüğü üzere 34 değeri medyan olduğu için 0 oldu, diğerleri onun etrafında şekillendi.
# 1000 değeri hala çok büyük (18.941), yani aykırı olduğunu net şekilde izole ettik.

# Aykırı veri: 1000
liste = np.array([1, 34, 45, 65, 1000, 7, 1])

q75, q25 = np.percentile(liste, [75, 25])
aykiri = q75 - q25 # IQR = Q3 - Q1 (55.0 - 4.0 = 51.0)
medyan = np.median(liste) # Küçükten büyüğe sıralandığında tam ortadaki sayı: 34

x = (liste - medyan) / aykiri
# Çıktı: [-0.647, 0., 0.215, 0.607, 18.941, -0.529, -0.647]

"""
2. Z-Skor Normalizasyonu (Standart Sapma Tabanlı) 
"""

zskor = (list - np.mean(list)) / np.std(list)
# Çıktı: [-0.435, -0.342, -0.311, -0.254,  2.385, -0.418, -0.435]

"""
3. L2 Normu (Vektör Uzunluğuna Bölme)
"""
l2 = np.linalg.norm(list, keepdims=True)
normal = list / l2
# Çıktı: [0.0009, 0.033, 0.044, 0.064, 0.997, 0.006, 0.0009]

"""
4. Logaritmik Dönüşüm (Logaritma 10, 100, 1000 gibi üstel artan verileri doğrusallaştırıp aralığı daraltmak için çok etkilidir.)
"""

right = np.log1p(list)
# Çıktı: [0.69, 3.55, 3.82, 4.18, 6.90, 2.07, 0.69]


# Hangisinin kullanılacağı verinin karakterine (çok mu dağınık, çok mu uçuk) göre seçilir.