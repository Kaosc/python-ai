"""
ÖRNEK: KNN SINIFLANDIRMASI
KNN Sınıflandırması ve Karmaşıklık Matrisi
"""

# K-En Yakın Komşu (KNN) algoritması, yeni gelen bir çiçeğin ölçümlerine bakıp
# uzayda kendisine en yakın $K$ komşusunun çoğunluğu hangi türse o sınıfı atar

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

iris = load_iris()
x = iris.data  # Bağımsız değişkenler (150 satır, 4 sütun)
y = iris.target  # Sınıf etiketleri (0: setosa, 1: versicolor, 2: virginica)

# Verinin %80'ini eğitim, %20'sini test olarak rastgele ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=17)

# Modeli oluşturma (Varsayılan komşu sayısı n_neighbors=5) ve eğitme
model = KNeighborsClassifier()
model.fit(x_train, y_train)

# Test verisiyle tahmin yapma
y_tahmin = model.predict(x_test)

# Karmaşıklık Matrisi (Confusion Matrix)
hata_matrisi = confusion_matrix(y_test, y_tahmin)
print(hata_matrisi)
# Çıktı:
# [[ 7  0  0]   -> Setosa: 7 doğru, 0 yanlış
#  [ 0 11  0]   -> Versicolor: 11 doğru, 0 yanlış
#  [ 0  2 10]]  -> Virginica: 10 doğru, 2 yanlış (Versicolor sanılmış)

# Hata matrisini tablo ve ısı haritası (Heatmap) olarak çizdirme
index = ["setosa", "vericolor", "virginica"]
cols = ["setosa", "vericolor", "virginica"]

hata_goster = pd.DataFrame(hata_matrisi, cols, index)

plt.figure(figsize=(10, 6))
sea.heatmap(hata_goster, annot=True)
plt.show()
