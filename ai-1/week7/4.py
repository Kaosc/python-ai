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
