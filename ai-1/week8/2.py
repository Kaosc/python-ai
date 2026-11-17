"""
SVM Sınıflandırması, Ölçekleme ve Karar Sınırları
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib.colors import ListedColormap

# 1. Veri setini yükleme (Sadece 2. ve 3. indeksli sütunlar: petal length ve petal width)
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# 2. Eğitim (%80) ve Test (%20) ayrımı
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Standardizasyon (Z-Skor ölçekleme)
sc = StandardScaler()
sc.fit(x_train)  # Eğitim verisinin ortalama ve standart sapmasını öğrenir
x_train_std = sc.transform(x_train)  # Eğitim verisini ölçekler
x_test_std = sc.transform(x_test)  # Test verisini aynı ölçekle dönüştürür
# Örnek Çıktı (x_train_std ilk 2 satır):
# [[-1.3412724  -1.31260282]
#  [ 0.13214928 -0.26425812]]

# 4. Eğitim ve test verilerini tekrar birleştirme
x_combined_std = np.vstack((x_train_std, x_test_std))  # Dikey (alt alta) birleştirme
y_combined_Std = np.hstack((y_train, y_test))  # Yatay (yan yana) birleştirme


# 5. Karar bölgelerini çizen fonksiyon
def karar(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # Grafik sınırlarını belirleme
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Izgara (Grid) koordinat matrisleri oluşturma
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # Izgaradaki tüm noktalar için modelden tahmin alma
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)


# 6. SVM Modelini oluşturma ve eğitme
# kernel="linear" -> Doğrusal ayrım, kernel="rbf" -> Gauss yöntemi (eğrisel ayrım)
svm = SVC(kernel="linear", random_state=1)
svm.fit(x_train_std, y_train)
