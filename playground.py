import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

iris = load_iris()
x = iris.data  # Girdiler (150 çiçek, 4 özellik)
y = iris.target  # Etiketler (0: setosa, 1: versicolor, 2: virginica)

# 1. Veriyi %80 Eğitim, %20 Test olarak rastgele karıştırıp bölme
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=17)

# 2. Modeli kurma (Varsayılan komşu sayısı K = 5) ve eğitme
model = KNeighborsClassifier()
model.fit(x_train, y_train)

# 3. Test sınavı ve Hata Matrisi
y_tahmin = model.predict(x_test)
hata_matrisi = confusion_matrix(y_test, y_tahmin)
# Çıktı:
# [[ 7  0  0]   -> Setosa: 7 doğru, 0 yanlış
#  [ 0 11  0]   -> Versicolor: 11 doğru, 0 yanlış
#  [ 0  2 10]]  -> Virginica: 10 doğru, 2 yanlış (Versicolor sanılmış)

# 4. Isı Haritası (Heatmap) çizimi
index = ["setosa", "vericolor", "virginica"]
cols = ["setosa", "vericolor", "virginica"]
hata_goster = pd.DataFrame(hata_matrisi, cols, index)

sea.heatmap(hata_goster, annot=True)
plt.show()
