"""
Grafik Çizme
Hiyerarşik Kümeleme ve Dendrogram
"""

# Elinde sınıf etiketleri (target) olmadığında, verilerin birbirine olan
# geometrik uzaklıklarına göre kendiliğinden gruplanmasını sağlayan yöntemdir.

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage

iris = load_iris()

x = iris.data[:, 0]  # 1. sütundaki tüm elemanlar (sepal_length)
y = iris.data[:, 1]  # 2. sütundaki tüm elemanlar (sepal_width)

plt.scatter(x, y)  # İki özelliğin nokta dağılım grafiği

# Hiyerarşik kümeleme matrisi oluşturma
data = iris.data
linkage_data = linkage(data, method="ward", metric="euclidean")

# Dendrogram (Ağaç şeması) çizimi
plt.figure(figsize=(12, 6))
dendrogram(linkage_data)

plt.title("Kümeleme Grafiği")
plt.xlabel("Çiçek Örnekleri")
plt.ylabel("Küme Mesafeleri")
plt.show()
