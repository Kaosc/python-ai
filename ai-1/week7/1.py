"""
GRAFİK ÇİZME
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Kümeleme işlemi için
from scipy.cluster.hierarchy import dendrogram, linkage

iris = load_iris()

x = iris.data[:, 0]  # İlk sütundaki tüm elemanlar
y = iris.data[:, 1]

# Grafik

plt.scatter(x, y)
# plt.show()

# Verilerin nerede yoğunlaştığını ve ortak merkez noktalarının nerede
# bulunduğunu görmek için hiyerarşik kümeleme grafiği oluşturulur.
data = iris.data
linkage_data = linkage(data, method="ward", metric="euclidean")
# Grafiği çiz
plt.figure(figsize=(12, 6))
dendrogram(linkage_data)

plt.title("Kümeleme Grafiği")
plt.xlabel("Çiçek Örnekleri")
plt.ylabel("Küme Mesafeleri")
plt.show()
