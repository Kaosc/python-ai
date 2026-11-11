import scipy.linalg as la
import numpy as np

# ==============================================
# 1. Matris İşlemleri
# ==============================================

x = np.array([[1, 2], [3, 4]])

y = np.array([[5, 6], [7, 8]])

# Matris Çarpımı (@)
# Birinci matrisin satırları ile ikinci matrisin sütunlarının çarpımı alınır ve toplanır.
c = x @ y
print("Matris Çarpımı (@):\n", c)
# Çıktı:
# [[19 22]
#  [43 50]]

# Transpoz (.T) (X matrisinin transpozu)
# Transpoz, bir matrisin satır ve sütunlarının yer değiştirmesidir. Yani, matrisin satırları sütun olur ve sütunları satır olur.
ters = x.T
print("Transpoz (.T):\n", ters)
# Çıktı:
# [[1 3]
#  [2 4]]

# ==============================================
# 2. Determinant, Matris Tersi ve Denklem Çözümü
# ==============================================
matris = [[1, 2, 5], [3, 4, 6], [6, 3, 7]]

# Determinant (la.det)
# Determinant, bir kare matrisin tek bir sayısal değerini temsil eder ve matrisin tersinin olup olmadığını belirler.
# Eğer determinant sıfır ise, matrisin tersi yoktur.
det_sonuc = la.det(matris)
print("Determinant:", det_sonuc)
# Çıktı: 15.0

# Matris Tersi (la.inv) (inv = inverse)
# Matrisin tersi, bir matrisin çarpımı ile birim matrisi elde etmek için kullanılan başka bir matristir.
# Yani, A * A^-1 = I (birim matris).
inv_sonuc = la.inv(matris)
print("Matris Tersi:\n", inv_sonuc)
# Çıktı:
# [[-0.66666667  0.06666667  0.53333333]
#  [-1.          1.5        -0.66666667]
#  [ 1.         -0.66666667  0.13333333]]

# ==============================================
# 3. Lineer Denklem Sistemi Çözümü (Ax = b)
# ==============================================

# 2x + 1y = 5
# 1x + 3y = 6
a = [[2, 1], [1, 3]]  # Katsayılar matrisi
b = [5, 6]  # Sonuç matrisi

denklem_sonuc = la.solve(a, b)
print("Denklem Çözümü [x, y]:", denklem_sonuc)
# Çıktı: [1.8 1.4]  (x = 1.8, y = 1.4)
