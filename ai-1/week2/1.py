import pandas
import numpy as np
import statistics

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