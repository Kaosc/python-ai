import pandas as pd
import numpy as np 

import scipy.linalg as la
import scipy.stats as sa

# Matris tanımlama. (Kare olmak zorunda, 2x2, 3x3 etc.)

matris = [
    [1,2,5],
    [3,4,6],
    [6,3,7]
  ]

# Matris determinat hesaplama.
res = la.det(matris)

# Matrisin satırlarını sütunlara çeviri. Tersini alma.
res2 = la.inv(matris)

print(res)


# örnek denklemi matrise çevirerek çözüm

# 2x + y = 5 --> y = 5 - 2x
# x + 3y = 6 --> y = (6 - x) / 2

# Katsayılar matrisi x değişken matrisi = y

# Katsayılar matirisi
a = [[2,1], [1,3]] 

# Değişkenler matrisi
b = [5, 6]

# Katsayılar ve değiken matrisi çarpımı bulma
# - Birinci matrisin satırı ile 2. matrisin sütunu eşit değilse çarpma yapılamaz
res3 = la.solve(a, b) # // [1.8, 1.4] 1.si= X, 2.si = Y
print(res3)


