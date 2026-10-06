import pandas as pd
import numpy as np 

import scipy.linalg as la
import scipy.stats as sa

from scipy.signal import find_peaks
from scipy.integrate import quad # intergral
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d

# Denklem optimizasyonu ile denklem minumum değeri bulma.

x = [1,2,4]
y = [10, 17, 21]

# Tahmin (Bir aralık içi tahmin)
f = interp1d(x, y)

# sonuc = f(1) --> 10
# sonuc = f(2) --> 17
sonuc = f(4) # --> 21.0

print(sonuc) # 21.0

# Aralık dışı tahmin işlemi

# Doğrusallık oluşturarak olmayan bir x verisini tahmin etme 
# örneğin 5 verisi x de yok dolayısı ile karşılığı yok.
# 3. parametre denklem parametresi (1. derece, 2. derece)
# n = x'in katsayısı, b = hata payı
# 1. derece bir denklem dönderir.
# doğrusal regresyon

n, b = np.polyfit(x, y, 1)

# tahmin etmek istediğimiz değer
tahmin = 5

sonuc2 = n * tahmin + b

print(sonuc2) # 25.14285714285714