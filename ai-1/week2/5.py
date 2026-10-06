import pandas as pd
import numpy as np 

import scipy.linalg as la
import scipy.stats as sa

from scipy.signal import find_peaks
from scipy.integrate import quad # intergral
import matplotlib.pyplot as plt 

# Denklem optimizasyonu ile denklem minumum değeri bulma.

veri = [1,2,3,4,15,2,4,1]

tepe, bilgiler = find_peaks(veri)

print(tepe) # [4 6]

# Verideki uzunluk değeri
x = range(len(veri))

plt.plot(x, veri, marker="o", label="veriler")

plt.show()