import pandas as pd
import numpy as np 

import scipy.linalg as la
import scipy.stats as sa

from scipy.optimize import minimize_scalar
from scipy.integrate import quad # intergral

# Denklem optimizasyonu ile denklem minumum değeri bulma.

def f(x) :
    return x**2

# 0 alt limit, 2 üst limit.
sonuc = quad(f, 0, 2)

print(sonuc)

"""
1. değer integral değeri
2. değer hata oranı (hata değeri neredeyse 0)
(2.666666666666667, 2.960594732333751e-14)
"""