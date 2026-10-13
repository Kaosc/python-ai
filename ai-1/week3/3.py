import pandas as pd
import numpy as np 

import scipy.linalg as la
import scipy.stats as sa

from scipy.optimize import minimize_scalar

# Denklem optimizasyonu ile denklem minumum değeri bulma.

def f(x) :
    return x**2-4*x+5

sonuc = minimize_scalar(f)

print(sonuc)

"""
 message: 
          Optimization terminated successfully;
          The returned value satisfies the termination criteria
          (using xtol = 1.48e-08 )
 success: True
     fun: 1.0 <---- Minimum fonksiyon değeri
       x: 2.0 <---- minimum x değeri
     nit: 4
    nfev: 8
"""