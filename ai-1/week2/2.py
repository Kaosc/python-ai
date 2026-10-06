import pandas as pd
import numpy as np 

import scipy.linalg as la
import scipy.stats as sa

a = [10, 12, 11, 13, 12]
b = [20, 21, 22, 23, 24]

# T testi yapma. İki veri arasındaki değerler
sonuc = sa.ttest_ind(a, b)

# 1.si t değeri, 2.si p değeri (anlamlılık oranı)
print("t değeri: ", sonuc.statistic)
print("p değeri: ", sonuc.pvalue)


