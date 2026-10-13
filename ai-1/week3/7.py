import pandas as pd
import numpy as np 

# 2 Matrisin çarpımı

x = np.array([[1,2], [3,4]])
y = np.array([[5,6], [7,8]])

# @ işareti matrislerin çarpımı için
c = x @ y

print(c)

"""
[[19 22]
 [43 50]]
"""

# Matrisi ters çevirme (satırla sütun yer değiştirir)

ters = x.T
print(ters)

"""
[[1 3]
 [2 4]]
"""

# 

inverse = np.la.inv