import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = [1, 2, 3]
y = [10, 11, 12]

# 1. X'i 2 boyutlu sütuna çevir ve
x = np.array(x).reshape(-1, 1)

# %80-%20 böl
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

# 2. Modeli kur ve eğit (fit)
model = LinearRegression()
model.fit(x_train, y_train)

# 3. Tahmin yap (predict)
y_tahmin = model.predict(x_test)

print(y_tahmin)

# ===================================================================

from scipy import stats

yeni_x = 5

# DİKKAT: Sol tarafa mutlaka 5 değişken yazılır!
slope, intercept, r, p, std_err = stats.linregress(x, y)

# y = ax + b formülüyle tahmin:
y_tahmin = slope * yeni_x + intercept

# ===================================================================

import numpy as np

n = len(x_train)
toplamx = np.sum(x_train)
toplamy = np.sum(y_train)
toplamxy = np.sum(x_train * y_train)
toplamx2 = np.sum(x_train**2)

a = (n * toplamxy - toplamx * toplamy) / (n * toplamx2 - toplamx**2)
b = (toplamy - a * toplamx) / n

y_tahmin = a * x_test + b
