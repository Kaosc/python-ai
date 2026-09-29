import pandas as pd
import numpy as np
import statistics as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =======================================
# PAGE 1
# =======================================
print("\n=======================================\n")

a = [1, 2, 3]
st.mean(a)  # 2.0
print("Mean: ", st.mean(a))  # 2.0

a = np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9]))
dataFrame = pd.DataFrame(a)
print("DataFrame: ", dataFrame)


matris = np.random.randint(1, 10, size=(3, 4))  # 3 satır 4 sütun
print("Matris: ", matris)

# =======================================
# PAGE 2
# =======================================
print("\n=======================================\n")


list = np.array([1, 34, 45, 65, 1000, 7, 1])

# Aykırı veri: 1000.
# Aykırı veri temizleme

# İki çeyrek arası (Çeyrek (Q)). Dizi ilk paramtere. q -> aykırı değer yapısı
q75, q25 = np.percentile(list, [75, 25])
aykiri = q75 - q25
medyan = np.median(list)

print("Medyan", medyan)  # 34

# Dar aralığa çekme - normalizasyon

# Standart Normalizasyon - Medyan
x = (list - medyan) / aykiri
print("Normalize:", x)  # [-0.64705882, 0. , 0.21568627, 0.60784314, 18.94117647, -0.52941176, -0.64705882]

# Z skor ile normalizasyon
zskor = (list - np.mean(list)) / np.std(list)
print("Z skor: ", zskor)

l2 = np.linalg.norm(list, keepdims=True)
normal = list / l2
print("Normal: ", normal)

# Verilerin logaritması
right = np.log1p(list)
print("Right: ", right)

# =======================================
# PAGE 3
# =======================================
print("\n=======================================\n")


# -5 ile 5 arasında 100 tane sayı (float)
list = np.linspace(-5, 5, 50)
print("List: ", list)

# =======================================
# PAGE 4
# =======================================
print("\n=======================================\n")

# Hata payı bulma
import pandas as pd
import numpy as np
import statistics as st

gercek = [12, 123, 34, 56, 7, 88, 9]
tahmin = [11, 110, 25, 58, 9, 92, 11]

# 1 MAE hesaplama
mae = mean_absolute_error(gercek, tahmin)
print("mae: ", mae)  # 4.714285714285714 = %4

# R2 hesap
r2 = r2_score(gercek, tahmin)
print("r2: ", r2)  # 0.9767034068136272 = %97

# =======================================
# !! below codes auto generated !!
# =======================================

# RMSE hesap
rmse = np.sqrt(mean_squared_error(gercek, tahmin))
print("rmse: ", rmse)  # 7.0710678118654755 = %7

# MAPE hesap
mape = np.mean(np.abs((np.array(gercek) - np.array(tahmin)) / np.array(gercek))) * 100
print("mape: ", mape)  # 7.071067811865475
