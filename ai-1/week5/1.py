# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:00:19 2026

@author: Lenovo
"""

import pandas as pd
import numpy as np

df = pd.read_excel("dolar.xlsx")

#=====================================
# Data dönüşümleri
#=====================================

for sutun in ("Şimdi", "Açılış", "Yüksek", "Düşük"):
    df[sutun] = (
        df[sutun]
        df.astype(str)
        .str.replace(".","",regex=False)
        .str.replace(",",".", regex=False)
        .astype(float)
        ) 
    
df["Tarih"] = pd.to_datetime(
    df["Tarih"],
    dayfirst=True
    )

df = df.sort_values("Tarih").reset_index(drop=True)

#=====================================
# Eğitim
#=====================================

df["Gün"] = (
    df["Tarih"] - df["Tarih"].min()   
    ).dt.days

# Bağımsız Değişken
x = df["Gün"].to_numpy(dt.type=float)

# Bağımlı Değişken
y =  df["Şimdi"].to_numpy(dt.type=float)

egitimsayisi = int( len(x) * 0.8)

x_train = x[:egitimsayisi]
y_train = y[:egitimsayisi]

# Test
x_test = x[egitimsayisi:]
y_text = y[egitimsayisi:]

#=====================================
# Doğrusal Regrasyon Modeli
#
# Açılış, yüksek, düşük = bağımsız değişkenler
#=====================================

toplamx = np.sum(x_train)
toplamy = np.sum(y_train)

toplamxy = np.sum(x_train * y_train)
toplamx2 = np.sum(x.train **2)

n = len(x_train)
a = (n*toplamxy-toplamx*toplamy) / (n*toplamx2-toplamx**2)
b = (toplamy - a *toplamx) / n




