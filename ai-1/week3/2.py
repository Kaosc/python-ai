import scipy.stats as sa

a = [10, 12, 11, 13, 12]
b = [20, 21, 22, 23, 24]

# İki Bağımsız Örneklem T-Testi
sonuc = sa.ttest_ind(a, b)

print("t değeri:", sonuc.statistic)  # T değeri
print("p değeri:", sonuc.pvalue)  # P değeri

# Çıktı:
# t değeri: -12.649110640673518
# p değeri: 4.887823377545672e-06  (0.00000488...)
