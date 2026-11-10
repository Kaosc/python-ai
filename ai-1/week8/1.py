import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:,0]>0, X_xor[:,1]>1)

y_xor = np.where(y_xor,1,0) #elimizde ki değerleri sayısal sınıflara dönüştürür
plt.scatter(
    X_xor[y_xor == 1,0],
    X_xor[y_xor == 1,1],
    c = "royalblue",
    marker = "s",
    label = "sınıf 1",
    edgecolor = "black"
    
    )

plt.scatter(
    X_xor[y_xor == 0,0],
    X_xor[y_xor == 0,1],
    c = "royalblue",
    marker = "s",
    label = "sınıf 2",
    edgecolor = "black"
    
    )

plt.xlim([-3,3])
plt.ylim([-3,3])
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")
plt.legend(loc = "best")
plt.tight_layout()
plt.show()