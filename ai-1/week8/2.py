# svm

import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import datasets
from matplotlib.colors import ListedColormap

# veri seti yükle
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

# test size = 0.2
# train size = 0.8

# Ağı eğitmek içi gerekli olan standartitasyon işelmerli
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

sc=StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

#

# eğitim verilerini dikeyde/sütunda birleştir.
x_combined_std = np.vstack(x_train_std, x_test_std)

#yatayda
y_combined_Std = np.hstack(y_train, y_test)

#karar bölgelerini çizen fonksyion

def karar(x,y,classifier,test_idx=True, resulation=0.02):
    markers = ("s","x","o","^","v")
    colors = ("red","blue","lightgreen","grav","cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
    
    xx1, xx2 = np.meshgrid(x1_min, x1_max,resulation), np.meshgrid(x2_min, x2_max,resulation)
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # sondaki t [x1, x2] şeklindeki listeyi
    # [x1,
    # x2,]
    # şekline çeviri.
    Z = Z.reshape(xx1.shape)
    
   
# model # rbf = gauss yöntem # lineer = doğrusal,
svm = SVC(kernel = "linear", random_state= 1)

# Modelin test aşamasında kullanılacak standart aşamasını gösterir
svm.fit(x_train_std, y_train)