Yapay Zekâ

Veri işleme: Tahmin, sınıflandırma, kümeleme.
Hazır kütüphane kullanmadan ham veriden algoritmayı kendimiz yazacağız; genelde diziler ve metotlar kullanılacak.
Kullanılan kütüphaneler: pandas, numpy, statistics, scipy (scipy.stats).

Hata payını (değerini/yüzdesini) bulma: Hata değerlendirme yöntemlerinin ayrı formülleri bulunmaktadır. Değerlendirmede 2 adet girdi kullanılır: Y (Gerçek veri seti) ve Y' (Tahmin edilen değerler / model çıktısı).
Kullanılan yöntemler: 
   RMSE (Kök Ortalama Kare Hata), 
   MASE (Ölçeklenmiş Mutlak Hata Ortalaması), 
   MAPE (Ortalama Mutlak Yüzde Hata), 
   R² (R-Kare Skoru).