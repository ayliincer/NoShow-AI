# Danışman Değerlendirmesi Sonrası Yapılan Düzeltmeler

Bu belge, danışman (hoca) değerlendirmesindeki dört kritik madde ve ikincil
iyileştirmelerin kod üzerinde nasıl uygulandığını ve ÇALIŞTIRILARAK doğrulanmış
sonuçlarını özetler.

## MADDE 1 — Eşik optimizasyonu test setinde yapılıyordu (ince sızıntı)
DÜZELTİLDİ. Script 30 ve 35'te F1-optimal eşik artık TEST'ten değil, eğitim
setinde (30: 5-katlı CV out-of-fold; 35: eğitim dönemi tahminleri) seçilip
test'e SABİT uygulanıyor.
Doğrulanmış etki: RF recall (optimal eşik) 0.56 -> 0.52 (30); zaman-stabil
recall 0.51 -> 0.37 (35). ROC-AUC/PR-AUC değişmedi (beklendiği gibi). Değerler
artık iyimser değil, dürüst.

## MADDE 2 — İki farklı model paketi; downstream çıktılar uyuşmuyordu
DÜZELTİLDİ. SHAP (17), simülasyonlar (20, 21, 23) ve standalone pipeline (31)
artık script 30'un ürettiği v4 "tam adil" paketini (hava durumsuz, scaler'sız)
yüklüyor. Eski scaler/surekli_sutunlar blokları kaldırıldı.
Doğrulandı: standalone pipeline ROC-AUC=0.7753 (birebir); SHAP v4 ile çalıştı
(#1 öznitelik appointment_year=0.037); üç simülasyon aynı olasılıkları (ort.
0.1049) üretiyor.

## MADDE 3 — Rastgele bölünmede hasta sızıntısı (en güçlü bilimsel fırsat)
DÜZELTİLDİ + GENİŞLETİLDİ. Script 34 (artık 34_makale_kesin_alti_senaryo.py),
GroupShuffleSplit(groups=pseudo_id) ile üçüncü bir "hasta-gruplu" bölme ekleyerek
{satır-rastgele, hasta-gruplu, kronolojik} × {geçmiş yok, geçmiş var} ızgarası
üretir. pseudo_id NaN olan kayıtlar tekil hasta sayılır (veri kaybı yok).
DOĞRULANMIŞ SONUÇ (düşüşün ayrıştırılması, geçmiş yok):
  Satır-rastgele: 0.775 [0.760-0.791]
  Hasta-gruplu:   0.658 [0.638-0.677]   (hasta sızıntısı payı: -0.117)
  Kronolojik:     0.548 [0.532-0.563]   (zamansal kayma payı:  -0.110)
Yani 0.78->0.55 düşüşünün ~yarısı hasta sızıntısı, ~yarısı zamansal kayma.

## MADDE 4 — Çöküşün mekanizmasını nicelleştirin
YAPILDI. Yeni script 36_cokusun_mekanizmasi.py:
  (a) Yıl bazında no-show taban oranı: 2016 %4.0 -> 2022 %20.4 [label/prior shift]
  (b) Yıl bazında test AUC: 2021=0.555, 2022=0.496 (dağılım kaydıkça bozuluyor)
  (c) Kronolojik kalibrasyon: Brier=0.141; model ort. tahmin 0.207 > gerçek 0.160
      (prior kaymasını yakalayamıyor)
  (d) Takvim özniteliklerine bağımlılık (drop-column): tam öznitelik setinde
      takvimi çıkarmak AUC'yi 0.548->0.542 DÜŞÜRÜR.
ÖNEMLİ DÜRÜST NOT: Script 35'teki kurtarma (0.55->0.64) tek başına "takvimi
çıkarmak"tan DEĞİL, tüm öznitelik setini yalnızca 6 zaman-stabil özniteliğe
indirgemekten gelir. Makalede bu ayrım açıkça belirtilmeli (label shift +
durağan-olmayan takvim bağımlılığı + genel aşırı-uyum birlikte).

## MADDE 5 — Pseudo-ID sınırlaması (makale metnine ait)
Script 32 çakışma/bölünme oranlarını ölçüyor (çakışma %0.7, bölünme %3.3).
Metot ve Sınırlamalar bölümlerinde: geçmiş özniteliklerinin "yaklaşık hasta
geçmişi" olduğu ve kesinlikle geçmişe-dönük (cumsum - kendi etiketi) kurulduğu
vurgulanmalı.

## MADDE 6 — Tutarlılık ve sağlamlık
KISMEN. Script 34'te tek sabit RF konfigürasyonu (n_estimators=500) kullanılıyor
ve ana sonuçlara bootstrap %95 güven aralığı eklendi. Script 35 docstring'indeki
eski numara atıfı (script 37) düzeltildi. Kalan: 24/22'deki farklı RF
hiperparametrelerinin manşet sayılar için kullanılmadığı (yalnızca 30 ve 34'ün
manşet ürettiği) makalede netleştirilmeli.

## GÜNCELLENMESİ GEREKEN
Makale taslağındaki sayı tabloları bu yeni (çalıştırılmış) sonuçlarla
güncellenmelidir; özellikle 6-senaryolu ızgara, düşüş ayrıştırması ve dürüst
recall değerleri.
