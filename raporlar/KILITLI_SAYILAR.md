# MAKALE İÇİN KİLİTLİ SAYILAR (koddan birebir doğrulanmış)
# Kaynak: NoShow-AI-TAM-calisir.zip / veriler/ (15 Ağustos güncel sürüm)
# UYARI: Yol haritasındaki ESKİ sayılar (0.8153, 49593, 39674, -31.61) KULLANILMAYACAK.

## 1. KOHORT
- Temizlenmiş toplam: 46.641 randevu
- Eğitim: 37.312 (%80)  |  Saklı dış test: 9.329 (%20)
- Test no-show prevalansı: 0,0999 (~%10)
- Ham kohort (temizleme öncesi): ~49.593 → temizlik sonrası 46.641

## 2. ANA BULGU — ÜÇ DOĞRULAMA (Random Forable şampiyon, geçmiş YOK satırı)
- Satır-rastgele:  ROC-AUC = 0,775 [0,760–0,791], PR-AUC = 0,297
- Hasta-gruplu:    ROC-AUC = 0,658 [0,638–0,677], PR-AUC = 0,198
- Kronolojik:      ROC-AUC = 0,548 [0,532–0,563], PR-AUC = 0,187
- DÜŞÜŞ AYRIŞTIRMASI: toplam −0,227 = hasta sızıntısı (−0,117) + zamansal kayma (−0,110)
- Hasta geçmişi HİÇBİR senaryoda iyileştirmedi (0,775→0,764; 0,658→0,650; 0,548→0,548)

## 3. MODEL KARŞILAŞTIRMA (dış test, tüm modeller eşit optimize, satır-rastgele)
- Random Forest (ŞAMPİYON): ROC-AUC=0,775, PR-AUC=0,295, Brier=0,081
- LightGBM:  ROC-AUC=0,754, PR-AUC=0,285
- XGBoost:   ROC-AUC=0,751, PR-AUC=0,277
- CatBoost:  ROC-AUC=0,747, PR-AUC=0,264
- Logistic Regression: 0,638  |  Decision Tree: 0,630
- Eşik EĞİTİMDE seçildi (F1-optimal=0,171); test'e sabit → Recall=0,521 (sabit 0,5'te 0,07)

## 4. ÇÖKÜŞ MEKANİZMASI
- Yıl bazında no-show: 2016=%4,0 → 2017=%8,4 → 2018=%8,3 → 2019=%13,2 → 2020=%14,1 → 2021=%15,7 → 2022=%20,4
- Kronolojik kalibrasyon: Brier=0,141; ort tahmin=0,207 > gerçek taban=0,160 (model fazla tahmin)
- Takvim drop-column: tüm öznitelik 0,548 → takvim çıkınca 0,542 (düşüyor)
- SHAP #1 öznitelik: appointment_year (0,037), #2 lead_time (0,019)

## 5. ZAMAN-STABİL KISMİ KURTARMA
- Kronolojik tam öznitelik: 0,548 → zaman-stabil alt küme: 0,642 (PR-AUC 0,233, recall 0,367)
- DÜRÜST NOT: kazanım tek başına takvimi çıkarmaktan DEĞİL, öznitelik setini
  stabil alt kümeye indirgemekten. (Takvimi tek çıkarmak 0,548→0,542 düşürür.)

## 6. KLİNİK FAYDA (Karar Eğrisi)
- Model, eşik olasılık 0,05–0,50 aralığının TAMAMINDA "hepsi"/"hiçbiri"nden üstün net fayda
- Örn pt=0,10: model net fayda=0,041, "hepsi"=−0,0001, "hiçbiri"=0

## 7. ALT-GRUP ADALET (dış test)
- Kadın: AUC=0,769 (N=2288) | Erkek: AUC=0,777 (N=7040)
- Yaş 0-17: 0,778 | 18-39: 0,766 | 40-64: 0,750 | 65+: 0,764
- Gruplar arası fark: 0,029 (<0,10) → adil

## 8. BASELINE + OVERFITTING
- Model PR-AUC=0,295 vs prevalans baseline PR-AUC=0,100 → 3× üstün
- Genelleme boşluğu: train AUC=0,982, test AUC=0,775 → boşluk 0,207 (RF doğası)

## 9. SİMÜLASYON İKİLİ RAPORLAMA (aynı gerçek dünya, iki model kararı)
- İyimser (rastgele): atıl azalma=19,53 dk/gün, hasta artışı=0,53/gün
- Gerçekçi (kronolojik): atıl azalma=4,73 dk/gün, hasta artışı=0,13/gün
- DANIŞMAN DÜZELTMESİ: gerçek gelmeme artık test döneminin GERÇEK no_show
  etiketlerinden (yer gerçeği); rastgele prevalanstan değil. Gerçekçi ≈ iyimserin
  %24'ü. Her ikisi p<0,001 (iyimser p≈6e-129, gerçekçi p≈2e-32).

## VERİ SETİ KİMLİĞİ
- Kaynak: Salazar ve ark. (2022), Future Internet 14(1):3. UNIVALI/CER rehabilitasyon merkezi, güney Brezilya, 2016-2022.
- Etik: Univali no 4270.234, LGPD uyumlu. Doğrudan hasta ID YOK (vekil kimlik kullanıldı).
- KaggleV2-May-2016.csv KULLANILMADI (farklı şehir/sistem).

## 10. PSEUDO-ID DUYARLILIK ANALİZİ (danışman Talep 2)
- Pseudo-ID: 1395 benzersiz hasta, ort 35,6 randevu/hasta
- Çakışma oranı: %0,7 (10/1395) | Bölünme oranı: %3,3 (45/1395)
- Zaman-stabil kurtarma DUYARLILIK:
  * geçmiş öznitelikleri DAHİL: ROC-AUC = 0,550
  * geçmiş öznitelikleri HARİÇ: ROC-AUC = 0,541
  * Proxy geçmişinin katkısı: yalnızca +0,009
- YORUM: Kurtarma esas olarak TEMEL zaman-stabil özniteliklerden gelir;
  kırılgan proxy geçmişinden DEĞİL. Sonuç proxy gürültüsüne dayanmıyor.

## 11. SCRIPT 35 OOF DÜZELTMESİ (danışman Talep 3a) — YENİDEN ÇALIŞTIRILDI
- HATA BULUNDU VE DÜZELTİLDİ: OOF eşik modeli class_weight="balanced"
  kullanıyordu, AUC/test modeli kullanmıyordu — kalibrasyon uyumsuzluğu
  eşiği anlamsız yükseltip Recall'u 0,005'e düşürüyordu.
  Düzeltme: iki model BİREBİR aynı konfigürasyon (rs=42, class_weight yok).
- GÜNCEL SONUÇ: Kronolojik ROC-AUC = 0,642 (değişmedi), Recall(OOF eşik) = 0,777,
  F1 = 0,293 (önceki in-sample recall 0,367'den bile daha iyi, artık OOF disiplinli).
