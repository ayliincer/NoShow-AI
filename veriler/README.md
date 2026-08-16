# veriler/ Klasörü

Bu klasör, analiz **sonuç** dosyalarını (CSV) içerir; ham ve büyük ara veri
dosyaları GİZLİLİK ve BOYUT nedeniyle bu depoya dahil EDİLMEMİŞTİR.

## Depoda olan (sonuç dosyaları)
Model karşılaştırmaları, altı-senaryo tablosu, kalibrasyon, karar eğrisi,
alt-grup adalet, simülasyon ikili raporlama gibi tüm ANALİZ ÇIKTILARI burada.

## Depoda OLMAYAN (ham/işlenmiş veri)
Aşağıdaki dosyalar, hasta düzeyinde bilgi içerdiğinden ve veri koruma
(LGPD/etik kurul) gereği halka açık paylaşılamayacağından depoya konmamıştır:
- medical-appointments-no-show-en.csv (ham veri)
- medical_appointments_train.csv / _test.csv
- medical_appointments_preprocessed_step01/step02.csv
- pseudo_hasta_gecmisi_ozellikleri.csv, step02_pseudo_gecmis_dahil.csv
- KaggleV2-May-2016.csv (bu çalışmada kullanılmamıştır)

## Veriye erişim
Ham veri, ilgili kurumun etik onayı çerçevesinde talep üzerine
sağlanabilir. Veri elde edildikten sonra, kodlar/PIPELINE_SIRASI.md'deki
sırayla (10-16 ön işleme) işlenmiş dosyalar yeniden üretilir.
