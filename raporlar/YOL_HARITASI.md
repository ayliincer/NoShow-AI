# NoShow-AI MAKALE YOL HARİTASI (SBED, güncel sayılarla)

## SABİT KARARLAR
- Dergi: KTÜ Sosyal Bilimler Enstitüsü Sosyal Bilimler Dergisi (SBED)
- Format: IMRAD, Türkçe tam metin + İngilizce Abstract
- Çerçeve: "Kronolojik doğrulama farkı / metodolojik uyarı" (SOTA yenme İDDİASI YOK)
- Yazım dili: Türkçe (SBED Türkçe/İngilizce kabul ediyor; Türkçe seçtik)
- Şampiyon: Random Forest (Optimize), 56 öznitelik, hava durumsuz
- Uzunluk: max 25 sayfa; Times New Roman 11pt; öz max 200 kelime 9pt

## YAZIM SIRASI (senin sıran doğru, aynen)
ADIM 1: Yöntem  →  ADIM 2: Bulgular+Tablolar  →  ADIM 3: Giriş+İlgili Çalışmalar
→  ADIM 4: Tartışma+Sınırlamalar  →  ADIM 5: Öz+Başlık+Sonuç

---

## ADIM 1 — YÖNTEM (en somut, buradan başlıyoruz)
Alt bölümler:
  3.1 Veri seti ve etik (Salazar 2022, UNIVALI/CER, 46.641 kayıt, LGPD, hasta ID yok)
  3.2 Ön işleme ve sızıntı önleme (train'den öğrenme, hava durumu çıkarma → 56 öznitelik)
  3.3 Model ve eşik seçimi (6 model, 5-kat CV, eşik EĞİTİMDE seçilir)
  3.4 Üç doğrulama stratejisi (satır-rastgele / hasta-gruplu GroupShuffleSplit / kronolojik)
  3.5 Vekil hasta kimliği (dogum-cinsiyet-şehir; çakışma %0,7; geçmişe-dönük)
  3.6 Açıklanabilirlik + simülasyon (SHAP TreeExplainer; SimPy overbooking)
TABLO YOK (metin ağırlıklı). Sızıntı disiplinini burada vurgula.

## ADIM 2 — BULGULAR (sayıların kilitlendiği yer)
  4.1 Model seçimi + rastgele performans → TABLO 1 (6 model, RF şampiyon 0,775)
  4.2 Vekil kimlik güvenilirliği (çakışma %0,7, geçmiş no-show ilişkisi)
  4.3 MERKEZİ BULGU: üç doğrulama → TABLO 2 (6 senaryo) + TABLO 3 (düşüş ayrıştırma)
      0,775 → 0,658 → 0,548; sızıntı −0,117, kayma −0,110
  4.4 Çöküş mekanizması (yıl bazında %4→%20, kalibrasyon, SHAP appointment_year)
  4.5 Sağlamlık + klinik fayda → TABLO 4 (alt-grup) + karar eğrisi + baseline
  4.6 Simülasyon ikili raporlama → TABLO 5 (iyimser 24,2 vs gerçekçi 6,5 dk)

## ADIM 3 — GİRİŞ + İLGİLİ ÇALIŞMALAR
  Giriş: no-show yükü, ML yaygınlığı, rastgele-split iki gizli sızıntı, boşluk
  İlgili çalışmalar: no-show ML literatürü + veri sızıntısı + zamansal doğrulama
  → TABLO (literatür vs bu çalışma konumu)
  3 özgün katkı: (1) sızıntısız protokol (2) üç doğrulama + ayrıştırma (3) mekanizma

## ADIM 4 — TARTIŞMA + SINIRLAMALAR
  Hakem zırhları:
   - Genelleme boşluğu 0,207 → RF doğası, test performansı esas
   - appointment_year bağımlılığı → yıllık re-training gereği
   - Hasta geçmişi neden işe yaramadı → vekil kimlik + zamansal yapı
   - Simülasyon: bekleme değil atıl/kapasite odağı (dürüst)
  Sınırlamalar: tek merkez, vekil kimlik, tek kronolojik nokta, prospektif yok

## ADIM 5 — ÖZ + BAŞLIK + SONUÇ
  Türkçe Öz (200 kelime, 9pt) + İngilizce Abstract + 3-5 anahtar kelime
  Başlık: "...Performans Düşüşünün Ayrıştırılması..." 
  SONUÇ ve DEĞERLENDİRME (SBED'de bu başlık numarasız)

---

## HER ADIMDA ÇALIŞMA DİSİPLİNİ
1. O adımı yaz → 2. Sayıları KİLİTLİ_SAYILAR.md ile karşılaştır → 
3. SBED formatına uygunluk kontrol → 4. Onayla → 5. Sonraki adım
GERİ DÖNÜŞ YOK: her adım kapanmadan sonrakine geçilmez.
