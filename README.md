# NoShow-AI
TÜBİTAK 2209-A - AI-based Patient No-Show Prediction System
# Explainable AI-Based Dynamic No-Show Prediction System

## Overview

This project presents an explainable machine learning pipeline for predicting outpatient appointment no-shows using electronic healthcare records.

The pipeline includes:

- Exploratory Data Analysis (EDA)
- Missing Value Analysis
- Duplicate Record Analysis
- Outlier Detection
- Descriptive Statistics
- Correlation Analysis
- Feature Engineering
- Missing Data Imputation
- Categorical Encoding
- Model Development
- Cross Validation
- External Validation
- Explainable Artificial Intelligence (SHAP)
- Model Serialization

The project follows a leakage-aware machine learning workflow to ensure scientific validity.

---

## Dataset

**Primary Dataset (actually used throughout the pipeline) — KAYNAK ÇÖZÜLDÜ:**

- File: `veriler/medical-appointments-no-show-en.csv` → `medical_appointments_preprocessed_step01/02.csv` → `medical_appointments_train/test.csv`
- ~49.593 ham randevu kaydı (2016–2022); specialty (fizyoterapi/psikoterapi/konuşma terapisi vb.), city (13 Brezilya belediyesi), disability status, ICD kodu, hava durumu (sıcaklık/yağış) değişkenleri, appointment year içerir.
- **Kaynak (doğrulandı, web araştırmasıyla tespit edildi):** Bu veri seti, Universidade do Vale do Itajaí (UNIVALI) **Fiziksel ve Zihinsel Rehabilitasyon Uzmanlık Merkezi'nin (CER)** hasta randevu kayıtlarına dayanmaktadır. Orijinal 4.812 kayıtlık pilot veri seti ve metodoloji şu makalede yayınlanmıştır:

  > Salazar, L.H.A.; Leithardt, V.R.Q.; Parreira, W.D.; da Rocha Fernandes, A.M.; Barbosa, J.L.V.; Correia, S.D. (2022). "Application of Machine Learning Techniques to Predict a Patient's No-Show in the Healthcare Sector." *Future Internet*, 14(1), 3. https://doi.org/10.3390/fi14010003

  Bu makale, projenin başındaki TÜBİTAK metnindeki gizemli **"4.812"** rakamının kaynağıdır (makalede: *"4812 medical records from an electronic spreadsheet of 2017 and 2019"*). Sütun yapısı (disability type, entry-into-service date, city, ICD, INMET meteorolojik verisi — Itajaí şehri için) bizim veri setimizle birebir örtüşmektedir. Makalenin "Gelecek Çalışma" bölümü, *"güney Brezilya'daki tüm halk sağlığı sistemi randevularını kapsayan daha büyük bir veri seti toplama sürecinde olduklarını"* belirtmektedir — bizim 2016–2022 dönemini kapsayan genişletilmiş (49.593 satırlık) veri setimiz büyük olasılıkla bu takip çalışmasıdır.
- **Etik kurul onayı:** Orijinal (2019) CER veri seti için Univali Etik Kurulu onayı mevcuttur (karar no 4270.234, LGPD/GDPR uyumlu). Genişletilmiş (2016–2022) veri setinin aynı/güncellenmiş bir etik onay kapsamında olup olmadığı makale yazarları tarafından teyit edilmeli ve makalede belirtilmelidir.
- **Not:** Hasta kimliği (patient ID) bu veri setinde yoktur, bu nedenle hasta bazlı geçmiş öznitelikler (örn. önceki no-show sayısı) türetilememektedir — bkz. kronolojik genelleme bulgusu.

**`KaggleV2-May-2016.csv` (repoda duruyor ama pipeline'da KULLANILMIYOR):**

- Bu, iyi bilinen halka açık Kaggle "Medical Appointment No Shows" veri setidir (Vitória, Espírito Santo, Brezilya), 110.527 satır, sadece Nisan–Haziran 2016'yı kapsar. **Bizim projemizle ilgisi yoktur** — farklı bir şehir/sistem, farklı sütun yapısı. Salazar ve ark. (2022) makalesi de bu Kaggle veri setinden AYRI, kendi topladıkları bir veri seti kullandıklarını açıkça belirtmektedir (makalede: *"the lack of information about how [Kaggle] data was pre-processed... we have opted to collect the dataset on our own"*).

**Bilinen sınırlama (kritik):** Birincil veri seti 2016–2022'yi kapsamakta ve zaman içinde belirgin bir no-show oranı artışı göstermektedir (~%4'ten ~%20'ye). Gerçek kronolojik (ileriye dönük) doğrulama, eğitilen modelin ayırt edicilik gücünün eğitim penceresi boyutundan bağımsız olarak neredeyse şans seviyesine (ROC-AUC ≈ 0.53) düştüğünü göstermektedir (bkz. `kodlar/25_kronolojik_genelleme_analizi.py` ve `veriler/kronolojik_genisleyen_pencere_dogrulama.csv`). Bu repodaki diğer yerlerde raporlanan ROC-AUC=0.7753 değeri, tabakalı RASTGELE train/test bölünmesini yansıtır ve prospektif/dağıtım geçerliliğinin kanıtı olarak yorumlanmamalıdır.

---

## Machine Learning Models

The following algorithms are evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

---

## Explainable AI

Model interpretation is performed using SHAP.

Generated explanations include

- SHAP Summary Plot
- SHAP Bar Plot

---

## Project Structure

```
NoShow-AI
│
├── kodlar/
├── veriler/
├── modeller/
├── gorseller/
├── makale/
├── README.md
└── requirements.txt
```

---

## Installation

```bash
git clone <repository>

cd NoShow-AI

pip install -r requirements.txt
```

---

## Running

Execute the Python scripts sequentially.

```
01_veri_seti_genel_bakis.py

↓

02_eksik_veri_ve_tekrarlayan_kayitlar.py

↓

...

↓

20_modeli_diske_kaydet.py
```

---

## Output

The project produces

- Cleaned datasets
- Feature engineered datasets
- Cross-validation results
- External validation results
- SHAP visualizations
- Serialized production-ready model

---

## Author

Aylin Cer
Beyza Nur Dinçer

Management Information Systems (MIS)

Karadeniz Technical University

---

## License

This project is developed for academic research purposes.
---

## ÖNEMLİ GÜNCELLEME (Danışman Değerlendirmesi Sonrası)

Bu projenin model seçim süreci, bir danışman/hoca değerlendirmesi sonrası ciddi
metodolojik sorunlar nedeniyle **baştan kurulmuştur**. Özet:

- **Eski süreç (ARTIK GEÇERSİZ):** Model seçimi 6 algoritmayı doğrudan saklı dış test
  setinde karşılaştırıp en iyisini seçiyordu (test seti sızıntısı); hava durumu
  değişkenleri (zamansal sızıntı riski taşıyan 22 öznitelik) modelde kalmıştı;
  modeller birbirinden farklı (tutarsız) dengesizlik stratejileriyle eğitiliyordu.
- **Yeni süreç:** Model seçimi SADECE eğitim seti + 5 katlı CV ile yapılıyor
  (`kodlar/26-32` scriptleri); hava durumu değişkenleri tamamen çıkarılmış;
  tüm modeller aynı koşulda karşılaştırılmıştır. Saklı dış test seti yalnızca
  TEK SEFER, seçim tamamlandıktan sonra açılmıştır.
- **Güncel şampiyon:** Random Forest (yeniden optimize edilmiş), Dış Test
  ROC-AUC=0,7753. Detaylar ve tam gerekçe için bkz.
  `raporlar/Bulgular_v2_Duzeltilmis_DanismanElestirisi.docx` — bu belge,
  önceki `raporlar/Makale_Bulgular_Bolumu.docx` belgesinin YERİNE geçer.
- **Değişmeyen kritik bulgu:** Modelin gerçek kronolojik (ileri yönlü) doğrulamada
  yazı-tura seviyesine (ROC-AUC≈0,53) düştüğü bulgusu, yeni modelle de aynen
  geçerlidir (bkz. `kodlar/25_kronolojik_genelleme_analizi.py`).
