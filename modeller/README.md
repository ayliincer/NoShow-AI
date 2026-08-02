# Model Dosyaları

Bu klasördeki eğitilmiş model dosyaları (.joblib) GitHub'ın 100MB dosya boyutu
sınırını aştığı için repoya dahil edilmemiştir (bkz. .gitignore).

- `nihai_no_show_model_paketi.joblib` (~117MB) — Nihai şampiyon model (Random Forest, Optimize)
- `nihai_no_show_model_paketi_v1_asiri_uyumlu_yedek.joblib` (~86MB) — Önceki (budanmamış) model, referans/karşılaştırma amaçlı yedek

Modelleri yeniden üretmek için `kodlar/23_regularize_edilmis_rf_optimizasyonu.py` scriptini çalıştırın
(yalnızca `veriler/` klasöründeki eğitim/test CSV'lerine ihtiyaç duyar).

Model dosyalarını paylaşmak isterseniz Git LFS (`git lfs track "*.joblib"`) veya
Zenodo/Hugging Face gibi harici bir model deposu kullanılması önerilir.
