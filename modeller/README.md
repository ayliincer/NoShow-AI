# Model Dosyaları

Bu klasör, pipeline'ın çalışması için gerekli modelleri içerir.

## Dahil edilen (gerekli) modeller
- **nihai_no_show_model_paketi_v4_tam_adil.joblib** — RESMİ ŞAMPİYON. Random Forest
  (Optimize), hava durumsuz (56 öznitelik), yalnızca CV ile seçilmiş, dış test
  ROC-AUC=0,775. TÜM downstream script'ler (SHAP, simülasyon, pipeline) bunu yükler.
- **{rf,lightgbm,catboost,xgboost}_optimizasyon_sonucu.joblib** — Her modelin
  hiperparametre optimizasyon sonuçları (script 30'un girdisi).

## Dahil edilmeyen (yeniden üretilebilir) modeller
Aşağıdakiler boyut nedeniyle pakete alınmamıştır; ilgili script çalıştırılınca
otomatik yeniden üretilir:
- nihai_no_show_model_paketi_pipeline_STANDALONE.joblib  -> script 31 üretir
- nihai_no_show_model_paketi_v5_pipeline.joblib          -> script 31 üretir
- Eski yedekler (v1/v2/v3 ve eski scaler'lı paket) — artık hiçbir kod kullanmaz.

## Sıfırdan yeniden üretim
Tüm modelleri sıfırdan üretmek için (kodlar/PIPELINE_SIRASI.md'deki sıra):
25 → 26/27/28/29 (optimizasyonlar) → 30 (şampiyon v4 + dış test) → 31 (pipeline).
