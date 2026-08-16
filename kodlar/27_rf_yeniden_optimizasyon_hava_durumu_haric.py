import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

KOK = Path(__file__).resolve().parent.parent
HAVA_DURUMU_SUTUNLARI = [
    "average_temp_day", "average_rain_day", "max_temp_day", "max_rain_day",
    "rainy_day_before", "storm_day_before",
    "average_temp_day_nan", "average_rain_day_nan", "max_temp_day_nan", "max_rain_day_nan",
    "temp_range", "rain_range", "is_rainy",
    "rain_intensity_heavy", "rain_intensity_moderate", "rain_intensity_no_rain", "rain_intensity_weak",
    "heat_intensity_cold", "heat_intensity_heavy_cold", "heat_intensity_heavy_warm",
    "heat_intensity_mild", "heat_intensity_warm",
]


def veri_hazirla():
    train = pd.read_csv(KOK / "veriler" / "medical_appointments_train.csv")
    if "appointment_time" in train.columns:
        train.drop(columns=["appointment_time"], inplace=True)
    freq = train["icd"].value_counts(normalize=True)
    train["icd_frekans"] = train["icd"].map(freq)
    train.drop(columns=["icd"], inplace=True)
    y = train["no_show"].map({"no": 0, "yes": 1})
    X = train.drop(columns=["no_show"])
    X = X.drop(columns=[c for c in HAVA_DURUMU_SUTUNLARI if c in X.columns])
    return X, y, freq


def main():
    X, y, freq = veri_hazirla()
    print(f"Öznitelik sayısı (hava durumu hariç): {X.shape[1]}")
    
    arama_uzayi = {
        "n_estimators": [200, 300, 500],
        "max_depth": [15, 20, 25, 30, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split": [2, 5, 10],
        "max_features": [0.2, 0.3, 0.4, 0.5, 0.6, "sqrt"],
        "class_weight": [None],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    taban = RandomForestClassifier(random_state=42, n_jobs=-1)

    arama = RandomizedSearchCV(
        taban, arama_uzayi, n_iter=25, scoring="roc_auc", cv=cv,
        random_state=42, n_jobs=1, refit=True, verbose=1,
    )
    print("=" * 100)
    print("SIZINTISIZ RANDOM FOREST HİPERPARAMETRE ARAMASI (YENİ ÖZNİTELİK KÜMESİ, SADECE EĞİTİM SETİ)")
    print("=" * 100)
    t0 = time.time()
    arama.fit(X, y)
    print(f"Süre: {time.time()-t0:.1f}sn | En iyi CV ROC-AUC: {arama.best_score_:.4f}")
    print("En iyi parametreler:", arama.best_params_)
    print(f"\nKarşılaştırma: LightGBM (script 27) CV ROC-AUC = 0.7350")
    print(f"               Varsayılan RF (script 26) CV ROC-AUC = 0.7043")

    joblib.dump({
        "en_iyi_parametreler": arama.best_params_,
        "en_iyi_cv_roc_auc": arama.best_score_,
        "ozellik_sirasi": list(X.columns),
        "icd_frekans_haritasi": freq.to_dict(),
    }, KOK / "modeller" / "rf_yeniden_optimizasyon_sonucu.joblib")
    print("-> Kaydedildi: modeller/rf_yeniden_optimizasyon_sonucu.joblib")


if __name__ == "__main__":
    main()
