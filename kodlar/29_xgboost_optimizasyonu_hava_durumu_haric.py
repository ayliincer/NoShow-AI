import time
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
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
    return X, y


def main():
    X, y = veri_hazirla()
    print(f"Öznitelik sayısı: {X.shape[1]}")

    arama_uzayi = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
        "min_child_weight": [1, 5, 10],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    taban = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)

    arama = RandomizedSearchCV(
        taban, arama_uzayi, n_iter=20, scoring="roc_auc", cv=cv,
        random_state=42, n_jobs=1, refit=True, verbose=1,
    )
    print("=" * 100)
    print("SIZINTISIZ XGBoost HİPERPARAMETRE ARAMASI (SADECE EĞİTİM SETİ)")
    print("=" * 100)
    t0 = time.time()
    arama.fit(X, y)
    print(f"Süre: {time.time()-t0:.1f}sn | En iyi CV ROC-AUC: {arama.best_score_:.4f}")
    print("En iyi parametreler:", arama.best_params_)

    import joblib
    joblib.dump({"en_iyi_parametreler": arama.best_params_, "en_iyi_cv_roc_auc": arama.best_score_},
                KOK / "modeller" / "xgboost_optimizasyon_sonucu.joblib")
    print("-> Kaydedildi: modeller/xgboost_optimizasyon_sonucu.joblib")


if __name__ == "__main__":
    main()
