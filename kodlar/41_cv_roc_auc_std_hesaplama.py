import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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
    print(f"Öznitelik sayısı (hava durumu hariç): {X.shape[1]}")

    rf_p = joblib.load(KOK / "modeller" / "rf_yeniden_optimizasyon_sonucu.joblib")["en_iyi_parametreler"]
    lgbm_p = joblib.load(KOK / "modeller" / "lightgbm_optimizasyon_sonucu.joblib")["en_iyi_parametreler"]
    cat_p = joblib.load(KOK / "modeller" / "catboost_optimizasyon_sonucu.joblib")["en_iyi_parametreler"]
    xgb_p = joblib.load(KOK / "modeller" / "xgboost_optimizasyon_sonucu.joblib")["en_iyi_parametreler"]

    modeller = {
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1, **rf_p),
        "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1, **lgbm_p),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, **xgb_p),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=False, allow_writing_files=False, **cat_p),
        "Lojistik Regresyon": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Karar Ağacı": DecisionTreeClassifier(random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=" * 100)
    print("5 KATLI CV ROC-AUC (ORTALAMA ± STD) - SADECE EĞİTİM SETİ, SIZINTISIZ")
    print("=" * 100)

    sonuclar = []
    for isim, model in modeller.items():
        skorlar = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        ortalama = skorlar.mean()
        std = skorlar.std()
        print(f"{isim:22s} | katlar: {np.round(skorlar, 4)} | ortalama={ortalama:.4f} | std={std:.4f}")
        sonuclar.append({
            "Model": isim,
            "CV ROC-AUC (ortalama)": round(ortalama, 4),
            "CV ROC-AUC (std)": round(std, 4),
            "Fold 1": round(skorlar[0], 4),
            "Fold 2": round(skorlar[1], 4),
            "Fold 3": round(skorlar[2], 4),
            "Fold 4": round(skorlar[3], 4),
            "Fold 5": round(skorlar[4], 4),
        })

    df = pd.DataFrame(sonuclar).sort_values("CV ROC-AUC (ortalama)", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 100)
    print(df.to_string(index=False))

    cikti_yolu = KOK / "veriler" / "cv_roc_auc_std_sonuclari.csv"
    df.to_csv(cikti_yolu, index=False, encoding="utf-8-sig")
    print(f"\n-> Kaydedildi: {cikti_yolu}")


if __name__ == "__main__":
    main()
