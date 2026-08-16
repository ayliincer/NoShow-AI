import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
    test = pd.read_csv(KOK / "veriler" / "medical_appointments_test.csv")
    for df in (train, test):
        if "appointment_time" in df.columns:
            df.drop(columns=["appointment_time"], inplace=True)
    freq = train["icd"].value_counts(normalize=True)
    train["icd_frekans"] = train["icd"].map(freq)
    test["icd_frekans"] = test["icd"].map(freq).fillna(0.0)
    train.drop(columns=["icd"], inplace=True)
    test.drop(columns=["icd"], inplace=True)
    y_train = train["no_show"].map({"no": 0, "yes": 1})
    y_test = test["no_show"].map({"no": 0, "yes": 1})
    X_train = train.drop(columns=["no_show"])
    X_test = test.drop(columns=["no_show"])

    silinecek = [c for c in HAVA_DURUMU_SUTUNLARI if c in X_train.columns]
    print(f"Çıkarılan hava durumu öznitelik sayısı: {len(silinecek)} / kalan: {X_train.shape[1]-len(silinecek)}")
    X_train = X_train.drop(columns=silinecek)
    X_test = X_test.drop(columns=[c for c in silinecek if c in X_test.columns])
    return X_train, X_test, y_train, y_test, freq


def main():
    X_train, X_test, y_train, y_test, freq = veri_hazirla()

    modeller = {
        "Logistic Regression": (StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "Decision Tree": (None, DecisionTreeClassifier(random_state=42)),
        "Random Forest": (None, RandomForestClassifier(random_state=42, n_jobs=-1)),
        "XGBoost": (None, XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)),
        "LightGBM": (None, LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1)),
        "CatBoost": (None, CatBoostClassifier(random_state=42, verbose=False, allow_writing_files=False)),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrikler = ["roc_auc", "average_precision", "f1", "recall", "neg_brier_score"]

    print("=" * 100)
    print("ADİL SIZINTISIZ CV KARŞILAŞTIRMASI (Hava durumu çıkarılmış, tutarlı koşul: dengeleme yok)")
    print("=" * 100)
    sonuclar = []
    for isim, (scaler, model) in modeller.items():
        if scaler is not None:
            from sklearn.pipeline import Pipeline
            pipe = Pipeline([("scaler", scaler), ("model", model)])
        else:
            pipe = model
        skorlar = cross_validate(pipe, X_train, y_train, cv=cv, scoring=metrikler, n_jobs=1)
        satir = {"Model": isim}
        for m in metrikler:
            ort, std = skorlar[f"test_{m}"].mean(), skorlar[f"test_{m}"].std()
            ad = "Brier Score" if m == "neg_brier_score" else m
            val = -ort if m == "neg_brier_score" else ort
            satir[ad] = f"{val:.4f} (+/-{std:.4f})"
        sonuclar.append(satir)
        print(f"{isim:22s} ROC-AUC={satir['roc_auc']}  PR-AUC={satir['average_precision']}  Brier={satir['Brier Score']}")

    df = pd.DataFrame(sonuclar)
    df["roc_auc_sayisal"] = df["roc_auc"].str.extract(r"([0-9.]+)").astype(float)
    df = df.sort_values("roc_auc_sayisal", ascending=False).drop(columns="roc_auc_sayisal").reset_index(drop=True)
    df.to_csv(KOK / "veriler" / "adil_cv_karsilastirma_hava_durumu_haric.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("ADİL CV ŞAMPİYONU (bu sonuca göre seçilmelidir, test setine HENÜZ bakılmadı)")
    print("=" * 100)
    print(df.iloc[0])
    print(f"\n-> Kaydedildi: adil_cv_karsilastirma_hava_durumu_haric.csv")


if __name__ == "__main__":
    main()
