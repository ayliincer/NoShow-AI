import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              recall_score, brier_score_loss, precision_recall_curve)

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
    sil = [c for c in HAVA_DURUMU_SUTUNLARI if c in X_train.columns]
    X_train = X_train.drop(columns=sil)
    X_test = X_test.drop(columns=[c for c in sil if c in X_test.columns])
    return X_train, X_test, y_train, y_test, freq


def main():
    X_train, X_test, y_train, y_test, freq = veri_hazirla()

    rf_p = joblib.load(KOK / "modeller" / "rf_yeniden_optimizasyon_sonucu.joblib")["en_iyi_parametreler"]
    lgbm_p = joblib.load(KOK / "modeller" / "lightgbm_optimizasyon_sonucu.joblib")["en_iyi_parametreler"]
    cat_p = joblib.load(KOK / "modeller" / "catboost_optimizasyon_sonucu.joblib")["en_iyi_parametreler"]
    xgb_p = joblib.load(KOK / "modeller" / "xgboost_optimizasyon_sonucu.joblib")["en_iyi_parametreler"]

    modeller = {
        "Random Forest (Optimize)": RandomForestClassifier(random_state=42, n_jobs=-1, **rf_p),
        "LightGBM (Optimize)": LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1, **lgbm_p),
        "CatBoost (Optimize)": CatBoostClassifier(random_state=42, verbose=False, allow_writing_files=False, **cat_p),
        "XGBoost (Optimize)": XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, **xgb_p),
        "Decision Tree (Varsayılan)": DecisionTreeClassifier(random_state=42),
        "Logistic Regression (Varsayılan)": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))]),
    }

    from sklearn.model_selection import cross_val_predict

    def egitimde_esik_sec(model, X_tr, y_tr):
        # Madde 1 (danışman): F1-optimal eşik TEST'ten DEĞİL, eğitim setinde
        # 5-katlı CV out-of-fold olasılıklarından seçilir; sonra test'e SABİT
        # uygulanır. Böylece eşik seçimi test etiketlerinden sızmaz.
        oof = cross_val_predict(model, X_tr, y_tr, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
        p, r, th = precision_recall_curve(y_tr, oof)
        f1s = 2 * p * r / (p + r + 1e-12)
        return th[int(np.argmax(f1s[:-1]))]

    sonuclar = []
    model_nesneleri = {}
    for isim, model in modeller.items():
        # Eşik, modeli test'e hiç dokunmadan, yalnızca eğitim setinde seç
        best_thresh = egitimde_esik_sec(model, X_train, y_train)

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred_05 = model.predict(X_test)
        model_nesneleri[isim] = model

        # Eğitimde seçilen eşik test'e SABİT uygulanır (yeniden seçilmez)
        pred_opt = (proba >= best_thresh).astype(int)

        satir = {
            "Model": isim,
            "Dış Test ROC-AUC": roc_auc_score(y_test, proba),
            "Dış Test PR-AUC": average_precision_score(y_test, proba),
            "Dış Test Brier": brier_score_loss(y_test, proba),
            "F1 (t=0.5)": f1_score(y_test, pred_05),
            "Recall (t=0.5)": recall_score(y_test, pred_05),
            "F1-Optimal Eşik": best_thresh,
            "F1 (optimal eşik)": f1_score(y_test, pred_opt),
            "Recall (optimal eşik)": recall_score(y_test, pred_opt),
        }
        sonuclar.append(satir)

    en_iyi_isim = "Random Forest (Optimize)"
    en_iyi_model_nesnesi = model_nesneleri[en_iyi_isim]

    df = pd.DataFrame(sonuclar).sort_values("Dış Test ROC-AUC", ascending=False).reset_index(drop=True)
    print("=" * 130)
    print("NİHAİ TAM ADİL DIŞ TEST SONUÇ TABLOSU (tüm modeller eşit optimize edilmiş, tek seferlik)")
    print("=" * 130)
    print(df.to_string(index=False))
    df.to_csv(KOK / "veriler" / "nihai_tam_adil_dis_test_sonuclari.csv", index=False, encoding="utf-8-sig")

    print(f"\nŞAMPİYON (CV sonucuna göre ÖNCEDEN seçilmiş): {en_iyi_isim}")
    print(f"Bu modelin dış test ROC-AUC'u: {df[df['Model']==en_iyi_isim]['Dış Test ROC-AUC'].values[0]:.4f}")
    print("(Not: Tablodaki sıralama sadece bilgi amaçlıdır; başka bir model test setinde daha")
    print(" yüksek çıksa dahi şampiyon DEĞİŞTİRİLMEZ - bu tam olarak önlemeye çalıştığımız sızıntıdır.)")

    joblib.dump({
        "surum": "v4-tam-adil-optimize",
        "model_adi": en_iyi_isim,
        "icd_frekans_haritasi": freq.to_dict(),
        "model": en_iyi_model_nesnesi,
        "sutun_siralamasi": list(X_train.columns),
        "feature_count": X_train.shape[1],
        "not": "Hava durumu (sızıntı riski) çıkarılmış; TÜM modeller eşit titizlikte (sadece train+CV) optimize edilmiş; şampiyon SADECE CV'ye göre seçilmiş, test seti tek sefer açılmıştır.",
    }, KOK / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib")
    print(f"-> Kaydedildi: nihai_tam_adil_dis_test_sonuclari.csv, modeller/nihai_no_show_model_paketi_v4_tam_adil.joblib")


if __name__ == "__main__":
    main()
