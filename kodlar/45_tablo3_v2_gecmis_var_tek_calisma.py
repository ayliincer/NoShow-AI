import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

KOK = Path(__file__).resolve().parent.parent

RF = dict(n_estimators=500, max_depth=25, min_samples_leaf=1, min_samples_split=10,
          max_features=0.5, random_state=42, n_jobs=-1)

GECMIS = ["gecmis_randevu_sayisi", "gecmis_no_show_sayisi", "gecmis_no_show_orani", "ilk_ziyaret_mi"]

HAVA = ["average_temp_day", "average_rain_day", "max_temp_day", "max_rain_day",
        "rainy_day_before", "storm_day_before", "rain_intensity", "heat_intensity",
        "temp_range", "rain_range"]


def ozellik_uret(v):
    v = v.copy()
    v["appointment_date"] = pd.to_datetime(v["appointment_date"], errors="coerce")
    v["entry_service_date"] = pd.to_datetime(v["entry_service_date"], errors="coerce")
    v["lead_time"] = ((v["appointment_date"] - v["entry_service_date"]).dt.days).clip(lower=0)
    v["appointment_day_of_week"] = v["appointment_date"].dt.dayofweek
    v["appointment_hour"] = pd.to_datetime(v["appointment_time"], format="%H:%M", errors="coerce").dt.hour
    v["lead_time"] = v["lead_time"].fillna(v["lead_time"].median())
    v["appointment_hour"] = v["appointment_hour"].fillna(v["appointment_hour"].median())
    v["appointment_day_of_week"] = v["appointment_day_of_week"].fillna(0)
    return v


def kodla(egitim, test, gecmis_dahil):
    freq = egitim["icd"].value_counts(normalize=True)
    egitim = egitim.copy(); test = test.copy()
    egitim["icd_frekans"] = egitim["icd"].map(freq)
    test["icd_frekans"] = test["icd"].map(freq).fillna(0.0)
    kat = [c for c in ["specialty", "gender", "disability", "city", "appointment_month", "appointment_shift"] if c in egitim.columns]
    egitim = pd.get_dummies(egitim, columns=kat)
    test = pd.get_dummies(test, columns=kat)
    test = test.reindex(columns=egitim.columns, fill_value=0)
    dus = (["no_show", "no_show_bin", "pseudo_id", "_orijinal_sira", "icd", "appointment_date",
            "entry_service_date", "date_of_birth", "appointment_time", "no_show_reason"] + HAVA)
    if not gecmis_dahil:
        dus += GECMIS
    oz = [c for c in egitim.columns if c not in dus]
    Xtr = egitim[oz].select_dtypes(include=[np.number, bool]).astype(float).fillna(0)
    Xte = test[oz].select_dtypes(include=[np.number, bool]).astype(float).fillna(0)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, egitim["no_show_bin"], Xte, test["no_show_bin"]


def bol_hasta_gruplu(df, random_state):
    gruplar = df["pseudo_id"].copy()
    nan_maske = gruplar.isna()
    gruplar = gruplar.astype("object")
    gruplar[nan_maske] = ["__tekil_%d" % i for i in range(nan_maske.sum())]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    i_tr, i_te = next(gss.split(df, df["no_show_bin"], groups=gruplar))
    return df.iloc[i_tr].copy(), df.iloc[i_te].copy()


def degerlendir(df, gecmis_dahil, random_state):
    egitim, test = bol_hasta_gruplu(df, random_state)
    Xtr, ytr, Xte, yte = kodla(egitim, test, gecmis_dahil)
    m = RandomForestClassifier(**RF); m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)[:, 1]
    rng = np.random.default_rng(random_state)
    boots = []
    yte_arr = yte.values
    for _ in range(200):
        idx = rng.integers(0, len(yte_arr), len(yte_arr))
        if len(np.unique(yte_arr[idx])) < 2:
            continue
        boots.append(roc_auc_score(yte_arr[idx], proba[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"ROC-AUC": roc_auc_score(yte, proba), "GA_alt": lo, "GA_ust": hi,
            "PR-AUC": average_precision_score(yte, proba),
            "Brier": brier_score_loss(yte, proba), "N_test": len(yte),
            "Test_Hasta_Sayisi": test["pseudo_id"].nunique()}


def main():
    df = pd.read_csv(KOK / "veriler" / "step02_pseudo_gecmis_duzeltilmis.csv", low_memory=False)
    df = ozellik_uret(df)
    df["no_show_bin"] = (df["no_show"] == "yes").astype(int)

    sonuc = []
    for gecmis_dahil, getiket in [(False, "geçmiş YOK"), (True, "geçmiş VAR")]:
        m = degerlendir(df, gecmis_dahil, random_state=42)
        sonuc.append({"Bölme": "Hasta-gruplu (düzeltilmiş pseudo_id)", "Geçmiş": getiket, "random_state": 42, **m})
        print(f"{getiket:10s} | ROC-AUC={m['ROC-AUC']:.4f} [{m['GA_alt']:.4f}-{m['GA_ust']:.4f}] "
              f"PR-AUC={m['PR-AUC']:.4f} Brier={m['Brier']:.4f} N_test={m['N_test']} "
              f"Test_Hasta={m['Test_Hasta_Sayisi']}")

    df_out = pd.DataFrame(sonuc)
    cikti = KOK / "veriler" / "tablo3_v2_hasta_gruplu_gecmis_var_yok_seed42.csv"
    df_out.to_csv(cikti, index=False, encoding="utf-8-sig")
    print(f"\n-> Kaydedildi (YENİ dosya): {cikti.name}")


if __name__ == "__main__":
    main()
