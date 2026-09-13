import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

KOK = Path(__file__).resolve().parent.parent

RF = dict(n_estimators=500, max_depth=25, min_samples_leaf=1, min_samples_split=10,
          max_features=0.5, random_state=42, n_jobs=-1)

GECMIS = ["gecmis_randevu_sayisi", "gecmis_no_show_sayisi", "gecmis_no_show_orani", "ilk_ziyaret_mi"]

HAVA = ["average_temp_day", "average_rain_day", "max_temp_day", "max_rain_day",
        "rainy_day_before", "storm_day_before", "rain_intensity", "heat_intensity",
        "temp_range", "rain_range"]


def naif_metrikler(y_train, y_test):
    prevalans = y_test.mean()
    naif_proba = np.full(len(y_test), prevalans)
    naif_pr_auc = average_precision_score(y_test, naif_proba)
    naif_brier = prevalans * (1 - prevalans)
    return prevalans, naif_pr_auc, naif_brier


def satir_rastgele_sonucu():
    p = joblib.load(KOK / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib")
    model = p["model"]

    def veri_hazirla(dosya):
        d = pd.read_csv(KOK / "veriler" / dosya)
        if "appointment_time" in d.columns:
            d = d.drop(columns=["appointment_time"])
        d["icd_frekans"] = d["icd"].map(p["icd_frekans_haritasi"]).fillna(0.0)
        d = d.drop(columns=["icd"])
        y = d["no_show"].map({"no": 0, "yes": 1}).values
        X = d.drop(columns=["no_show"])
        for c in p["sutun_siralamasi"]:
            if c not in X.columns:
                X[c] = 0
        return X[p["sutun_siralamasi"]], y

    Xtr, ytr = veri_hazirla("medical_appointments_train.csv")
    Xte, yte = veri_hazirla("medical_appointments_test.csv")
    proba_te = model.predict_proba(Xte)[:, 1]

    prevalans, naif_pr_auc, naif_brier = naif_metrikler(ytr, yte)
    return {
        "Bölme Stratejisi": "Satır-rastgele",
        "Taban Oranı": prevalans,
        "Model PR-AUC": average_precision_score(yte, proba_te),
        "Naif PR-AUC": naif_pr_auc,
        "Model Brier": brier_score_loss(yte, proba_te),
        "Naif Brier": naif_brier,
    }


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


def kodla(egitim, test):
    freq = egitim["icd"].value_counts(normalize=True)
    egitim = egitim.copy(); test = test.copy()
    egitim["icd_frekans"] = egitim["icd"].map(freq)
    test["icd_frekans"] = test["icd"].map(freq).fillna(0.0)
    kat = [c for c in ["specialty", "gender", "disability", "city", "appointment_month", "appointment_shift"] if c in egitim.columns]
    egitim = pd.get_dummies(egitim, columns=kat)
    test = pd.get_dummies(test, columns=kat)
    test = test.reindex(columns=egitim.columns, fill_value=0)
    dus = (["no_show", "no_show_bin", "pseudo_id", "_orijinal_sira", "icd", "appointment_date",
            "entry_service_date", "date_of_birth", "appointment_time", "no_show_reason"]
           + HAVA + GECMIS)
    oz = [c for c in egitim.columns if c not in dus]
    Xtr = egitim[oz].select_dtypes(include=[np.number, bool]).astype(float).fillna(0)
    Xte = test[oz].select_dtypes(include=[np.number, bool]).astype(float).fillna(0)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, egitim["no_show_bin"], Xte, test["no_show_bin"]


def rf_egit_ve_degerlendir(egitim, test, bolme_adi, referans_pr_auc=None, referans_brier=None):
    Xtr, ytr, Xte, yte = kodla(egitim, test)
    m = RandomForestClassifier(**RF); m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)[:, 1]
    model_pr_auc = average_precision_score(yte, proba)
    model_brier = brier_score_loss(yte, proba)

    if referans_pr_auc is not None:
        fark = abs(model_pr_auc - referans_pr_auc)
        print(f"  [doğrulama] {bolme_adi}: yeniden hesaplanan PR-AUC={model_pr_auc:.4f} "
              f"vs önceden kaydedilen={referans_pr_auc:.4f} (fark={fark:.4f})")
    if referans_brier is not None:
        fark_b = abs(model_brier - referans_brier)
        print(f"  [doğrulama] {bolme_adi}: yeniden hesaplanan Brier={model_brier:.4f} "
              f"vs önceden kaydedilen={referans_brier:.4f} (fark={fark_b:.4f})")

    prevalans, naif_pr_auc, naif_brier = naif_metrikler(ytr.values, yte.values)
    return {
        "Bölme Stratejisi": bolme_adi,
        "Taban Oranı": prevalans,
        "Model PR-AUC": model_pr_auc,
        "Naif PR-AUC": naif_pr_auc,
        "Model Brier": model_brier,
        "Naif Brier": naif_brier,
    }



def hasta_gruplu_sonucu():
    df = pd.read_csv(KOK / "veriler" / "step02_pseudo_gecmis_duzeltilmis.csv", low_memory=False)
    df = ozellik_uret(df)
    df["no_show_bin"] = (df["no_show"] == "yes").astype(int)

    gruplar = df["pseudo_id"].copy()
    nan_maske = gruplar.isna()
    gruplar = gruplar.astype("object")
    gruplar[nan_maske] = ["__tekil_%d" % i for i in range(nan_maske.sum())]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    i_tr, i_te = next(gss.split(df, df["no_show_bin"], groups=gruplar))
    egitim, test = df.iloc[i_tr].copy(), df.iloc[i_te].copy()

    return rf_egit_ve_degerlendir(egitim, test, "Hasta-gruplu",
                                   referans_pr_auc=0.1946, referans_brier=0.0823)



def kronolojik_sonucu():
    df = pd.read_csv(KOK / "veriler" / "step02_pseudo_gecmis_duzeltilmis.csv", low_memory=False)
    df = ozellik_uret(df)
    df["no_show_bin"] = (df["no_show"] == "yes").astype(int)

    egitim = df[df["appointment_year"] <= 2020].copy()
    test = df[df["appointment_year"] >= 2021].copy()

    return rf_egit_ve_degerlendir(egitim, test, "Kronolojik",
                                   referans_pr_auc=0.186879, referans_brier=0.140897)


def main():
    print("=" * 100)
    print("ÜÇ SENARYO NAİF PREVALANS BASELINE KARŞILAŞTIRMASI")
    print("=" * 100)

    sonuc = []
    print("\n(1) Satır-rastgele...")
    sonuc.append(satir_rastgele_sonucu())
    print("(2) Hasta-gruplu...")
    sonuc.append(hasta_gruplu_sonucu())
    print("(3) Kronolojik...")
    sonuc.append(kronolojik_sonucu())

    df_sonuc = pd.DataFrame(sonuc)
    df_sonuc["Kazanç Katsayısı"] = df_sonuc["Model PR-AUC"] / df_sonuc["Naif PR-AUC"]
    df_sonuc = df_sonuc[["Bölme Stratejisi", "Taban Oranı", "Model PR-AUC", "Naif PR-AUC",
                          "Kazanç Katsayısı", "Model Brier", "Naif Brier"]]
    df_sonuc_yuvarlanmis = df_sonuc.copy()
    for c in df_sonuc_yuvarlanmis.columns[1:]:
        df_sonuc_yuvarlanmis[c] = df_sonuc_yuvarlanmis[c].round(4)

    print("\n" + "=" * 100)
    print("SONUÇ TABLOSU")
    print("=" * 100)
    print(df_sonuc_yuvarlanmis.to_string(index=False))

    cikti = KOK / "veriler" / "baseline_ve_overfitting_uc_senaryo.csv"
    df_sonuc_yuvarlanmis.to_csv(cikti, index=False, encoding="utf-8-sig")
    print(f"\n-> Kaydedildi: {cikti.name}")


if __name__ == "__main__":
    main()