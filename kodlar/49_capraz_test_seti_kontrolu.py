import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

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
           + HAVA + GECMIS)  # geçmiş öznitelik YOK sabit
    oz = [c for c in egitim.columns if c not in dus]
    Xtr = egitim[oz].select_dtypes(include=[np.number, bool]).astype(float).fillna(0)
    Xte = test[oz].select_dtypes(include=[np.number, bool]).astype(float).fillna(0)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, egitim["no_show_bin"], Xte, test["no_show_bin"]


def main():
    df = pd.read_csv(KOK / "veriler" / "step02_pseudo_gecmis_duzeltilmis.csv", low_memory=False)
    df = ozellik_uret(df)
    df["no_show_bin"] = (df["no_show"] == "yes").astype(int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    i_tr, i_te = next(sss.split(df, df["no_show_bin"]))
    egitim_satir = df.iloc[i_tr].copy()

    gruplar = df["pseudo_id"].copy()
    nan_maske = gruplar.isna()
    gruplar = gruplar.astype("object")
    gruplar[nan_maske] = ["__tekil_%d" % i for i in range(nan_maske.sum())]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    i_tr2, i_te2 = next(gss.split(df, df["no_show_bin"], groups=gruplar))
    egitim_hasta = df.iloc[i_tr2].copy()
    egitim_kron = df[df["appointment_year"] <= 2020].copy()
    test_kron_sabit = df[df["appointment_year"] >= 2021].copy()
    print(f"Sabit (ortak) test seti: N={len(test_kron_sabit)}, "
          f"taban oranı={test_kron_sabit['no_show_bin'].mean():.4f}")

    egitimler = [
        ("Satır-rastgele", egitim_satir),
        ("Hasta-gruplu", egitim_hasta),
        ("Kronolojik", egitim_kron),
    ]

    print("\n" + "=" * 100)
    print("ÜÇ EĞİTİM REJİMİ, AYNI (KRONOLOJİK) TEST SETİNDE DEĞERLENDİRME")
    print("=" * 100)

    sonuc = []
    for ad, egitim in egitimler:
        Xtr, ytr, Xte, yte = kodla(egitim, test_kron_sabit)
        m = RandomForestClassifier(**RF)
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        roc = roc_auc_score(yte, proba)
        pr = average_precision_score(yte, proba)
        sonuc.append({
            "Eğitim Tasarımı": ad,
            "Eğitim N": len(egitim),
            "Aynı Test Setinde ROC-AUC": round(roc, 4),
            "Aynı Test Setinde PR-AUC": round(pr, 4),
        })
        print(f"{ad:16s} (eğitim N={len(egitim):6d}) -> ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")

    df_sonuc = pd.DataFrame(sonuc)


    kendi_test_seti = {
        "Satır-rastgele": {"ROC-AUC": 0.7753, "PR-AUC": 0.2970, "Test N": 9329, "Taban Oranı": 0.0999},
        "Hasta-gruplu": {"ROC-AUC": 0.6772, "PR-AUC": 0.1946, "Test N": 9728, "Taban Oranı": 0.0924},
        "Kronolojik": {"ROC-AUC": 0.5480, "PR-AUC": 0.1869, "Test N": len(test_kron_sabit), "Taban Oranı": 0.1604},
    }
    karsilastirma = []
    for row in sonuc:
        ad = row["Eğitim Tasarımı"]
        kts = kendi_test_seti[ad]
        karsilastirma.append({
            "Eğitim Tasarımı": ad,
            "Kendi Test Setinde ROC-AUC": kts["ROC-AUC"],
            "Kendi Test N / Taban Oranı": f"{kts['Test N']} / {kts['Taban Oranı']:.4f}",
            "Aynı (Kronolojik) Test Setinde ROC-AUC": row["Aynı Test Setinde ROC-AUC"],
            "Fark (Aynı Test - Kendi Test)": round(row["Aynı Test Setinde ROC-AUC"] - kts["ROC-AUC"], 4),
            "Kendi Test Setinde PR-AUC": kts["PR-AUC"],
            "Aynı Test Setinde PR-AUC": row["Aynı Test Setinde PR-AUC"],
        })
    df_karsilastirma = pd.DataFrame(karsilastirma)

    print("\n" + "=" * 100)
    print("KARŞILAŞTIRMA: KENDİ TEST SETİ vs ORTAK (KRONOLOJİK) TEST SETİ")
    print("=" * 100)
    print(df_karsilastirma.to_string(index=False))

    print("\n" + "-" * 100)
    print("YORUM İÇİN VERİ (yorum eklenmedi, sadece rakamlar):")
    satir_fark = df_karsilastirma.loc[df_karsilastirma["Eğitim Tasarımı"] == "Satır-rastgele",
                                       "Fark (Aynı Test - Kendi Test)"].values[0]
    hasta_fark = df_karsilastirma.loc[df_karsilastirma["Eğitim Tasarımı"] == "Hasta-gruplu",
                                       "Fark (Aynı Test - Kendi Test)"].values[0]
    print(f"  Satır-rastgele modeli, kendi test setinde vs kronolojik test setinde: {satir_fark:+.4f}")
    print(f"  Hasta-gruplu modeli,   kendi test setinde vs kronolojik test setinde: {hasta_fark:+.4f}")

    cikti = KOK / "veriler" / "capraz_test_seti_kontrolu.csv"
    df_karsilastirma.to_csv(cikti, index=False, encoding="utf-8-sig")
    print(f"\n-> Kaydedildi: {cikti.name}")


if __name__ == "__main__":
    main()