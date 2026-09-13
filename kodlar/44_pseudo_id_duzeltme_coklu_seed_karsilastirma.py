import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

KOK = Path(__file__).resolve().parent.parent

RF = dict(n_estimators=500, max_depth=25, min_samples_leaf=1, min_samples_split=10,
          max_features=0.5, random_state=42, n_jobs=-1)

GECMIS = ["gecmis_randevu_sayisi", "gecmis_no_show_sayisi", "gecmis_no_show_orani", "ilk_ziyaret_mi"]

HAVA = ["average_temp_day", "average_rain_day", "max_temp_day", "max_rain_day",
        "rainy_day_before", "storm_day_before", "rain_intensity", "heat_intensity",
        "temp_range", "rain_range"]

SEEDS = [42, 7, 123, 99, 2024, 55, 8, 17, 256, 500]


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


def bol(df, strateji, random_state):
    if strateji == "satir_rastgele":
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
        i_tr, i_te = next(sss.split(df, df["no_show_bin"]))
        return df.iloc[i_tr].copy(), df.iloc[i_te].copy()

    if strateji == "hasta_gruplu":
        gruplar = df["pseudo_id"].copy()
        nan_maske = gruplar.isna()
        gruplar = gruplar.astype("object")
        gruplar[nan_maske] = ["__tekil_%d" % i for i in range(nan_maske.sum())]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
        i_tr, i_te = next(gss.split(df, df["no_show_bin"], groups=gruplar))
        return df.iloc[i_tr].copy(), df.iloc[i_te].copy()

    if strateji == "kronolojik":
        return df[df["appointment_year"] <= 2020].copy(), df[df["appointment_year"] >= 2021].copy()
    raise ValueError(strateji)


def degerlendir(df, strateji, gecmis_dahil, random_state):
    egitim, test = bol(df, strateji, random_state)
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


def yukle(dosya_adi):
    df = pd.read_csv(KOK / "veriler" / dosya_adi, low_memory=False)
    df = ozellik_uret(df)
    df["no_show_bin"] = (df["no_show"] == "yes").astype(int)
    return df


def main():
    t0 = time.time()
    print("Veri yükleniyor...")
    eski_df = yukle("step02_pseudo_gecmis_dahil.csv")
    yeni_df = yukle("step02_pseudo_gecmis_duzeltilmis.csv")
    print(f"Yüklendi ({time.time()-t0:.1f}sn). Eski N={len(eski_df):,}, Yeni N={len(yeni_df):,}")

    veri_setleri = [("Eski (bozuk pseudo_id)", eski_df), ("Düzeltilmiş pseudo_id", yeni_df)]
    sonuc = []

    print("\n" + "=" * 100)
    print("HASTA-GRUPLU SENARYO: 10 RANDOM_STATE x 2 VERİ KÜMESİ (geçmiş YOK sabit)")
    print("=" * 100)
    for vlabel, vdf in veri_setleri:
        for seed in SEEDS:
            m = degerlendir(vdf, "hasta_gruplu", gecmis_dahil=False, random_state=seed)
            sonuc.append({"Veri Kümesi": vlabel, "Bölme": "Hasta-gruplu", "random_state": seed, **m})
            print(f"{vlabel:24s} | seed={seed:5d} | ROC-AUC={m['ROC-AUC']:.4f} | "
                  f"N_test={m['N_test']:5d} | Test_Hasta={m['Test_Hasta_Sayisi']:5d}")

    print("\n" + "=" * 100)
    print("SATIR-RASTGELE ve KRONOLOJİK: DOĞRULAMA (random_state=42, tek çalıştırma)")
    print("=" * 100)
    for vlabel, vdf in veri_setleri:
        for strateji, sad in [("satir_rastgele", "Satır-rastgele"), ("kronolojik", "Kronolojik")]:
            m = degerlendir(vdf, strateji, gecmis_dahil=False, random_state=42)
            sonuc.append({"Veri Kümesi": vlabel, "Bölme": sad, "random_state": 42, **m})
            print(f"{vlabel:24s} | {sad:15s} | ROC-AUC={m['ROC-AUC']:.4f} | "
                  f"N_test={m['N_test']:5d} | Test_Hasta={m['Test_Hasta_Sayisi']:5d}")

    df_sonuc = pd.DataFrame(sonuc)
    ham_yolu = KOK / "veriler" / "makale_KESIN_alti_senaryo_v2.csv"
    df_sonuc.to_csv(ham_yolu, index=False, encoding="utf-8-sig")

    hg = df_sonuc[df_sonuc["Bölme"] == "Hasta-gruplu"]
    ozet = hg.groupby("Veri Kümesi").agg(
        ROC_AUC_ortalama=("ROC-AUC", "mean"),
        ROC_AUC_std=("ROC-AUC", "std"),
        ROC_AUC_min=("ROC-AUC", "min"),
        ROC_AUC_maks=("ROC-AUC", "max"),
        N_test_ortalama=("N_test", "mean"),
        N_test_std=("N_test", "std"),
        N_test_min=("N_test", "min"),
        N_test_maks=("N_test", "max"),
    ).round(4).reset_index()
    ozet_yolu = KOK / "veriler" / "makale_KESIN_alti_senaryo_v2_ozet.csv"
    ozet.to_csv(ozet_yolu, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("ÖZET: HASTA-GRUPLU SENARYO, 10 RANDOM_STATE ÜZERİNDEN (geçmiş YOK)")
    print("=" * 100)
    print(ozet.to_string(index=False))
    print("\n" + "=" * 100)
    print("SATIR-RASTGELE / KRONOLOJİK: ESKİ vs YENİ FARK KONTROLÜ")
    print("=" * 100)
    diger = df_sonuc[df_sonuc["Bölme"] != "Hasta-gruplu"]
    for sad in ["Satır-rastgele", "Kronolojik"]:
        alt = diger[diger["Bölme"] == sad]
        eski_v = alt[alt["Veri Kümesi"] == "Eski (bozuk pseudo_id)"]["ROC-AUC"].values[0]
        yeni_v = alt[alt["Veri Kümesi"] == "Düzeltilmiş pseudo_id"]["ROC-AUC"].values[0]
        fark = yeni_v - eski_v
        print(f"{sad:15s} | Eski ROC-AUC={eski_v:.4f} | Yeni ROC-AUC={yeni_v:.4f} | "
              f"Fark={fark:+.4f} | {'DEĞİŞMEDİ' if abs(fark) < 0.0005 else 'DEĞİŞTİ'}")

    print(f"\n-> Kaydedildi: {ham_yolu.name}")
    print(f"-> Kaydedildi: {ozet_yolu.name}")
    print(f"\nToplam süre: {time.time()-t0:.1f}sn")


if __name__ == "__main__":
    main()