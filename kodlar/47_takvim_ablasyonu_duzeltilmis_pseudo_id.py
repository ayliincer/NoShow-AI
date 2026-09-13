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

# lead_time KASITLI OLARAK burada YOK (script 42 ile aynı gerekçe).
TAKVIM = ["appointment_year", "appointment_month", "appointment_day_of_week", "appointment_hour"]

SABIT_SONUCLAR = {
    "Satır-rastgele (hasta sızıntısı VAR)": {"Takvim VAR": 0.7753, "Takvim YOK": 0.7247},
    "Kronolojik (hasta-ayrık + zamansal kayma)": {"Takvim VAR": 0.5480, "Takvim YOK": 0.5420},
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


def kodla(egitim, test, gecmis_dahil, takvim_dahil):
    freq = egitim["icd"].value_counts(normalize=True)
    egitim = egitim.copy(); test = test.copy()
    egitim["icd_frekans"] = egitim["icd"].map(freq)
    test["icd_frekans"] = test["icd"].map(freq).fillna(0.0)
    kat = [c for c in ["specialty", "gender", "disability", "city", "appointment_month", "appointment_shift"] if c in egitim.columns]
    if not takvim_dahil:
        kat = [c for c in kat if c != "appointment_month"]
    egitim = pd.get_dummies(egitim, columns=kat)
    test = pd.get_dummies(test, columns=kat)
    test = test.reindex(columns=egitim.columns, fill_value=0)
    dus = (["no_show", "no_show_bin", "pseudo_id", "_orijinal_sira", "icd", "appointment_date",
            "entry_service_date", "date_of_birth", "appointment_time", "no_show_reason"] + HAVA)
    if not gecmis_dahil:
        dus += GECMIS
    if not takvim_dahil:
        dus += TAKVIM
    oz = [c for c in egitim.columns if c not in dus]
    Xtr = egitim[oz].select_dtypes(include=[np.number, bool]).astype(float).fillna(0)
    Xte = test[oz].select_dtypes(include=[np.number, bool]).astype(float).fillna(0)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, egitim["no_show_bin"], Xte, test["no_show_bin"]


def bol_hasta_gruplu(df):
    gruplar = df["pseudo_id"].copy()
    nan_maske = gruplar.isna()
    gruplar = gruplar.astype("object")
    gruplar[nan_maske] = ["__tekil_%d" % i for i in range(nan_maske.sum())]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    i_tr, i_te = next(gss.split(df, df["no_show_bin"], groups=gruplar))
    return df.iloc[i_tr].copy(), df.iloc[i_te].copy()


def degerlendir(df, gecmis_dahil, takvim_dahil):
    egitim, test = bol_hasta_gruplu(df)
    Xtr, ytr, Xte, yte = kodla(egitim, test, gecmis_dahil, takvim_dahil)
    m = RandomForestClassifier(**RF); m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)[:, 1]
    rng = np.random.default_rng(42)
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
            "feature_count": Xtr.shape[1]}


def main():
    df = pd.read_csv(KOK / "veriler" / "step02_pseudo_gecmis_duzeltilmis.csv", low_memory=False)
    df = ozellik_uret(df)
    df["no_show_bin"] = (df["no_show"] == "yes").astype(int)

    hasta_gruplu_etiket = "Hasta-gruplu (hasta-ayrık, zamanda rastgele)"

    print("=" * 100)
    print("TAKVİM ÖZNİTELİK ABLASYONU - DÜZELTİLMİŞ PSEUDO_ID (sadece hasta-gruplu)")
    print("=" * 100)

    sonuc = []
    for takvim_dahil, tetiket in [(True, "Takvim VAR"), (False, "Takvim YOK")]:
        m = degerlendir(df, gecmis_dahil=False, takvim_dahil=takvim_dahil)
        sonuc.append({"Bölme": hasta_gruplu_etiket, "Takvim": tetiket, **m})
        print(f"{hasta_gruplu_etiket:46s} | {tetiket:11s} | "
              f"ROC-AUC={m['ROC-AUC']:.4f} [{m['GA_alt']:.4f}-{m['GA_ust']:.4f}] "
              f"PR-AUC={m['PR-AUC']:.4f} N_ozellik={m['feature_count']}")

    df_uzun = pd.DataFrame(sonuc)

    var_row = [s for s in sonuc if s["Takvim"] == "Takvim VAR"][0]
    yok_row = [s for s in sonuc if s["Takvim"] == "Takvim YOK"][0]

    genis = []


    for sad, degerler in SABIT_SONUCLAR.items():
        fark = degerler["Takvim YOK"] - degerler["Takvim VAR"]
        genis.append({
            "Bölme Stratejisi": sad,
            "ROC-AUC (Takvim VAR)": degerler["Takvim VAR"],
            "ROC-AUC (Takvim YOK)": degerler["Takvim YOK"],
            "Fark (YOK - VAR)": round(fark, 4),
            "Yön": "İYİLEŞTİ" if fark > 0.0005 else ("KÖTÜLEŞTİ" if fark < -0.0005 else "DEĞİŞMEDİ (~0)"),
        })

    fark_hg = yok_row["ROC-AUC"] - var_row["ROC-AUC"]
    genis.append({
        "Bölme Stratejisi": hasta_gruplu_etiket,
        "ROC-AUC (Takvim VAR)": round(var_row["ROC-AUC"], 4),
        "ROC-AUC (Takvim YOK)": round(yok_row["ROC-AUC"], 4),
        "Fark (YOK - VAR)": round(fark_hg, 4),
        "Yön": "İYİLEŞTİ" if fark_hg > 0.0005 else ("KÖTÜLEŞTİ" if fark_hg < -0.0005 else "DEĞİŞMEDİ (~0)"),
    })



    sira = ["Satır-rastgele (hasta sızıntısı VAR)", hasta_gruplu_etiket,
            "Kronolojik (hasta-ayrık + zamansal kayma)"]
    df_genis = pd.DataFrame(genis).set_index("Bölme Stratejisi").loc[sira].reset_index()

    print("\n" + "=" * 100)
    print("KARŞILAŞTIRMA TABLOSU v2 (geçmiş YOK sabit; hasta-gruplu = düzeltilmiş pseudo_id,")
    print("satır-rastgele/kronolojik = script 42'den değişmeden)")
    print("=" * 100)
    print(df_genis.to_string(index=False))

    hg = df_genis[df_genis["Bölme Stratejisi"] == hasta_gruplu_etiket].iloc[0]
    print("\n" + "-" * 100)
    print("HASTA-GRUPLU SENARYO (yeniden hesaplanan, asıl soru):")
    print(f"  Takvim VAR: {hg['ROC-AUC (Takvim VAR)']:.4f}")
    print(f"  Takvim YOK: {hg['ROC-AUC (Takvim YOK)']:.4f}")
    print(f"  Fark (YOK - VAR): {hg['Fark (YOK - VAR)']:+.4f}  -> {hg['Yön']}")

    uzun_yolu = KOK / "veriler" / "takvim_ablasyon_uzun_v2.csv"
    genis_yolu = KOK / "veriler" / "takvim_ablasyon_karsilastirma_v2.csv"
    df_uzun.to_csv(uzun_yolu, index=False, encoding="utf-8-sig")
    df_genis.to_csv(genis_yolu, index=False, encoding="utf-8-sig")
    print(f"\n-> Kaydedildi: {uzun_yolu}")
    print(f"-> Kaydedildi: {genis_yolu}")


if __name__ == "__main__":
    main()