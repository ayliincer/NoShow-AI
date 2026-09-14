import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from scipy import stats

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


def main():
    df = pd.read_csv(KOK / "veriler" / "step02_pseudo_gecmis_duzeltilmis.csv", low_memory=False)
    df = ozellik_uret(df)
    df["no_show_bin"] = (df["no_show"] == "yes").astype(int)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    i_tr, i_te = next(sss.split(df, df["no_show_bin"]))
    egitim_ham, test_ham = df.iloc[i_tr].copy(), df.iloc[i_te].copy()

    print("=" * 100)
    print("(1) PEARSON KORELASYON: gecmis_no_show_orani vs diğer önemli öznitelikler")
    print("=" * 100)

    freq_tr = egitim_ham["icd"].value_counts(normalize=True)
    icd_frekans_tr = egitim_ham["icd"].map(freq_tr)

    korelasyon_kayit = []
    hedef = {
        "lead_time": egitim_ham["lead_time"],
        "age": egitim_ham["age"] if "age" in egitim_ham.columns else None,
        "appointment_hour": egitim_ham["appointment_hour"],
        "icd_frekans": icd_frekans_tr,
    }
    gno = egitim_ham["gecmis_no_show_orani"]
    for ad, seri in hedef.items():
        if seri is None:
            print(f"  {ad}: sütun bulunamadı, atlandı")
            continue
        mask = gno.notna() & seri.notna()
        r, p = stats.pearsonr(gno[mask], seri[mask])
        korelasyon_kayit.append({"Öznitelik": ad, "Pearson r": round(r, 4), "p-değeri": p, "N": int(mask.sum())})
        print(f"  gecmis_no_show_orani vs {ad:20s}: r={r:+.4f}  p={p:.4g}  N={mask.sum()}")

    df_korelasyon = pd.DataFrame(korelasyon_kayit)
    df_korelasyon.to_csv(KOK / "veriler" / "gecmis_ozellik_korelasyon.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("(2) SHAP ANALİZİ - GEÇMİŞ ÖZNİTELİKLER DAHİL MODEL")
    print("=" * 100)

    Xtr_dahil, ytr_dahil, Xte_dahil, yte_dahil = kodla(egitim_ham, test_ham, gecmis_dahil=True)
    model_dahil = RandomForestClassifier(**RF)
    model_dahil.fit(Xtr_dahil, ytr_dahil)
    proba_dahil = model_dahil.predict_proba(Xte_dahil)[:, 1]
    roc_dahil = roc_auc_score(yte_dahil, proba_dahil)
    print(f"  Model (geçmiş DAHİL) dış test ROC-AUC = {roc_dahil:.4f}  (N_özellik={Xtr_dahil.shape[1]})")

    ornek_n = min(1000, len(Xte_dahil))
    X_sample = Xte_dahil.sample(n=ornek_n, random_state=42)
    explainer = shap.TreeExplainer(model_dahil)
    shap_values = explainer(X_sample, check_additivity=False)
    vals = shap_values.values
    if len(vals.shape) == 3:
        vals = vals[:, :, 1]
    mean_abs = np.abs(vals).mean(axis=0)
    cols = list(Xtr_dahil.columns)
    order = np.argsort(mean_abs)[::-1]

    df_shap = pd.DataFrame({"ozellik": [cols[i] for i in order],
                            "mean_abs_shap": [mean_abs[i] for i in order]})
    df_shap["sira"] = np.arange(1, len(df_shap) + 1)
    df_shap.to_csv(KOK / "veriler" / "gecmis_dahil_shap_siralamasi.csv", index=False, encoding="utf-8-sig")

    print(f"\n  Top 15 öznitelik (Mean |SHAP|), toplam {len(cols)} öznitelik arasından:")
    for _, row in df_shap.head(15).iterrows():
        isaret = "  <-- GEÇMİŞ ÖZNİTELİĞİ" if row["ozellik"] in GECMIS else ""
        print(f"    {int(row['sira']):2d}. {row['ozellik']:35s} {row['mean_abs_shap']:.5f}{isaret}")

    print(f"\n  Geçmiş özniteliklerinin sıradaki yeri (toplam {len(cols)} öznitelik içinde):")
    for g in GECMIS:
        if g in df_shap["ozellik"].values:
            satir = df_shap[df_shap["ozellik"] == g].iloc[0]
            print(f"    {g:30s}: sıra {int(satir['sira'])}/{len(cols)}  (mean|SHAP|={satir['mean_abs_shap']:.5f})")


    print("\n" + "=" * 100)
    print("(3) TEK DEĞİŞKENLİ KONTROL: gecmis_no_show_orani TEK BAŞINA")
    print("=" * 100)

    tek_degisken_roc = roc_auc_score(test_ham["no_show_bin"], test_ham["gecmis_no_show_orani"])
    print(f"  gecmis_no_show_orani TEK BAŞINA (rank-tabanlı) dış test ROC-AUC = {tek_degisken_roc:.4f}")


    Xtr_haric, ytr_haric, Xte_haric, yte_haric = kodla(egitim_ham, test_ham, gecmis_dahil=False)
    model_haric = RandomForestClassifier(**RF)
    model_haric.fit(Xtr_haric, ytr_haric)
    proba_haric = model_haric.predict_proba(Xte_haric)[:, 1]
    roc_haric = roc_auc_score(yte_haric, proba_haric)
    print(f"  Tam model (geçmiş HARİÇ), N_özellik={Xtr_haric.shape[1]}: ROC-AUC = {roc_haric:.4f}")
    print(f"  Tam model (geçmiş DAHİL), N_özellik={Xtr_dahil.shape[1]}: ROC-AUC = {roc_dahil:.4f}")

    print("\n" + "-" * 100)
    print("ÖZET (yorum eklenmedi, sadece rakamlar):")
    print(f"  Tek değişkenli (sadece gecmis_no_show_orani):  {tek_degisken_roc:.4f}")
    print(f"  Tam model, geçmiş HARİÇ:                        {roc_haric:.4f}")
    print(f"  Tam model, geçmiş DAHİL:                        {roc_dahil:.4f}")
    print(f"  Fark (DAHİL - HARİÇ):                           {roc_dahil - roc_haric:+.4f}")

    ozet = pd.DataFrame([{
        "Tek Değişkenli ROC-AUC (sadece gecmis_no_show_orani)": round(tek_degisken_roc, 4),
        "Tam Model ROC-AUC (geçmiş HARİÇ)": round(roc_haric, 4),
        "Tam Model ROC-AUC (geçmiş DAHİL)": round(roc_dahil, 4),
        "Fark (DAHİL - HARİÇ)": round(roc_dahil - roc_haric, 4),
    }])
    ozet.to_csv(KOK / "veriler" / "gecmis_ozellik_roc_auc_karsilastirma.csv", index=False, encoding="utf-8-sig")

    print(f"\n-> Kaydedildi: veriler/gecmis_ozellik_korelasyon.csv")
    print(f"-> Kaydedildi: veriler/gecmis_dahil_shap_siralamasi.csv")
    print(f"-> Kaydedildi: veriler/gecmis_ozellik_roc_auc_karsilastirma.csv")


if __name__ == "__main__":
    main()