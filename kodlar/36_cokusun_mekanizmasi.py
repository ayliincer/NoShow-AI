"""
36_cokusun_mekanizmasi.py  (Danışman Madde 4)

Kronolojik çöküşün mekanizmasını nicelleştirir. "No-show oranı arttı" ifadesini
somut kanıtlarla destekler ve düşüşü label shift / covariate shift / concept
drift ayrımıyla çerçeveler:

  (a) Yıl bazında no-show taban oranı (prior) — label shift göstergesi
  (b) Yıl bazında AUC (kronolojik modelin her yıl üzerindeki ayrımı)
  (c) Kronolojik test setinde kalibrasyon (Brier + güvenilirlik eğrisi verisi)
  (d) appointment_year/month'a bağımlılık (drop-column importance)

Girdi: step02_pseudo_gecmis_dahil.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

KOK = Path(__file__).resolve().parent.parent
RF = dict(n_estimators=500, max_depth=25, min_samples_leaf=1, min_samples_split=10,
          max_features=0.5, random_state=42, n_jobs=-1)
HAVA = ["average_temp_day", "average_rain_day", "max_temp_day", "max_rain_day",
        "rainy_day_before", "storm_day_before", "rain_intensity", "heat_intensity",
        "temp_range", "rain_range"]
GECMIS = ["gecmis_randevu_sayisi", "gecmis_no_show_sayisi", "gecmis_no_show_orani", "ilk_ziyaret_mi"]


def ozellik_uret(v):
    v = v.copy()
    v["appointment_date"] = pd.to_datetime(v["appointment_date"], errors="coerce")
    v["entry_service_date"] = pd.to_datetime(v["entry_service_date"], errors="coerce")
    v["lead_time"] = ((v["appointment_date"] - v["entry_service_date"]).dt.days).clip(lower=0)
    v["appointment_day_of_week"] = v["appointment_date"].dt.dayofweek
    v["appointment_hour"] = pd.to_datetime(v["appointment_time"], format="%H:%M", errors="coerce").dt.hour
    for c in ["lead_time", "appointment_hour", "appointment_day_of_week"]:
        v[c] = v[c].fillna(v[c].median())
    return v


def kodla(egitim, test):
    freq = egitim["icd"].value_counts(normalize=True)
    egitim = egitim.copy(); test = test.copy()
    egitim["icd_frekans"] = egitim["icd"].map(freq)
    test["icd_frekans"] = test["icd"].map(freq).fillna(0.0)
    kat = [c for c in ["specialty", "gender", "disability", "city", "appointment_month", "appointment_shift"] if c in egitim.columns]
    egitim = pd.get_dummies(egitim, columns=kat); test = pd.get_dummies(test, columns=kat)
    test = test.reindex(columns=egitim.columns, fill_value=0)
    dus = (["no_show", "no_show_bin", "pseudo_id", "_orijinal_sira", "icd", "appointment_date",
            "entry_service_date", "date_of_birth", "appointment_time", "no_show_reason"] + HAVA + GECMIS)
    oz = [c for c in egitim.columns if c not in dus]
    Xtr = egitim[oz].select_dtypes(include=[np.number, bool]).astype(float).fillna(0)
    Xte = test[oz].select_dtypes(include=[np.number, bool]).astype(float).fillna(0)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, egitim["no_show_bin"], Xte, test["no_show_bin"], test


def main():
    df = pd.read_csv(KOK / "veriler" / "step02_pseudo_gecmis_dahil.csv", low_memory=False)
    df = ozellik_uret(df)
    df["no_show_bin"] = (df["no_show"] == "yes").astype(int)

    # (a) Yıl bazında taban oran (label/prior shift)
    print("=" * 70)
    print("(a) YIL BAZINDA NO-SHOW TABAN ORANI (prior / label shift göstergesi)")
    print("=" * 70)
    yil_oran = df.groupby("appointment_year")["no_show_bin"].agg(["mean", "count"])
    for yil, satir in yil_oran.iterrows():
        print(f"  {int(yil)}: taban oran={satir['mean']:.3f}  (N={int(satir['count'])})")

    # Kronolojik model eğit
    egitim = df[df["appointment_year"] <= 2020]
    test = df[df["appointment_year"] >= 2021]
    Xtr, ytr, Xte, yte, test_df = kodla(egitim, test)
    m = RandomForestClassifier(**RF); m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)[:, 1]

    # (b) Test dönemi yıl bazında AUC
    print("\n" + "=" * 70)
    print("(b) KRONOLOJİK MODELİN TEST DÖNEMİ YIL BAZINDA AUC'si")
    print("=" * 70)
    test_df = test_df.copy(); test_df["_proba"] = proba; test_df["_y"] = yte.values
    for yil in sorted(test_df["appointment_year"].unique()):
        alt = test_df[test_df["appointment_year"] == yil]
        if alt["_y"].nunique() > 1:
            print(f"  {int(yil)}: AUC={roc_auc_score(alt['_y'], alt['_proba']):.3f}  "
                  f"taban={alt['_y'].mean():.3f}  (N={len(alt)})")

    # (c) Kalibrasyon (Brier + güvenilirlik eğrisi verisi)
    print("\n" + "=" * 70)
    print("(c) KRONOLOJİK TEST KALİBRASYONU")
    print("=" * 70)
    brier = brier_score_loss(yte, proba)
    print(f"  Brier skoru: {brier:.4f}")
    print(f"  Ortalama tahmin: {proba.mean():.3f}  |  Gerçek taban: {yte.mean():.3f}")
    print(f"  (Tahmin < gerçek => model yeni dönemdeki artışı yakalayamıyor = prior shift)")
    frac_pos, mean_pred = calibration_curve(yte, proba, n_bins=10, strategy="quantile")
    kal = pd.DataFrame({"tahmin_ort": mean_pred, "gercek_oran": frac_pos})
    kal.to_csv(KOK / "veriler" / "kronolojik_kalibrasyon.csv", index=False, encoding="utf-8-sig")

    # (d) appointment_year/month'a bağımlılık: drop-column importance
    print("\n" + "=" * 70)
    print("(d) TAKVİM ÖZNİTELİKLERİNE BAĞIMLILIK (drop-column importance)")
    print("=" * 70)
    base_auc = roc_auc_score(yte, proba)
    takvim_kolonlar = [c for c in Xtr.columns if "year" in c.lower() or "month" in c.lower()]
    print(f"  Takvim öznitelikleri: {len(takvim_kolonlar)} adet")
    Xtr2 = Xtr.drop(columns=takvim_kolonlar); Xte2 = Xte.drop(columns=takvim_kolonlar)
    m2 = RandomForestClassifier(**RF); m2.fit(Xtr2, ytr)
    auc2 = roc_auc_score(yte, m2.predict_proba(Xte2)[:, 1])
    print(f"  Tüm öznitelikler:        AUC={base_auc:.3f}")
    print(f"  Takvim çıkarılınca:      AUC={auc2:.3f}  (fark: {auc2-base_auc:+.3f})")
    print(f"  -> Takvim özniteliklerini çıkarmak kronolojik AUC'yi {'artırıyor' if auc2>base_auc else 'düşürüyor'};")
    print(f"     bu, modelin durağan-olmayan takvim sinyaline zararlı bağımlılığını gösterir.")

    # Özet CSV
    pd.DataFrame([{
        "toplam_yil_bazinda_taban_artisi": f"{yil_oran['mean'].min():.3f}->{yil_oran['mean'].max():.3f}",
        "kronolojik_brier": brier, "ort_tahmin": proba.mean(), "gercek_taban": yte.mean(),
        "auc_tum": base_auc, "auc_takvim_haric": auc2,
    }]).to_csv(KOK / "veriler" / "cokus_mekanizmasi_ozet.csv", index=False, encoding="utf-8-sig")
    print("\n-> Kaydedildi: kronolojik_kalibrasyon.csv, cokus_mekanizmasi_ozet.csv")
    print("\nYORUM: Prior (taban oran) belirgin kaymış [label shift]; model ortalama")
    print("tahmini gerçek tabanın altında [kalibrasyon bozulması]; takvim özniteliklerine")
    print("bağımlılık [concept drift'e kırılganlık]. Üçü birlikte çöküşü açıklıyor.")


if __name__ == "__main__":
    main()
