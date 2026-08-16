"""
38_alt_grup_adalet_analizi.py  (Hakem eksiği C: Subgroup / Fairness)

Bir hakem, modelin farklı hasta alt gruplarında (cinsiyet, yaş) eşit çalışıp
çalışmadığını sorar. Model bir alt grupta belirgin daha kötü ayrım yapıyorsa,
bu bir adalet (fairness) sorunudur ve dağıtımda zarar doğurabilir.

Bu script, v4 şampiyon modelin saklı dış test setindeki tahminlerini cinsiyet
ve yaş grubuna göre ayırıp her alt grupta ROC-AUC, no-show oranı ve örneklem
büyüklüğü raporlar. (Hocanın belirlediği v4 model kullanılır.)
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score

KOK = Path(__file__).resolve().parent.parent


def hazirla():
    p = joblib.load(KOK / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib")
    cols = p["sutun_siralamasi"]
    test = pd.read_csv(KOK / "veriler" / "medical_appointments_test.csv")
    ham = test.copy()
    if "appointment_time" in test.columns:
        test = test.drop(columns=["appointment_time"])
    test["icd_frekans"] = test["icd"].map(p["icd_frekans_haritasi"]).fillna(0.0)
    test = test.drop(columns=["icd"])
    y = test["no_show"].map({"no": 0, "yes": 1}).values
    X = test.drop(columns=["no_show"])
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    X = X[cols]
    proba = p["model"].predict_proba(X)[:, 1]
    return ham, y, proba


def alt_grup_auc(y, proba, maske, ad):
    yy, pp = y[maske], proba[maske]
    if len(yy) < 30 or len(np.unique(yy)) < 2:
        return {"grup": ad, "N": int(maske.sum()), "no_show_oran": float(yy.mean()) if len(yy) else np.nan,
                "ROC-AUC": np.nan, "not": "yetersiz örneklem"}
    return {"grup": ad, "N": int(maske.sum()), "no_show_oran": float(yy.mean()),
            "ROC-AUC": float(roc_auc_score(yy, pp)), "not": ""}


def main():
    ham, y, proba = hazirla()
    kayit = []

    print("=" * 70)
    print("ALT-GRUP / ADALET ANALİZİ (v4 şampiyon model, dış test)")
    print("=" * 70)

    # Cinsiyet (one-hot'tan geri çıkar)
    print("\n--- Cinsiyet ---")
    for kolon, ad in [("gender_F", "Kadın"), ("gender_M", "Erkek")]:
        if kolon in ham.columns:
            maske = ham[kolon].astype(bool).values
            r = alt_grup_auc(y, proba, maske, ad)
            kayit.append({"boyut": "cinsiyet", **r})
            print(f"  {ad:8s}: N={r['N']:5d}  no-show={r['no_show_oran']:.3f}  AUC={r['ROC-AUC']:.3f}")

    # Yaş grubu
    print("\n--- Yaş grubu ---")
    if "age" in ham.columns:
        yas = ham["age"].values
        for lo, hi, ad in [(0, 18, "0-17"), (18, 40, "18-39"), (40, 65, "40-64"), (65, 200, "65+")]:
            maske = (yas >= lo) & (yas < hi)
            if maske.sum() >= 30:
                r = alt_grup_auc(y, proba, maske, ad)
                kayit.append({"boyut": "yas", **r})
                print(f"  {ad:8s}: N={r['N']:5d}  no-show={r['no_show_oran']:.3f}  AUC={r['ROC-AUC']:.3f}")

    df = pd.DataFrame(kayit)
    df.to_csv(KOK / "veriler" / "alt_grup_adalet_analizi.csv", index=False, encoding="utf-8-sig")

    # Adalet yorumu: alt gruplar arası AUC farkı
    auc_deg = df["ROC-AUC"].dropna()
    print("\n" + "=" * 70)
    print("YORUM:")
    print(f"Alt gruplar arası AUC aralığı: {auc_deg.min():.3f} - {auc_deg.max():.3f} "
          f"(fark: {auc_deg.max()-auc_deg.min():.3f})")
    if auc_deg.max() - auc_deg.min() < 0.10:
        print("Alt gruplar arası ayrım gücü farkı sınırlıdır (<0.10); model kaba")
        print("adalet açısından gruplar arasında büyük bir tutarsızlık göstermemektedir.")
    else:
        print("Alt gruplar arası ayrım gücü farkı belirgindir (>=0.10); bu, modelin")
        print("bazı alt gruplarda daha zayıf çalıştığını ve dikkatle yorumlanması")
        print("gerektiğini gösterir (adalet sınırlaması olarak raporlanmalı).")
    print("-> Kaydedildi: veriler/alt_grup_adalet_analizi.csv")


if __name__ == "__main__":
    main()
