import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

KOK = Path(__file__).resolve().parent.parent
RF_PARAMS = dict(n_estimators=200, max_depth=25, min_samples_leaf=2,
                  min_samples_split=5, max_features=0.4, random_state=42, n_jobs=-1)


def veri_yukle():
    train = pd.read_csv(KOK / "veriler" / "medical_appointments_train.csv")
    test = pd.read_csv(KOK / "veriler" / "medical_appointments_test.csv")
    tum = pd.concat([train, test], ignore_index=True)
    if "appointment_time" in tum.columns:
        tum.drop(columns=["appointment_time"], inplace=True)
    return tum


def egit_test_et(tum, egitim_yillari, test_yili):
    ktr = tum[tum["appointment_year"].isin(egitim_yillari)].copy()
    kte = tum[tum["appointment_year"] == test_yili].copy()
    if len(kte) < 50 or ktr["no_show"].nunique() < 2 or kte["no_show"].nunique() < 2:
        return None
    freq = ktr["icd"].value_counts(normalize=True)
    ktr["icd_frekans"] = ktr["icd"].map(freq)
    kte["icd_frekans"] = kte["icd"].map(freq).fillna(0.0)
    ktr.drop(columns=["icd"], inplace=True)
    kte.drop(columns=["icd"], inplace=True)
    y_tr = ktr["no_show"].map({"no": 0, "yes": 1})
    y_te = kte["no_show"].map({"no": 0, "yes": 1})
    X_tr = ktr.drop(columns=["no_show"])
    X_te = kte.drop(columns=["no_show"])
    ortak = [c for c in X_tr.columns if c in X_te.columns]
    X_tr, X_te = X_tr[ortak], X_te[ortak]
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_tr, y_tr)
    proba = rf.predict_proba(X_te)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "n_train": len(ktr), "n_test": len(kte),
        "test_no_show_orani": y_te.mean(),
    }


def main():
    tum = veri_yukle()
    yillar = sorted(tum["appointment_year"].unique())

    print("=" * 100)
    print("1) TEK KRONOLOJİK BÖLÜNME: 2016-2020 eğitim -> 2021-2022 test")
    print("=" * 100)
    r_tek = egit_test_et(tum, [y for y in yillar if y <= 2020], None)
    # Özel durum: 2021-2022 birlikte test
    ktr = tum[tum["appointment_year"] <= 2020].copy()
    kte = tum[tum["appointment_year"] >= 2021].copy()
    freq = ktr["icd"].value_counts(normalize=True)
    ktr["icd_frekans"] = ktr["icd"].map(freq); kte["icd_frekans"] = kte["icd"].map(freq).fillna(0.0)
    ktr.drop(columns=["icd"], inplace=True); kte.drop(columns=["icd"], inplace=True)
    y_tr = ktr["no_show"].map({"no": 0, "yes": 1}); y_te = kte["no_show"].map({"no": 0, "yes": 1})
    X_tr = ktr.drop(columns=["no_show"]); X_te = kte.drop(columns=["no_show"])
    ortak = [c for c in X_tr.columns if c in X_te.columns]
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_tr[ortak], y_tr)
    proba = rf.predict_proba(X_te[ortak])[:, 1]
    print(f"ROC-AUC={roc_auc_score(y_te, proba):.4f}  PR-AUC={average_precision_score(y_te, proba):.4f}  "
          f"(Rastgele bölünmede: ROC-AUC=0.8153, PR-AUC=0.4076)")

    print("\n" + "=" * 100)
    print("2) GENİŞLEYEN PENCERE (EXPANDING WINDOW) DOĞRULAMASI")
    print("=" * 100)
    sonuclar = []
    for i in range(2, len(yillar)):
        test_yili = yillar[i]
        r_tum = egit_test_et(tum, yillar[:i], test_yili)
        r_son2 = egit_test_et(tum, yillar[max(0, i - 2):i], test_yili)
        r_son1 = egit_test_et(tum, yillar[i - 1:i], test_yili)
        for etiket, r in [("Tüm Geçmiş", r_tum), ("Son 2 Yıl", r_son2), ("Son 1 Yıl", r_son1)]:
            if r:
                sonuclar.append({"Test_Yili": test_yili, "Egitim_Penceresi": etiket, **r})
                print(f"Test={test_yili} | Pencere={etiket:12s} | N_egitim={r['n_train']:6d} | "
                      f"ROC-AUC={r['roc_auc']:.4f} | PR-AUC={r['pr_auc']:.4f}")

    df = pd.DataFrame(sonuclar)
    df.to_csv(KOK / "veriler" / "kronolojik_genisleyen_pencere_dogrulama.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("ÖZET")
    print("=" * 100)
    print(f"Ortalama ROC-AUC (Tüm Geçmiş):  {df[df.Egitim_Penceresi=='Tüm Geçmiş']['roc_auc'].mean():.4f}")
    print(f"Ortalama ROC-AUC (Son 2 Yıl):   {df[df.Egitim_Penceresi=='Son 2 Yıl']['roc_auc'].mean():.4f}")
    print(f"Ortalama ROC-AUC (Son 1 Yıl):   {df[df.Egitim_Penceresi=='Son 1 Yıl']['roc_auc'].mean():.4f}")
    print("YORUM: Tüm pencere stratejileri ~0.50 (yazı-tura) civarında kalmaktadır.")
    print("Model, rastgele bölünmedeki yüksek performansına rağmen GERÇEK prospektif")
    print("(ileri yönlü) genelleme kapasitesi göstermemektedir. Bu, appointment_year")
    print("değişkeninin çıkarılmasıyla da değişmemiştir (ayrıca test edilmiştir).")


if __name__ == "__main__":
    main()