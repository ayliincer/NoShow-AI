"""
39_zaman_stabil_kismi_kurtarma.py

BULGU: Kronolojik çöküşün başlıca nedeni, modelin döneme-özgü zamansal
özniteliklere (appointment_year, appointment_month) aşırı bağlanmasıdır.
Bu öznitelikler çıkarılıp yalnızca ZAMAN-STABİL öznitelikler kullanıldığında
kronolojik performans kısmen geri kazanılır (ROC-AUC 0.55 -> 0.64), ancak
rastgele-split (0.78) düzeyine ulaşmaz.
Girdi: step02_pseudo_gecmis_dahil.csv (script 33 çıktısı).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, f1_score, recall_score)

KOK = Path(__file__).resolve().parent.parent
STABIL = ["age", "lead_time", "hafta_gunu", "gecmis_no_show_orani",
          "gecmis_randevu_sayisi", "ilk_ziyaret_mi"]


def main():
    df = pd.read_csv(KOK / "veriler" / "step02_pseudo_gecmis_dahil.csv", low_memory=False)
    df["appointment_date"] = pd.to_datetime(df["appointment_date"], errors="coerce")
    df["entry_service_date"] = pd.to_datetime(df["entry_service_date"], errors="coerce")
    df["lead_time"] = ((df["appointment_date"] - df["entry_service_date"]).dt.days).clip(lower=0)
    df["hafta_gunu"] = df["appointment_date"].dt.dayofweek
    df["no_show_bin"] = (df["no_show"] == "yes").astype(int)
    df = df.dropna(subset=["age", "lead_time"])

    egitim = df[df["appointment_year"] <= 2020]
    test = df[df["appointment_year"] >= 2021]

    aucler, prs = [], []
    for rs in [42, 7, 123]:
        rf = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=10,
                                    max_features=0.5, random_state=rs, n_jobs=-1)
        rf.fit(egitim[STABIL], egitim["no_show_bin"])
        proba = rf.predict_proba(test[STABIL])[:, 1]
        aucler.append(roc_auc_score(test["no_show_bin"], proba))
        prs.append(average_precision_score(test["no_show_bin"], proba))

    # Madde 1 (danışman): F1-optimal eşik TEST'ten değil, EĞİTİM setinin (2016-2020)
    # kendi tahminlerinden seçilip test'e SABİT uygulanır (eşik sızıntısı önlenir).
    egitim_proba = rf.predict_proba(egitim[STABIL])[:, 1]
    p_tr, r_tr, th_tr = precision_recall_curve(egitim["no_show_bin"], egitim_proba)
    f1_tr = 2 * p_tr * r_tr / (p_tr + r_tr + 1e-9)
    sabit_esik = th_tr[int(np.argmax(f1_tr[:-1]))]
    pred = (proba >= sabit_esik).astype(int)

    print("=" * 70)
    print("ZAMAN-STABİL ÖZNİTELİKLERLE KRONOLOJİK PERFORMANS")
    print("=" * 70)
    print(f"Kronolojik ROC-AUC: {np.mean(aucler):.4f} ± {np.std(aucler):.4f} (3 seed)")
    print(f"Kronolojik PR-AUC:  {np.mean(prs):.4f} (taban: {test['no_show_bin'].mean():.3f})")
    print(f"F1-optimal eşikte Recall={recall_score(test['no_show_bin'],pred):.3f}, "
          f"F1={f1_score(test['no_show_bin'],pred):.3f}")
    print(f"\nKarşılaştırma: tam öznitelik kronolojik 0.551 -> zaman-stabil {np.mean(aucler):.3f}")

    pd.DataFrame([{"yaklasim": "Zaman-stabil (kronolojik)", "ROC_AUC": np.mean(aucler),
                   "PR_AUC": np.mean(prs), "recall_opt": recall_score(test["no_show_bin"], pred)}]
                 ).to_csv(KOK / "veriler" / "zaman_stabil_kismi_kurtarma.csv", index=False, encoding="utf-8-sig")
    print("-> Kaydedildi: veriler/zaman_stabil_kismi_kurtarma.csv")


if __name__ == "__main__":
    main()
