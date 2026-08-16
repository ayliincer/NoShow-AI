"""
39_baseline_ve_overfitting_kontrolu.py  (Hakem eksikleri: naive baseline + overfit)

Bir hakem iki şey sorar:
(1) Model, önemsiz bir "çoğunluk sınıfı / prevalans" tahmincisinden gerçekten
    daha mı iyi? (naive baseline karşılaştırması)
(2) Model eğitim setini ezberliyor mu? (train vs test AUC farkı = genelleme boşluğu)

v4 şampiyon modelin train ve test AUC'lerini karşılaştırır ve naive baseline'ları
raporlar. (Hocanın belirlediği v4 model ve öznitelik seti kullanılır.)
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.dummy import DummyClassifier

KOK = Path(__file__).resolve().parent.parent


def veri_hazirla(dosya, p):
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


def main():
    p = joblib.load(KOK / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib")
    model = p["model"]
    Xtr, ytr = veri_hazirla("medical_appointments_train.csv", p)
    Xte, yte = veri_hazirla("medical_appointments_test.csv", p)

    print("=" * 70)
    print("(1) NAIVE BASELINE KARŞILAŞTIRMASI (dış test)")
    print("=" * 70)
    # Prevalans tabanlı baseline: herkese aynı olasılık (prevalans)
    prevalence = ytr.mean()
    baseline_proba = np.full(len(yte), prevalence)
    print(f"  Prevalans tahmincisi (herkes={prevalence:.3f}):")
    print(f"    ROC-AUC = 0.500 (tanım gereği), PR-AUC = {average_precision_score(yte, baseline_proba):.3f}")
    print(f"  Şampiyon model (v4):")
    proba_te = model.predict_proba(Xte)[:, 1]
    print(f"    ROC-AUC = {roc_auc_score(yte, proba_te):.3f}, PR-AUC = {average_precision_score(yte, proba_te):.3f}")
    print(f"  -> Model, naive prevalans tahmincisinin PR-AUC'sini "
          f"{average_precision_score(yte, proba_te)/average_precision_score(yte, baseline_proba):.1f} kat aşıyor.")

    print("\n" + "=" * 70)
    print("(2) GENELLEME BOŞLUĞU (train vs test — overfitting kontrolü)")
    print("=" * 70)
    proba_tr = model.predict_proba(Xtr)[:, 1]
    auc_tr = roc_auc_score(ytr, proba_tr)
    auc_te = roc_auc_score(yte, proba_te)
    print(f"  Eğitim ROC-AUC: {auc_tr:.3f}")
    print(f"  Test   ROC-AUC: {auc_te:.3f}")
    print(f"  Genelleme boşluğu: {auc_tr - auc_te:.3f}")
    if auc_tr - auc_te > 0.15:
        print("  -> Boşluk büyük (>0.15): model eğitim setine belirgin aşırı uyum gösteriyor.")
        print("     (Not: RF'de eğitim AUC'si doğası gereği yüksektir; asıl ölçüt test performansıdır.)")
    else:
        print("  -> Boşluk makul; model kabul edilebilir düzeyde genelliyor.")

    pd.DataFrame([{
        "prevalans": prevalence, "baseline_pr_auc": average_precision_score(yte, baseline_proba),
        "model_roc_auc_test": auc_te, "model_pr_auc_test": average_precision_score(yte, proba_te),
        "model_roc_auc_train": auc_tr, "genelleme_boslugu": auc_tr - auc_te,
    }]).to_csv(KOK / "veriler" / "baseline_ve_overfitting.csv", index=False, encoding="utf-8-sig")
    print("\n-> Kaydedildi: veriler/baseline_ve_overfitting.csv")


if __name__ == "__main__":
    main()
