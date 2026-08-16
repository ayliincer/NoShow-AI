import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, brier_score_loss

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

MODEL_ADI = "MLP (Derin Öğrenme, Optimize)"
KOK = Path(__file__).resolve().parent.parent


def veri_hazirla():
    egitim = pd.read_csv(KOK / "veriler" / "medical_appointments_train.csv")
    test = pd.read_csv(KOK / "veriler" / "medical_appointments_test.csv")
    for df in (egitim, test):
        if "appointment_time" in df.columns:
            df.drop(columns=["appointment_time"], inplace=True)
    freq = egitim["icd"].value_counts(normalize=True)
    egitim["icd_frekans"] = egitim["icd"].map(freq)
    test["icd_frekans"] = test["icd"].map(freq).fillna(0.0)
    egitim.drop(columns=["icd"], inplace=True)
    test.drop(columns=["icd"], inplace=True)
    y_train = egitim["no_show"].map({"no": 0, "yes": 1})
    y_test = test["no_show"].map({"no": 0, "yes": 1})
    X_train = egitim.drop(columns=["no_show"])
    X_test = test.drop(columns=["no_show"])
    return X_train, X_test, y_train, y_test, freq


def main():
    X_train, X_test, y_train, y_test, freq = veri_hazirla()

    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", MLPClassifier(early_stopping=True, n_iter_no_change=10, max_iter=200, random_state=42)),
    ])

    arama_uzayi = {
        "model__hidden_layer_sizes": [(32,), (64, 32), (128, 64), (128, 64, 32)],
        "model__alpha": [1e-4, 1e-3, 1e-2],
        "model__learning_rate_init": [1e-3, 5e-4],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    arama = RandomizedSearchCV(
        pipe, arama_uzayi, n_iter=8, scoring="roc_auc", cv=cv,
        random_state=42, n_jobs=1, refit=True, verbose=1,
    )

    print("=" * 100)
    print("SIZINTISIZ MLP HİPERPARAMETRE ARAMASI (SADECE EĞİTİM SETİ, 3 KATLI CV)")
    print("=" * 100)
    t0 = time.time()
    arama.fit(X_train, y_train)
    print(f"Süre: {time.time()-t0:.1f} sn | En iyi CV ROC-AUC: {arama.best_score_:.4f}")
    print("En iyi parametreler:", arama.best_params_)

    en_iyi = arama.best_estimator_
    proba_test = en_iyi.predict_proba(X_test)[:, 1]
    pred_test = en_iyi.predict(X_test)

    satir_dis_test = {
        "Model": MODEL_ADI,
        "Dış Test ROC-AUC": roc_auc_score(y_test, proba_test),
        "Dış Test PR-AUC (AP)": average_precision_score(y_test, proba_test),
        "Dış Test F1-Skoru (t=0.5)": f1_score(y_test, pred_test),
        "Dış Test Recall": recall_score(y_test, pred_test),
        "Dış Test Brier Skoru": brier_score_loss(y_test, proba_test),
    }
    print("\n=== SAKLI DIŞ TEST SONUCU (TEK SEFER) ===")
    for k, v in satir_dis_test.items():
        if k != "Model":
            print(f"{k}: {v:.4f}")

    yol = KOK / "veriler" / "nihai_dis_dogrulama_sonuclari.csv"
    mevcut = pd.read_csv(yol)
    mevcut = mevcut[mevcut["Model"] != MODEL_ADI]
    guncel = pd.concat([mevcut, pd.DataFrame([satir_dis_test])], ignore_index=True)
    guncel = guncel.sort_values(by="Dış Test ROC-AUC", ascending=False).reset_index(drop=True)
    guncel.to_csv(yol, index=False, encoding="utf-8-sig")
    print(f"\n-> Güncellendi: {yol.name}")


if __name__ == "__main__":
    main()
