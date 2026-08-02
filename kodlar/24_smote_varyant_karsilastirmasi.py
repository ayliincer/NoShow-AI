"""
24_smote_varyant_karsilastirmasi.py

AMAÇ: "Sadece düz SMOTE denendi, Borderline-SMOTE/Tomek-ENN denenmedi" eksikliğini
gerçekten kapatmak. Optimize edilmiş RF hiperparametreleriyle (n_estimators=300,
max_depth=25, min_samples_leaf=2, min_samples_split=5, max_features=0.4), 4 farklı
dengesizlik giderme stratejisi SADECE EĞİTİM SETİ + 5 katlı CV ile karşılaştırılır:
  0) Yeniden örnekleme yok (mevcut şampiyon, referans)
  1) Düz SMOTE
  2) Borderline-SMOTE (sınır bölgesindeki azınlık örneklerine odaklanır)
  3) SMOTETomek (SMOTE + Tomek Links ile gürültülü/çakışan örnekleri temizler)
  4) SMOTEENN (SMOTE + Edited Nearest Neighbours ile temizler)

Saklı dış test seti yalnızca en iyi CV performansı veren strateji için TEK SEFER açılır.
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, brier_score_loss

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

KOK = Path(__file__).resolve().parent.parent
RF_PARAMS = dict(n_estimators=300, max_depth=25, min_samples_leaf=2,
                  min_samples_split=5, max_features=0.4, random_state=42, n_jobs=-1)


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
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    stratejiler = {
        "Yeniden Örnekleme Yok (Mevcut Şampiyon)": None,
        "SMOTE (Düz)": SMOTE(random_state=42),
        "Borderline-SMOTE": BorderlineSMOTE(random_state=42),
        "SMOTETomek": SMOTETomek(random_state=42),
        "SMOTEENN": SMOTEENN(random_state=42),
    }

    sonuclar = []
    for isim, sampler in stratejiler.items():
        print("\n" + "=" * 100)
        print(f"STRATEJİ: {isim}")
        print("=" * 100)
        t0 = time.time()
        if sampler is None:
            model = RandomForestClassifier(**RF_PARAMS)
        else:
            model = ImbPipeline([("sampler", sampler), ("model", RandomForestClassifier(**RF_PARAMS))])

        skorlar = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["roc_auc", "average_precision", "neg_brier_score"], n_jobs=1,
        )
        cv_auc = skorlar["test_roc_auc"].mean()
        cv_ap = skorlar["test_average_precision"].mean()
        cv_brier = -skorlar["test_neg_brier_score"].mean()
        sure = time.time() - t0
        print(f"CV ROC-AUC={cv_auc:.4f}  CV PR-AUC={cv_ap:.4f}  CV Brier={cv_brier:.4f}  (süre={sure:.1f}s)")
        sonuclar.append({
            "Strateji": isim, "CV ROC-AUC": cv_auc, "CV PR-AUC": cv_ap, "CV Brier": cv_brier, "Süre (sn)": sure
        })

    df_sonuc = pd.DataFrame(sonuclar).sort_values("CV ROC-AUC", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 100)
    print("CV KARŞILAŞTIRMA TABLOSU (yalnızca eğitim seti, sızıntısız)")
    print("=" * 100)
    print(df_sonuc.to_string(index=False))
    df_sonuc.to_csv(KOK / "veriler" / "smote_varyant_cv_karsilastirmasi.csv", index=False, encoding="utf-8-sig")

    en_iyi_isim = df_sonuc.iloc[0]["Strateji"]
    print(f"\nEn iyi CV ROC-AUC'a sahip strateji: {en_iyi_isim} -> saklı dış test setinde TEK SEFER değerlendiriliyor")

    en_iyi_sampler = stratejiler[en_iyi_isim]
    if en_iyi_sampler is None:
        nihai_model = RandomForestClassifier(**RF_PARAMS)
        nihai_model.fit(X_train, y_train)
    else:
        nihai_model = ImbPipeline([("sampler", en_iyi_sampler), ("model", RandomForestClassifier(**RF_PARAMS))])
        nihai_model.fit(X_train, y_train)

    proba_test = nihai_model.predict_proba(X_test)[:, 1]
    pred_test = nihai_model.predict(X_test)
    dis_test_satir = {
        "Strateji": en_iyi_isim,
        "Dış Test ROC-AUC": roc_auc_score(y_test, proba_test),
        "Dış Test PR-AUC (AP)": average_precision_score(y_test, proba_test),
        "Dış Test F1-Skoru (t=0.5)": f1_score(y_test, pred_test),
        "Dış Test Recall": recall_score(y_test, pred_test),
        "Dış Test Brier Skoru": brier_score_loss(y_test, proba_test),
    }
    print("\n=== SAKLI DIŞ TEST SONUCU (TEK SEFER) ===")
    for k, v in dis_test_satir.items():
        if k != "Strateji":
            print(f"{k}: {v:.4f}")

    pd.DataFrame([dis_test_satir]).to_csv(
        KOK / "veriler" / "smote_varyant_dis_test_sonucu.csv", index=False, encoding="utf-8-sig"
    )
    print("\n-> Kaydedildi: smote_varyant_cv_karsilastirmasi.csv, smote_varyant_dis_test_sonucu.csv")


if __name__ == "__main__":
    main()
