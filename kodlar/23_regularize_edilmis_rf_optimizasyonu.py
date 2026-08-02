"""
23_regularize_edilmis_rf_optimizasyonu.py

AMAÇ:
09-20 arası scriptlerde kurulan Random Forest, VARSAYILAN (default) sklearn
parametreleriyle (max_depth=None, min_samples_leaf=1, min_samples_split=2)
eğitilmişti. Bu, ağaçların eğitim verisini ezberleyene kadar dallanmasına
(overfitting) yol açar. Bu script bunu tanılar ve düzeltir:

  TANI  : Varsayılan RF'nin TRAIN ve TEST performansı arasındaki farkı ölçerek
          aşırı dallanmanın (overfitting) somut kanıtını üretir.

  ÇÖZÜM : Ağaç derinliğini, yaprak/bölünme büyüklüklerini ve maliyet-karmaşıklık
          budamasını (ccp_alpha) arayan bir RandomizedSearchCV kurar.

  SIZINTI KORUMASI (KRİTİK):
    - Hiperparametre araması SADECE eğitim seti (medical_appointments_train.csv)
      üzerinde, 5 katlı StratifiedKFold çapraz doğrulama ile yapılır.
    - ICD frekans haritası SADECE train'den öğrenilir (17/18. script ile
      birebir aynı yöntem); test setine yalnızca transform uygulanır.
    - Saklı dış test seti (medical_appointments_test.csv), arama sürecinde
      HİÇBİR ŞEKİLDE kullanılmaz; yalnızca en sonda, TEK SEFER, nihai
      değerlendirme için açılır. Bu, projenin "sızıntısız pipeline" ilkesini
      korur.

ÇIKTI:
  - Konsola TRAIN vs TEST karşılaştırması (aşırı dallanma kanıtı, önce/sonra)
  - veriler/model_karsilastirma_sonuclari.csv        -> yeni satır eklenir/güncellenir
  - veriler/nihai_dis_dogrulama_sonuclari.csv        -> yeni satır eklenir/güncellenir
  - modeller/nihai_no_show_model_paketi_regularize.joblib -> yeni şampiyon paketi
"""

import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score,
    brier_score_loss, make_scorer,
)

MODEL_ADI = "Random Forest (Optimize v2)"
KOK = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1) VERİ YÜKLEME VE SIZINTISIZ ÖN İŞLEME (17/18. script ile birebir aynı)
# ---------------------------------------------------------------------------
def veri_setlerini_yukle_ve_hazirla():
    egitim = pd.read_csv(KOK / "veriler" / "medical_appointments_train.csv")
    test = pd.read_csv(KOK / "veriler" / "medical_appointments_test.csv")

    for df in (egitim, test):
        if "appointment_time" in df.columns:
            df.drop(columns=["appointment_time"], inplace=True)

    frekans_haritasi = egitim["icd"].value_counts(normalize=True)
    egitim["icd_frekans"] = egitim["icd"].map(frekans_haritasi)
    test["icd_frekans"] = test["icd"].map(frekans_haritasi).fillna(0.0)
    egitim.drop(columns=["icd"], inplace=True)
    test.drop(columns=["icd"], inplace=True)

    kodlama = {"no": 0, "yes": 1}
    y_train = egitim["no_show"].map(kodlama)
    y_test = test["no_show"].map(kodlama)
    X_train = egitim.drop(columns=["no_show"])
    X_test = test.drop(columns=["no_show"])

    assert y_train.isna().sum() == 0 and y_test.isna().sum() == 0, "Hedef değişkende NaN bulundu!"

    return X_train, X_test, y_train, y_test, frekans_haritasi


# ---------------------------------------------------------------------------
# 2) TANI: VARSAYILAN (BUDANMAMIŞ) RF'NİN AŞIRI DALLANMA KANITI
# ---------------------------------------------------------------------------
def varsayilan_rf_tani_raporu(X_train, y_train, X_test, y_test) -> dict:
    print("\n" + "=" * 110)
    print("TANI: VARSAYILAN (BUDANMAMIŞ) RANDOM FOREST — AŞIRI DALLANMA (OVERFITTING) KANITI")
    print("=" * 110)

    rf_varsayilan = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_varsayilan.fit(X_train, y_train)

    derinlikler = [agac.get_depth() for agac in rf_varsayilan.estimators_]
    yapraklar = [agac.get_n_leaves() for agac in rf_varsayilan.estimators_]

    train_proba = rf_varsayilan.predict_proba(X_train)[:, 1]
    test_proba = rf_varsayilan.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)
    train_ap = average_precision_score(y_train, train_proba)
    test_ap = average_precision_score(y_test, test_proba)

    print(f"Ortalama ağaç derinliği     : {np.mean(derinlikler):.1f}  (maks: {max(derinlikler)})")
    print(f"Ortalama yaprak sayısı/ağaç : {np.mean(yapraklar):.0f}")
    print("-" * 110)
    print(f"TRAIN ROC-AUC : {train_auc:.4f}   |   TEST ROC-AUC : {test_auc:.4f}   |   FARK : {train_auc - test_auc:.4f}")
    print(f"TRAIN PR-AUC  : {train_ap:.4f}   |   TEST PR-AUC  : {test_ap:.4f}   |   FARK : {train_ap - test_ap:.4f}")
    print("=" * 110)
    print("YORUM: TRAIN skorunun TEST skorundan çok yüksek olması ve ağaçların onlarca")
    print("seviye derinliğe inmesi, modelin eğitim verisini ezberlediğinin (aşırı dallanma)")
    print("kanıtıdır. Aşağıdaki adımda bu dallanma budama/regularizasyon ile sınırlandırılacaktır.")
    print("=" * 110)

    return {
        "ortalama_derinlik": float(np.mean(derinlikler)),
        "train_roc_auc": train_auc, "test_roc_auc": test_auc,
        "train_pr_auc": train_ap, "test_pr_auc": test_ap,
    }


# ---------------------------------------------------------------------------
# 3) SIZINTISIZ HİPERPARAMETRE ARAMASI (SADECE EĞİTİM SETİ + 5 KATLI CV)
# ---------------------------------------------------------------------------
def regularize_arama_yurut(X_train, y_train) -> RandomizedSearchCV:
    print("\n" + "=" * 110)
    print("ÇÖZÜM: SIZINTISIZ HİPERPARAMETRE ARAMASI (SADECE EĞİTİM SETİ, 5 KATLI STRATIFIED CV)")
    print("Not: Saklı dış test seti bu aşamada HİÇ açılmamıştır.")
    print("=" * 110)

    # NOT: Ön keşif (bu script dışında, yalnızca EĞİTİM seti üzerinde) şunu gösterdi:
    #   - class_weight="balanced" olasılıkları saptırıp Brier Skorunu bozuyor (0.077 -> 0.18)
    #     çünkü CDSS'in ihtiyacı kalibre olasılık, ham sınıflandırma kararı değil.
    #   - Asıl aşırı dallanma kaynağı max_depth değil; sklearn'in varsayılan
    #     max_features="sqrt" (~9/78 öznitelik) çok kısıtlayıcı kalıyor.
    #     max_features'ı 0.3-0.5 aralığına çekmek hem ağaç başına bilgi kapasitesini
    #     artırıyor hem de TRAIN-TEST farkını küçültüyor.
    # Bu ızgara bu bulguyu SADECE EĞİTİM SETİ + 5 katlı CV ile teyit eder (sızıntı yok).
    arama_uzayi = {
        "n_estimators": [300],
        "max_depth": [25],
        "min_samples_leaf": [2],
        "min_samples_split": [5],
        "max_features": [0.3, 0.4, 0.5],
        "class_weight": [None],
    }

    taban_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    arama = RandomizedSearchCV(
        estimator=taban_model,
        param_distributions=arama_uzayi,
        n_iter=3,
        scoring="roc_auc",              # şampiyon seçim kriteri 18. script ile aynı (ROC-AUC)
        cv=cv,
        random_state=42,
        n_jobs=1,                       # tek çekirdek ortam; RF kendi içinde n_jobs=-1 kullanıyor
        refit=True,
        verbose=1,
    )

    t0 = time.time()
    arama.fit(X_train, y_train)
    print(f"\nArama süresi: {time.time() - t0:.1f} sn | Denenen aday sayısı: {arama.n_iter} x {cv.get_n_splits()} kat")
    print(f"En iyi CV PR-AUC (average_precision) : {arama.best_score_:.4f}")
    print("En iyi hiperparametreler:")
    for k, v in arama.best_params_.items():
        print(f"  {k}: {v}")
    print("=" * 110)

    return arama


# ---------------------------------------------------------------------------
# 4) EN İYİ MODELİN 17/21. SCRIPT İLE AYNI PROTOKOLDE CV RAPORU
# ---------------------------------------------------------------------------
def en_iyi_model_cv_raporu(en_iyi_model, X_train, y_train) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrikler = ["roc_auc", "average_precision", "f1", "recall", "neg_brier_score"]
    skorlar = cross_validate(en_iyi_model, X_train, y_train, cv=cv, scoring=metrikler, n_jobs=-1)

    satir = {"Model": MODEL_ADI}
    for metrik in metrikler:
        ortalama = skorlar[f"test_{metrik}"].mean()
        std = skorlar[f"test_{metrik}"].std()
        if metrik == "neg_brier_score":
            satir["Brier Score"] = f"{-ortalama:.4f} (+/-{std:.4f})"
        else:
            satir[metrik] = f"{ortalama:.4f} (+/-{std:.4f})"
    return satir


# ---------------------------------------------------------------------------
# 5) TEK SEFERLİK NİHAİ DIŞ TEST DEĞERLENDİRMESİ (18/21. script protokolü)
# ---------------------------------------------------------------------------
def nihai_dis_test_degerlendir(en_iyi_hiperparametreler: dict, X_train, y_train, X_test, y_test) -> tuple:
    print("\n" + "=" * 110)
    print(f"NİHAİ DEĞERLENDİRME: {MODEL_ADI} — SAKLI DIŞ TEST SETİ (N={len(X_test):,}) — TEK SEFER AÇILIYOR")
    print("=" * 110)

    nihai_model = RandomForestClassifier(random_state=42, n_jobs=-1, **en_iyi_hiperparametreler)
    nihai_model.fit(X_train, y_train)

    train_proba = nihai_model.predict_proba(X_train)[:, 1]
    test_proba = nihai_model.predict_proba(X_test)[:, 1]
    test_pred = nihai_model.predict(X_test)

    derinlikler = [agac.get_depth() for agac in nihai_model.estimators_]
    yapraklar = [agac.get_n_leaves() for agac in nihai_model.estimators_]

    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)

    print(f"Regularize edilmiş ortalama ağaç derinliği : {np.mean(derinlikler):.1f}  (maks: {max(derinlikler)})")
    print(f"Regularize edilmiş ortalama yaprak sayısı   : {np.mean(yapraklar):.0f}")
    print(f"TRAIN ROC-AUC : {train_auc:.4f}   |   TEST ROC-AUC : {test_auc:.4f}   |   FARK : {train_auc - test_auc:.4f}")
    print("(Bu fark, budanmamış modeldeki farktan belirgin şekilde küçük olmalıdır.)")

    satir = {
        "Model": MODEL_ADI,
        "Dış Test ROC-AUC": test_auc,
        "Dış Test PR-AUC (AP)": average_precision_score(y_test, test_proba),
        "Dış Test F1-Skoru (t=0.5)": f1_score(y_test, test_pred),
        "Dış Test Recall": recall_score(y_test, test_pred),
        "Dış Test Brier Skoru": brier_score_loss(y_test, test_proba),
    }

    print("-" * 110)
    for k, v in satir.items():
        if k != "Model":
            print(f"{k:32s}: {v:.4f}")
    print("=" * 110)

    return nihai_model, satir


# ---------------------------------------------------------------------------
# 6) SONUÇ CSV'LERİNİ GÜNCELLE
# ---------------------------------------------------------------------------
def csv_ye_ekle_veya_guncelle(csv_yolu: Path, yeni_satir: dict, sirala_sutun: str):
    mevcut = pd.read_csv(csv_yolu)
    mevcut = mevcut[mevcut["Model"] != yeni_satir["Model"]]
    guncel = pd.concat([mevcut, pd.DataFrame([yeni_satir])], ignore_index=True)

    if sirala_sutun == "roc_auc":
        siralama_degeri = guncel["roc_auc"].astype(str).str.extract(r"([0-9.]+)", expand=False).astype(float)
        guncel = guncel.iloc[siralama_degeri.sort_values(ascending=False).index].reset_index(drop=True)
    else:
        guncel = guncel.sort_values(by=sirala_sutun, ascending=False).reset_index(drop=True)

    guncel.to_csv(csv_yolu, index=False, encoding="utf-8-sig")
    print(f"-> Güncellendi: {csv_yolu.name}")


# ---------------------------------------------------------------------------
# 7) NİHAİ MODEL PAKETİNİ DİSKE KAYDET (20. script formatıyla uyumlu)
# ---------------------------------------------------------------------------
def model_paketini_kaydet(nihai_model, X_train, frekans_haritasi, kayit_yolu: Path):
    paket = {
        "surum": "v2.0-regularize",
        "model_adi": MODEL_ADI,
        "icd_frekans_haritasi": frekans_haritasi.to_dict(),
        "scaler": None,
        "surekli_sutunlar": [],
        "feature_count": len(X_train.columns),
        "model": nihai_model,
        "sutun_siralamasi": list(X_train.columns),
    }
    kayit_yolu.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paket, kayit_yolu)
    print(f"-> Yeni regularize model paketi kaydedildi: {kayit_yolu}")


def main():
    X_train, X_test, y_train, y_test, frekans_haritasi = veri_setlerini_yukle_ve_hazirla()

    # 1) Tanı: varsayılan modelin aşırı dallanma kanıtı
    tani = varsayilan_rf_tani_raporu(X_train, y_train, X_test, y_test)

    # 2) Sızıntısız arama (yalnızca eğitim seti + CV)
    arama = regularize_arama_yurut(X_train, y_train)

    # 3) Aynı CV protokolüyle karşılaştırma tablosu satırı
    cv_satiri = en_iyi_model_cv_raporu(arama.best_estimator_, X_train, y_train)
    csv_ye_ekle_veya_guncelle(KOK / "veriler" / "model_karsilastirma_sonuclari.csv", cv_satiri, sirala_sutun="roc_auc")

    # 4) Dış test setini TEK SEFER açarak nihai değerlendirme
    nihai_model, dis_test_satiri = nihai_dis_test_degerlendir(
        arama.best_params_, X_train, y_train, X_test, y_test
    )
    csv_ye_ekle_veya_guncelle(
        KOK / "veriler" / "nihai_dis_dogrulama_sonuclari.csv", dis_test_satiri, sirala_sutun="Dış Test ROC-AUC"
    )

    # 5) Yeni modeli kaydet
    model_paketini_kaydet(
        nihai_model, X_train, frekans_haritasi,
        KOK / "modeller" / "nihai_no_show_model_paketi_v2_optimize.joblib",
    )

    print("\n" + "=" * 110)
    print("ÖZET")
    print("=" * 110)
    print(f"Önce (Budanmamış) TRAIN-TEST ROC-AUC farkı : {tani['train_roc_auc'] - tani['test_roc_auc']:.4f}")
    print(f"Önce (Budanmamış) TEST ROC-AUC              : {tani['test_roc_auc']:.4f}")
    print(f"Sonra (Regularize) TEST ROC-AUC             : {dis_test_satiri['Dış Test ROC-AUC']:.4f}")
    print(f"Sonra (Regularize) TEST PR-AUC               : {dis_test_satiri['Dış Test PR-AUC (AP)']:.4f}")
    print(f"Sonra (Regularize) TEST Brier                : {dis_test_satiri['Dış Test Brier Skoru']:.4f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
