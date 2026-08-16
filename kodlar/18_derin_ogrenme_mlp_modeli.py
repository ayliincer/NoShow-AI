import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, brier_score_loss

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

MODEL_ADI = "MLP (Derin Öğrenme)"


def veri_setlerini_yukle(egitim_yolu: Path, test_yolu: Path) -> tuple:
    try:
        egitim = pd.read_csv(egitim_yolu)
        test = pd.read_csv(test_yolu)
        return egitim, test
    except Exception as hata:
        print(f"\nVeri setleri yüklenirken hata oluştu:\n{hata}")
        return None, None


def kalan_kategorik_degiskenleri_isle(egitim: pd.DataFrame, test: pd.DataFrame) -> tuple:
    ham_zaman_sutunu = "appointment_time"
    for veri_kumesi in (egitim, test):
        if ham_zaman_sutunu in veri_kumesi.columns:
            veri_kumesi.drop(columns=[ham_zaman_sutunu], inplace=True)

    yuksek_kardinalite_sutunu = "icd"
    if yuksek_kardinalite_sutunu in egitim.columns:
        frekans_haritasi = egitim[yuksek_kardinalite_sutunu].value_counts(normalize=True)
        egitim[f"{yuksek_kardinalite_sutunu}_frekans"] = egitim[yuksek_kardinalite_sutunu].map(frekans_haritasi)
        test[f"{yuksek_kardinalite_sutunu}_frekans"] = (
            test[yuksek_kardinalite_sutunu].map(frekans_haritasi).fillna(0.0)
        )
        egitim.drop(columns=[yuksek_kardinalite_sutunu], inplace=True)
        test.drop(columns=[yuksek_kardinalite_sutunu], inplace=True)

    return egitim, test


def hedef_ve_oznitelikleri_ayir(egitim: pd.DataFrame, test: pd.DataFrame, hedef_sutun: str) -> tuple:
    kodlama_semasi = {"no": 0, "yes": 1}
    y_egitim = egitim[hedef_sutun].map(kodlama_semasi)
    y_test = test[hedef_sutun].map(kodlama_semasi)
    X_egitim = egitim.drop(columns=[hedef_sutun])
    X_test = test.drop(columns=[hedef_sutun])
    return X_egitim, X_test, y_egitim, y_test


def mlp_pipeline_olustur() -> ImbPipeline:
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=42,
        )),
    ])


def capraz_dogrulama_calistir(pipeline: ImbPipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    print("\n" + "=" * 90)
    print(f"İŞLEM: {MODEL_ADI} İÇİN 5 KATLI TABAKALI ÇAPRAZ DOĞRULAMA (17. Script ile Aynı Protokol)")
    print("=" * 90)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrikler = ["roc_auc", "average_precision", "f1", "recall", "neg_brier_score"]

    skorlar = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=metrikler, n_jobs=-1)

    satir = {"Model": MODEL_ADI}
    for metrik in metrikler:
        ortalama = skorlar[f"test_{metrik}"].mean()
        std = skorlar[f"test_{metrik}"].std()
        if metrik == "neg_brier_score":
            satir["Brier Score"] = f"{-ortalama:.4f} (+/-{std:.4f})"
        else:
            satir[metrik] = f"{ortalama:.4f} (+/-{std:.4f})"

    print(f"ROC-AUC (CV ortalama) : {satir['roc_auc']}")
    print(f"PR-AUC (CV ortalama)  : {satir['average_precision']}")
    print(f"Brier Score (CV)      : {satir['Brier Score']}")
    print("=" * 90)
    return satir


def dis_dogrulama_calistir(pipeline: ImbPipeline, X_train, y_train, X_test, y_test) -> dict:
    print("\n" + "=" * 110)
    print(f"İŞLEM: {MODEL_ADI} İÇİN SAKLI DIŞ TEST SETİ (N={len(X_test):,}) ÜZERİNDE DOĞRULAMA")
    print("=" * 110)

    pipeline.fit(X_train, y_train)
    tahmin_olasiliklari = pipeline.predict_proba(X_test)[:, 1]
    sinif_tahminleri = pipeline.predict(X_test)

    satir = {
        "Model": MODEL_ADI,
        "Dış Test ROC-AUC": roc_auc_score(y_test, tahmin_olasiliklari),
        "Dış Test PR-AUC (AP)": average_precision_score(y_test, tahmin_olasiliklari),
        "Dış Test F1-Skoru (t=0.5)": f1_score(y_test, sinif_tahminleri),
        "Dış Test Recall": recall_score(y_test, sinif_tahminleri),
        "Dış Test Brier Skoru": brier_score_loss(y_test, tahmin_olasiliklari),
    }

    print(f"Dış Test ROC-AUC     : {satir['Dış Test ROC-AUC']:.4f}")
    print(f"Dış Test PR-AUC      : {satir['Dış Test PR-AUC (AP)']:.4f}")
    print(f"Dış Test Brier Skoru : {satir['Dış Test Brier Skoru']:.4f}")
    print("=" * 110)
    return satir


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


def main():
    kok = Path(__file__).resolve().parent.parent
    egitim_yolu = kok / "veriler" / "medical_appointments_train.csv"
    test_yolu = kok / "veriler" / "medical_appointments_test.csv"

    egitim_veri, test_veri = veri_setlerini_yukle(egitim_yolu, test_yolu)
    if egitim_veri is None or test_veri is None:
        return

    egitim_veri, test_veri = kalan_kategorik_degiskenleri_isle(egitim_veri, test_veri)
    X_train, X_test, y_train, y_test = hedef_ve_oznitelikleri_ayir(egitim_veri, test_veri, "no_show")

    cv_pipeline = mlp_pipeline_olustur()
    cv_satiri = capraz_dogrulama_calistir(cv_pipeline, X_train, y_train)
    csv_ye_ekle_veya_guncelle(kok / "veriler" / "model_karsilastirma_sonuclari.csv", cv_satiri, sirala_sutun="roc_auc")

    disval_pipeline = mlp_pipeline_olustur()
    disval_satiri = dis_dogrulama_calistir(disval_pipeline, X_train, y_train, X_test, y_test)
    csv_ye_ekle_veya_guncelle(kok / "veriler" / "nihai_dis_dogrulama_sonuclari.csv", disval_satiri, sirala_sutun="Dış Test ROC-AUC")

    print("\n" + "=" * 110)
    print("SONUÇ: MLP modeli her iki karşılaştırma tablosuna da eklendi.")
    print("Güncel tabloları görüntülemek için model_karsilastirma_sonuclari.csv ve")
    print("nihai_dis_dogrulama_sonuclari.csv dosyalarını inceleyiniz.")
    print("=" * 110)


if __name__ == "__main__":
    main()
