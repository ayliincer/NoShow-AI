import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def islenmis_veri_setlerini_yukle(egitim_yolu: Path, test_yolu: Path) -> tuple:
    try:
        egitim = pd.read_csv(egitim_yolu)
        test = pd.read_csv(test_yolu)
        return egitim, test
    except Exception as hata:
        print(f"\nVeri setleri yüklenirken hata oluştu:\n{hata}")
        return None, None


def kalan_kategorik_degiskenleri_isle(egitim: pd.DataFrame, test: pd.DataFrame) -> tuple:
    print("\n" + "=" * 90)
    print("İŞLEM: 16. ADIMDAN KALAN KATEGORİK DEĞİŞKENLERİN İŞLENMESİ")
    print("=" * 90)

    ham_zaman_sutunu = "appointment_time"
    for veri_kumesi in (egitim, test):
        if ham_zaman_sutunu in veri_kumesi.columns:
            veri_kumesi.drop(columns=[ham_zaman_sutunu], inplace=True)
    print(
        f"Bilgi: 'appointment_hour' ve 'appointment_shift' türetilmiş öznitelikleriyle "
        f"örtüştüğü için ham '{ham_zaman_sutunu}' sütunu matris dışı bırakılmıştır."
    )

    yuksek_kardinalite_sutunu = "icd"
    if yuksek_kardinalite_sutunu in egitim.columns:
        frekans_haritasi = egitim[yuksek_kardinalite_sutunu].value_counts(normalize=True)

        egitim[f"{yuksek_kardinalite_sutunu}_frekans"] = egitim[yuksek_kardinalite_sutunu].map(frekans_haritasi)
        test[f"{yuksek_kardinalite_sutunu}_frekans"] = (
            test[yuksek_kardinalite_sutunu].map(frekans_haritasi).fillna(0.0)
        )

        egitim.drop(columns=[yuksek_kardinalite_sutunu], inplace=True)
        test.drop(columns=[yuksek_kardinalite_sutunu], inplace=True)

        print(
            f"Bilgi: '{yuksek_kardinalite_sutunu}' değişkeni ({frekans_haritasi.shape[0]} sınıf) "
            f"yalnızca eğitim verisinden öğrenilen Frequency Encoding ile "
            f"'{yuksek_kardinalite_sutunu}_frekans' olarak sayısallaştırılmıştır."
        )

    print("=" * 90)
    return egitim, test


def hedef_ve_oznitelikleri_ayir(egitim: pd.DataFrame, test: pd.DataFrame, hedef_sutun: str) -> tuple:
    print("\n" + "=" * 90)
    print("İŞLEM: HEDEF DEĞİŞKENİN İKİLİ KODLANMASI VE X / y AYRIMI")
    print("=" * 90)

    kodlama_semasi = {"no": 0, "yes": 1}

    y_egitim = egitim[hedef_sutun].map(kodlama_semasi)
    y_test = test[hedef_sutun].map(kodlama_semasi)

    X_egitim = egitim.drop(columns=[hedef_sutun])
    X_test = test.drop(columns=[hedef_sutun])

    print(f"X_train boyutu : {X_egitim.shape[0]:,} satır | {X_egitim.shape[1]} sütun")
    print(f"X_test boyutu  : {X_test.shape[0]:,} satır | {X_test.shape[1]} sütun")
    print("-" * 90)
    print("y_train sınıf dağılımı:")
    print(
        y_egitim.value_counts(normalize=True)
        .rename(index={0: "no_show=0 (Geldi)", 1: "no_show=1 (Gelmedi)"})
        .to_string()
    )
    print("=" * 90)

    return X_egitim, X_test, y_egitim, y_test


def model_pipeline_lerini_olustur(y_train: pd.Series) -> dict:
    print("\n" + "=" * 90)
    print("İŞLEM: MODEL PIPELINE'LARININ KURULMASI")
    print("=" * 90)

    sinif_sayilari = y_train.value_counts()
    scale_pos_weight = sinif_sayilari[0] / sinif_sayilari[1]
    print(f"XGBoost için hesaplanan scale_pos_weight : {scale_pos_weight:.4f}")
    print("=" * 90)

    pipelines = {
    "Logistic Regression": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        )),
    ]),

    "Decision Tree": ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", DecisionTreeClassifier(
            random_state=42
        )),
    ]),

    "Random Forest": ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )),
    ]),

    "XGBoost": ImbPipeline([
        ("model", XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )),
    ]),

    "LightGBM": ImbPipeline([
        ("model", LGBMClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )),
    ]),

    "CatBoost": ImbPipeline([
        ("model", CatBoostClassifier(
            auto_class_weights="Balanced",
            random_state=42,
            verbose=False,
            allow_writing_files=False
        )),
    ]),
}

    return pipelines


def capraz_dogrulama_calistir(pipelines: dict, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    print("\n" + "=" * 90)
    print("İŞLEM: 5 KATLI TABAKALI ÇAPRAZ DOĞRULAMA")
    print("=" * 90)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrikler = [
    "roc_auc",
    "average_precision",
    "f1",
    "recall",
    "neg_brier_score",
]
    sonuclar = []
    for isim, pipeline in pipelines.items():
        print(f"\nÇalıştırılıyor: {isim} ...")

        skorlar = cross_validate(
            pipeline, X_train, y_train,
            cv=cv, scoring=metrikler, n_jobs=-1,
        )

        satir = {"Model": isim}
        for metrik in metrikler:
            ortalama = skorlar[f"test_{metrik}"].mean()
            std = skorlar[f"test_{metrik}"].std()
            if metrik == "neg_brier_score":
                satir["Brier Score"] = f"{-ortalama:.4f} (+/-{std:.4f})"
            else:
                satir[metrik] = f"{ortalama:.4f} (+/-{std:.4f})"

        sonuclar.append(satir)

    sonuc_tablosu = pd.DataFrame(sonuclar)

    sonuc_tablosu["ROC_AUC_Sayisal"] = (
        sonuc_tablosu["roc_auc"]
        .str.extract(r"([0-9.]+)", expand=False)
        .astype(float)
    )

    sonuc_tablosu = (
        sonuc_tablosu
        .sort_values(by="ROC_AUC_Sayisal", ascending=False)
        .drop(columns="ROC_AUC_Sayisal")
        .reset_index(drop=True)
    )

    print("\n" + "=" * 90)
    print("ÇAPRAZ DOĞRULAMA SONUÇ TABLOSU (EĞİTİM SETİ)")
    print("=" * 90)
    print(sonuc_tablosu.to_string(index=False))
    print("=" * 90)

    print(f"\nToplam Karşılaştırılan Model Sayısı : {len(pipelines)}")

    en_iyi = sonuc_tablosu.iloc[0]

    print("-" * 90)
    print("ÇAPRAZ DOĞRULAMA ŞAMPİYONU")
    print("-" * 90)
    print(f"Model   : {en_iyi['Model']}")
    print(f"ROC_AUC : {en_iyi['roc_auc']}")
    print("=" * 90)

    return sonuc_tablosu


def sonuclari_diske_kaydet(sonuc_tablosu: pd.DataFrame, kayit_yolu: Path):
    sonuc_tablosu.to_csv(kayit_yolu, index=False, encoding="utf-8-sig")
    print("\nÇapraz doğrulama sonuçları başarıyla kaydedildi.")
    print(f"Dosya : {kayit_yolu}")


def main():
    egitim_yolu = Path(__file__).resolve().parent.parent / "veriler" / "medical_appointments_train.csv"
    test_yolu = Path(__file__).resolve().parent.parent / "veriler" / "medical_appointments_test.csv"

    egitim_veri, test_veri = islenmis_veri_setlerini_yukle(egitim_yolu, test_yolu)

    if egitim_veri is None or test_veri is None:
        return

    egitim_veri, test_veri = kalan_kategorik_degiskenleri_isle(egitim_veri, test_veri)

    X_train, X_test, y_train, y_test = hedef_ve_oznitelikleri_ayir(egitim_veri, test_veri, "no_show")

    pipelines = model_pipeline_lerini_olustur(y_train)

    sonuc_tablosu = capraz_dogrulama_calistir(pipelines, X_train, y_train)

    kayit_yolu = Path(__file__).resolve().parent.parent / "veriler" / "model_karsilastirma_sonuclari.csv"
    sonuclari_diske_kaydet(sonuc_tablosu, kayit_yolu)


if __name__ == "__main__":
    main()