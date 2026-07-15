import pandas as pd
import numpy as np
from pathlib import Path

# Modeller ve Araçlar
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, brier_score_loss


def veri_setlerini_yukle(egitim_yolu: Path, test_yolu: Path) -> tuple:
    """
    16. adımda kategorik kodlaması tamamlanmış eğitim ve test setlerini yükler.
    """
    try:
        egitim = pd.read_csv(egitim_yolu)
        test = pd.read_csv(test_yolu)
        return egitim, test
    except Exception as hata:
        print(f"\nVeri setleri yüklenirken hata oluştu:\n{hata}")
        return None, None


def kalan_kategorik_ve_frekans_isleme(egitim: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    ÖN İŞLEME VE DÖNÜŞÜM ADIMI

    'appointment_time' değişkenini çıkarır ve 'icd' değişkenine sızıntısız 
    Frequency Encoding uygular.
    """
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


def dis_dogrulama_yurut(X_tren: pd.DataFrame, y_tren: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    MODELLEME VE DIŞ DOĞRULAMA ADIMI

    Modelleri eğitir ve tamamen izole edilmiş dış test seti üzerinde ampirik olarak test eder.
    """
    print("=" * 110)
    print("İŞLEM: SAKLI DIŞ TEST SETİ (EXTERNAL TEST SET) ÜZERİNDE NİHAİ DIŞ DOĞRULAMA")
    print("=" * 110)

    # Lojistik regresyon için StandardScaler parametreleri sadece eğitim setinden öğrenilir
    surekli_sutunlar = ["age", "average_temp_day", "average_rain_day", "max_temp_day", "max_rain_day"]
    mevcut_surekliler = [s for s in surekli_sutunlar if s in X_tren.columns]

    X_tren_olcekli = X_tren.copy()
    X_test_olcekli = X_test.copy()

    if mevcut_surekliler:
        olcekleyici = StandardScaler()
        X_tren_olcekli[mevcut_surekliler] = olcekleyici.fit_transform(X_tren[mevcut_surekliler])
        X_test_olcekli[mevcut_surekliler] = olcekleyici.transform(X_test[mevcut_surekliler])

    # Sınıf dengesizliği ağırlığı (XGBoost için)
    sinif_sayilari = y_tren.value_counts()
    scale_pos_weight = sinif_sayilari[0] / sinif_sayilari[1]

    modeller = {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42
            ),
            "veri": (X_tren_olcekli, X_test_olcekli)
        },

        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "veri": (X_tren, X_test)
        },

        "Random Forest": {
            "model": RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
            "veri": (X_tren, X_test)
        },

        "XGBoost": {
            "model": XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1
            ),
            "veri": (X_tren, X_test)
        },

        "LightGBM": {
            "model": LGBMClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ),
            "veri": (X_tren, X_test)
        },

        "CatBoost": {
            "model": CatBoostClassifier(
                auto_class_weights="Balanced",
                random_state=42,
                verbose=False,
                allow_writing_files=False
            ),
            "veri": (X_tren, X_test)
        }
    }

    dis_dogrulama_sonuclari = []

    for isim, bilesen in modeller.items():
        print(f"Nihai model eğitiliyor ve test ediliyor: {isim} ...")
        X_tr, X_te = bilesen["veri"]
        model = bilesen["model"]

        model.fit(X_tr, y_tren)
        tahmin_olasiliklari = model.predict_proba(X_te)[:, 1]
        sinif_tahminleri = model.predict(X_te)

        # Metriklerin Hesaplanması
        roc_auc = roc_auc_score(y_test, tahmin_olasiliklari)
        average_precision = average_precision_score(y_test, tahmin_olasiliklari)
        f1 = f1_score(y_test, sinif_tahminleri)
        recall = recall_score(y_test, sinif_tahminleri)
        brier = brier_score_loss(y_test, tahmin_olasiliklari)

        dis_dogrulama_sonuclari.append({
            "Model": isim,
            "Dış Test ROC-AUC": roc_auc,
            "Dış Test PR-AUC (AP)": average_precision,
            "Dış Test F1-Skoru (t=0.5)": f1,
            "Dış Test Recall": recall,
            "Dış Test Brier Skoru": brier
        })

    rapor_tablosu = pd.DataFrame(dis_dogrulama_sonuclari)
    rapor_tablosu = (
    rapor_tablosu
    .sort_values(by="Dış Test ROC-AUC", ascending=False)
    .reset_index(drop=True)
    )
    
    print("\n" + "=" * 110)
    print("NİHAİ DIŞ DOĞRULAMA PERFORMANS TABLOSU (SAKLI TEST SETİ)")
    print("=" * 110)
    print(rapor_tablosu.to_string(index=False))
    print("=" * 110)

    # En başarılı modeli otomatik raporla
    en_iyi = rapor_tablosu.iloc[0]

    print("\n" + "-" * 110)
    print("NİHAİ EN BAŞARILI MODEL")
    print("-" * 110)
    print(f"Model   : {en_iyi['Model']}")
    print(f"ROC-AUC : {en_iyi['Dış Test ROC-AUC']:.4f}")
    print(f"PR-AUC  : {en_iyi['Dış Test PR-AUC (AP)']:.4f}")
    print("=" * 110)

    print("\nNot:")
    print("• ROC-AUC ve PR-AUC değerleri büyük oldukça modelin ayırt etme performansı artmaktadır.")
    print("• F1-Skoru ve Recall değerleri büyük oldukça pozitif sınıfın doğru tespit edilme başarısı artmaktadır.")
    print("• Brier Score değeri 0'a yaklaştıkça modelin olasılık tahminlerinin kalitesi artmaktadır.")

    return rapor_tablosu


def main():
    """
    Programın başlangıç noktası ve modüler akış yönetimi.
    """
    egitim_yolu = Path(__file__).resolve().parent.parent / "veriler" / "medical_appointments_train.csv"
    test_yolu = Path(__file__).resolve().parent.parent / "veriler" / "medical_appointments_test.csv"

    egitim_veri, test_veri = veri_setlerini_yukle(egitim_yolu, test_yolu)

    if egitim_veri is None or test_veri is None:
        return

    # Sütun yapısı güvenliği için kontrol (Kategorik verilerin sızmaması ve hata vermemesi için)
    nesne_sutunlari = egitim_veri.select_dtypes(include=["object"]).columns
    print(f"Bilgi: Eğitim setindeki ham kategorik sütunlar: {list(nesne_sutunlari)}")

    # Kalan kategorik ve icd dönüşümlerini uygula
    egitim_veri, test_veri = kalan_kategorik_ve_frekans_isleme(egitim_veri, test_veri)

    # Hedef Değişken Haritalama ve Sıkı NaN Denetimi (y_test güvenliği için)
    kodlama_semasi = {"no": 0, "yes": 1}
    
    y_train = egitim_veri["no_show"].map(kodlama_semasi)
    y_test = test_veri["no_show"].map(kodlama_semasi)

    # Güvenlik Koruması (Assertion)
    assert y_train.isna().sum() == 0, "Hata: y_train içinde haritalanamayan NaN değerler mevcut!"
    assert y_test.isna().sum() == 0, "Hata: y_test içinde haritalanamayan NaN değerler mevcut!"

    X_train = egitim_veri.drop(columns=["no_show"])
    X_test = test_veri.drop(columns=["no_show"])

    # Dış Doğrulamayı Başlat
    sonuc_tablosu = dis_dogrulama_yurut(X_train, y_train, X_test, y_test)

    kayit_yolu = Path(__file__).resolve().parent.parent / "veriler" / "nihai_dis_dogrulama_sonuclari.csv"
    sonuc_tablosu.to_csv(kayit_yolu, index=False, encoding="utf-8-sig")
    print(f"\nNihai dış doğrulama sonuçları başarıyla kaydedildi: {kayit_yolu.name}")


if __name__ == "__main__":
    main()