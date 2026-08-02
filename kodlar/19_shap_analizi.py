import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# Modeller ve Araçlar
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler


def veri_setlerini_yukle(egitim_yolu: Path, test_yolu: Path) -> tuple:
    try:
        egitim = pd.read_csv(egitim_yolu)
        test = pd.read_csv(test_yolu)
        return egitim, test
    except Exception as hata:
        print(f"\nVeri setleri yüklenirken hata oluştu:\n{hata}")
        return None, None


def kalan_kategorik_ve_frekans_isleme(egitim: pd.DataFrame, test: pd.DataFrame) -> tuple:
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


def en_iyi_modeli_tespit_et(sonuc_yolu: Path) -> str:
    try:
        sonuclar = pd.read_csv(sonuc_yolu)
        en_iyi_satir = sonuclar.loc[sonuclar["Dış Test ROC-AUC"].idxmax()]
        en_iyi_model_adi = en_iyi_satir["Model"]
        en_iyi_skor = en_iyi_satir["Dış Test ROC-AUC"]
        print(f"Bilgi: CSV Analiz Edildi. Dış Doğrulama Şampiyonu: {en_iyi_model_adi} (ROC-AUC: {en_iyi_skor:.6f})")
        return en_iyi_model_adi
    except Exception as hata:
        print(f"Uyarı: Sonuç CSV'si okunamadı, varsayılan olarak 'Random Forest' seçildi. Hata: {hata}")
        return "Random Forest"


def dinamik_model_olustur(model_adi: str, y_tren: pd.Series):
    sinif_sayilari = y_tren.value_counts()
    scale_pos_weight = sinif_sayilari[0] / sinif_sayilari[1]

    model_havuzu = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42, n_jobs=-1),
        "LightGBM": LGBMClassifier(class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1),
        "CatBoost": CatBoostClassifier(auto_class_weights="Balanced", random_state=42, verbose=False)
    }
    return model_havuzu.get(model_adi, model_havuzu["Random Forest"])


def olceklendirmeyi_uygula(model_adi: str, X_tren: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    X_tren_kullanilan = X_tren.copy()
    X_test_kullanilan = X_test.copy()

    if model_adi == "Logistic Regression":
        surekli_sutunlar = ["age", "average_temp_day", "average_rain_day", "max_temp_day", "max_rain_day"]
        mevcut_surekliler = [s for s in surekli_sutunlar if s in X_tren.columns]

        if mevcut_surekliler:
            print("Bilgi: Şampiyon model Lojistik Regresyon olduğu için StandardScaler "
                  "(yalnızca eğitim setinden fit) uygulanıyor...")
            olcekleyici = StandardScaler()
            X_tren_kullanilan[mevcut_surekliler] = olcekleyici.fit_transform(X_tren[mevcut_surekliler])
            X_test_kullanilan[mevcut_surekliler] = olcekleyici.transform(X_test[mevcut_surekliler])

    return X_tren_kullanilan, X_test_kullanilan


def shap_analizi_yurut(X_tren: pd.DataFrame, y_tren: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, en_iyi_model_adi: str, gorsel_klasor: Path):
    print("=" * 110)
    print("İŞLEM: DİNAMİK AÇIKLANABİLİR YAPAY ZEKA (XAI) - SHAP ANALİZİ VE TESCİLİ")
    print("=" * 110)

    X_tren_kullanilan, X_test_kullanilan = olceklendirmeyi_uygula(en_iyi_model_adi, X_tren, X_test)

    model = dinamik_model_olustur(en_iyi_model_adi, y_tren)
    print(f"Nihai model ({en_iyi_model_adi}) eğitim kümesinde eğitiliyor...")
    model.fit(X_tren_kullanilan, y_tren)
    print(">> Model eğitimi tamamlandı.")

    ornek_boyutu = min(500, len(X_test_kullanilan))
    X_test_ornek = X_test_kullanilan.sample(n=ornek_boyutu, random_state=42)

    print("SHAP Açıklayıcısı (Explainer) hazırlanıyor...")

    if en_iyi_model_adi == "Logistic Regression":
        arka_plan_boyutu = min(200, len(X_tren_kullanilan))
        X_tren_arka_plan = X_tren_kullanilan.sample(
            n=arka_plan_boyutu,
            random_state=42
        )

        print(">> LinearExplainer oluşturuluyor...")

        aciklayici = shap.LinearExplainer(
            model,
            X_tren_arka_plan
        )

        print(">> LinearExplainer hazır.")
        print(">> SHAP değerleri hesaplanıyor...")

        shap_degerleri = aciklayici(X_test_ornek)

        print(">> SHAP değerleri hesaplandı.")

    else:
        print(">> TreeExplainer oluşturuluyor...")

        aciklayici = shap.TreeExplainer(model)

        print(">> TreeExplainer hazır.")
        print(">> SHAP değerleri hesaplanıyor...")

        shap_degerleri = aciklayici(
            X_test_ornek,
            check_additivity=False
        )

    print(">> SHAP değerleri hesaplandı.")


    if len(shap_degerleri.values.shape) == 3:
        shap_degerleri_gorsel = shap_degerleri[:, :, 1]
    else:
        shap_degerleri_gorsel = shap_degerleri

    plt.figure(figsize=(12, 8))

    shap.plots.beeswarm(
        shap_degerleri_gorsel,
        max_display=20,
        show=False
    )

    plt.title(
        "SHAP Summary Plot (Random Forest)",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    plt.tight_layout()

    gorsel_klasor.mkdir(parents=True, exist_ok=True)
    grafik_yolu = gorsel_klasor / "shap_summary_plot.png"

    plt.savefig(
        grafik_yolu,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


    plt.figure(figsize=(10, 9))

    shap.plots.bar(
        shap_degerleri_gorsel,
        max_display=30,
        show=False
    )

    plt.title(
        "Global Feature Importance (Random Forest - SHAP)",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    plt.tight_layout()

    bar_grafik_yolu = gorsel_klasor / "shap_bar_plot.png"

    plt.savefig(
        bar_grafik_yolu,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n" + "-" * 90)
    print("DİNAMİK SHAP ANALİZİ TAMAMLANDI")
    print("-" * 90)
    print(f"-> Analiz Edilen Şampiyon Model     : {en_iyi_model_adi}")
    print(f"-> SHAP Analizinde Kullanılan Örneklem : {ornek_boyutu} test gözlemi")
    print(f"-> SHAP Summary Plot     : {grafik_yolu}")
    print(f"-> SHAP Feature Importance Plot : {bar_grafik_yolu}")
    print(f"-> Grafik Klasörü        : {gorsel_klasor}")
    print("=" * 110)


def main():
    egitim_yolu = Path(__file__).resolve().parent.parent / "veriler" / "medical_appointments_train.csv"
    test_yolu = Path(__file__).resolve().parent.parent / "veriler" / "medical_appointments_test.csv"
    sonuc_yolu = Path(__file__).resolve().parent.parent / "veriler" / "nihai_dis_dogrulama_sonuclari.csv"

    gorsel_klasor = Path(__file__).resolve().parent.parent / "gorseller"

    egitim_veri, test_veri = veri_setlerini_yukle(egitim_yolu, test_yolu)

    if egitim_veri is None or test_veri is None:
        return

    egitim_veri, test_veri = kalan_kategorik_ve_frekans_isleme(egitim_veri, test_veri)

    kodlama_semasi = {"no": 0, "yes": 1}
    y_train = egitim_veri["no_show"].map(kodlama_semasi)
    y_test = test_veri["no_show"].map(kodlama_semasi)

    X_train = egitim_veri.drop(columns=["no_show"])
    X_test = test_veri.drop(columns=["no_show"])

    en_iyi_model_adi = en_iyi_modeli_tespit_et(sonuc_yolu)


    shap_analizi_yurut(X_train, y_train, X_test, y_test, en_iyi_model_adi, gorsel_klasor)


if __name__ == "__main__":
    main()