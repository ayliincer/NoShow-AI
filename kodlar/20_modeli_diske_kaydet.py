import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Modeller ve Araçlar
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler


def veri_setlerini_yukle(egitim_yolu: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(egitim_yolu)
    except Exception as hata:
        print(f"\nEğitim veri seti yüklenirken hata oluştu:\n{hata}")
        return None


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


def nihai_bor_hatti_egit_ve_kaydet(egitim_veri: pd.DataFrame, en_iyi_model_adi: str, model_kayit_yolu: Path):
    print("=" * 110)
    print("İŞLEM: ŞAMPİYON MODELİN VE ÖN İŞLEME PARAMETRELERİNİN DİSKE TESCİL EDİLMESİ (SERIALIZATION)")
    print("=" * 110)

    ham_zaman_sutunu = "appointment_time"
    if ham_zaman_sutunu in egitim_veri.columns:
        egitim_veri = egitim_veri.drop(columns=[ham_zaman_sutunu])

    yuksek_kardinalite_sutunu = "icd"
    frekans_haritasi = None
    if yuksek_kardinalite_sutunu in egitim_veri.columns:
        frekans_haritasi = egitim_veri[yuksek_kardinalite_sutunu].value_counts(normalize=True).to_dict()

        egitim_veri[f"{yuksek_kardinalite_sutunu}_frekans"] = egitim_veri[yuksek_kardinalite_sutunu].map(frekans_haritasi)
        egitim_veri = egitim_veri.drop(columns=[yuksek_kardinalite_sutunu])

    kodlama_semasi = {"no": 0, "yes": 1}
    y_tren = egitim_veri["no_show"].map(kodlama_semasi)
    X_tren = egitim_veri.drop(columns=["no_show"])

    surekli_sutunlar = ["age", "average_temp_day", "average_rain_day", "max_temp_day", "max_rain_day"]
    mevcut_surekliler = [s for s in surekli_sutunlar if s in X_tren.columns]

    olcekleyici = None
    X_tren_nihai = X_tren.copy()

    if en_iyi_model_adi == "Logistic Regression" and mevcut_surekliler:
        print("-> Şampiyon model Lojistik Regresyon: StandardScaler eğitim setinden fit ediliyor...")
        olcekleyici = StandardScaler()
        X_tren_nihai[mevcut_surekliler] = olcekleyici.fit_transform(X_tren[mevcut_surekliler])
    else:
        print("-> Şampiyon model ağaç tabanlı (veya sürekli sütun yok): StandardScaler atlanıyor.")

    print(f"-> Şampiyon Algoritma: {en_iyi_model_adi}")
    print("-> Model tüm eğitim veri seti üzerinde nihai olarak eğitiliyor...")
    nihai_model = dinamik_model_olustur(en_iyi_model_adi, y_tren)
    nihai_model.fit(X_tren_nihai, y_tren)

    canli_sistem_paketi = {
    "surum": "v1.0",
    "model_adi": en_iyi_model_adi,
    "icd_frekans_haritasi": frekans_haritasi,
    "scaler": olcekleyici,
    "surekli_sutunlar": mevcut_surekliler if olcekleyici is not None else [],
    "feature_count": len(X_tren.columns),
    "model": nihai_model,
    "sutun_siralamasi": list(X_tren.columns)
    }

    model_kayit_yolu.parent.mkdir(parents=True, exist_ok=True)
    print("-> Model paketi oluşturuluyor...")
    joblib.dump(canli_sistem_paketi, model_kayit_yolu)
    print("-> Model paketi başarıyla diske yazıldı.")

    print("\n" + "-" * 90)
    print("TESCİL İŞLEMİ BAŞARIYLA TAMAMLANDI")
    print("-" * 90)
    print(f"-> Kaydedilen Model     : {en_iyi_model_adi}")
    print(f"-> Kaydedilen Dosya     : {model_kayit_yolu.name}")
    print(f"-> Kayıt Konumu         : {model_kayit_yolu}")
    print(f"-> Scaler Durumu        : {'Aktif (StandardScaler kaydedildi)' if olcekleyici is not None else 'Kullanılmadı (None)'}")
    print("-> Paket İçeriği        : [Sürüm, Model Adı, Model, Scaler, Sürekli Sütunlar, ICD Frekans Haritası, Sütun Sıralaması]")
    print("=" * 110)


def main():
    egitim_yolu = Path(__file__).resolve().parent.parent / "veriler" / "medical_appointments_train.csv"
    sonuc_yolu = Path(__file__).resolve().parent.parent / "veriler" / "nihai_dis_dogrulama_sonuclari.csv"
    model_kayit_yolu = Path(__file__).resolve().parent.parent / "modeller" / "nihai_no_show_model_paketi.joblib"

    egitim_veri = veri_setlerini_yukle(egitim_yolu)

    if egitim_veri is None:
        return

    en_iyi_model_adi = en_iyi_modeli_tespit_et(sonuc_yolu)
    nihai_bor_hatti_egit_ve_kaydet(egitim_veri, en_iyi_model_adi, model_kayit_yolu)


if __name__ == "__main__":
    main()