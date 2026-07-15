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
    """
    Kategorik kodlaması (Adım 16) tamamlanmış nihai eğitim veri setini yükler.
    Parametreleri (scaler ve frekans) bu set üzerinden öğreneceğiz.
    """
    try:
        return pd.read_csv(egitim_yolu)
    except Exception as hata:
        print(f"\nEğitim veri seti yüklenirken hata oluştu:\n{hata}")
        return None


def en_iyi_modeli_tespit_et(sonuc_yolu: Path) -> str:
    """
    18. adımdan üretilen dış doğrulama CSV dosyasını okuyarak 'Dış Test ROC-AUC'
    sütununa göre ampirik olarak en başarılı modeli dinamik olarak seçer.

    Bu fonksiyon, 19. adımdaki (SHAP) mantıkla birebir aynıdır; böylece
    18-19-20 adımları arasında model tutarlılığı garanti altına alınır.
    Dosya bulunamazsa veya okunamazsa, kod asla hata vermeden güvenli
    varsayılan olan "Random Forest" ile devam eder.
    """
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
    """
    Seçilen şampiyon modele ait nesneyi, 18. adımdaki hiperparametreler
    ve sınıf ağırlıklarıyla dinamik olarak kurar. Tanınmayan bir model adı
    gelirse (örn. CSV bozuksa), kod hata vermeden güvenli varsayılan olan
    Random Forest'a düşer.
    """
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
    """
    ÜRETİM / CANLIYA ALMA ADIMI

    1. Eğitim setinden ICD Frekans Haritasını çıkarır ve kaydeder.
    2. ICD dönüşümü ve zaman sütunu temizliği sonrası, YALNIZCA şampiyon model
       Lojistik Regresyon ise StandardScaler'ı fit eder (18. ve 19. adımla
       tutarlı, sızıntısız koşullu ölçeklendirme).
    3. 18. adımda dış doğrulamada en başarılı çıkan şampiyon modeli
       (bulunamazsa güvenli varsayılan Random Forest'ı) tüm eğitim seti
       üzerinde eğitir.
    4. Tüm bu bileşenleri tek bir sözlük (dictionary) halinde .joblib olarak diske yazar.
    """
    print("=" * 110)
    print("İŞLEM: ŞAMPİYON MODELİN VE ÖN İŞLEME PARAMETRELERİNİN DİSKE TESCİL EDİLMESİ (SERIALIZATION)")
    print("=" * 110)

    # 1. Ham zaman sütununun atılması
    ham_zaman_sutunu = "appointment_time"
    if ham_zaman_sutunu in egitim_veri.columns:
        egitim_veri = egitim_veri.drop(columns=[ham_zaman_sutunu])

    # 2. ICD için sızıntısız Frequency Encoding parametrelerinin öğrenilmesi
    yuksek_kardinalite_sutunu = "icd"
    frekans_haritasi = None
    if yuksek_kardinalite_sutunu in egitim_veri.columns:
        # Frekans haritası öğrenilir (Canlı sistemde yeni gelen veriyi dönüştürmek için saklanacak)
        frekans_haritasi = egitim_veri[yuksek_kardinalite_sutunu].value_counts(normalize=True).to_dict()

        egitim_veri[f"{yuksek_kardinalite_sutunu}_frekans"] = egitim_veri[yuksek_kardinalite_sutunu].map(frekans_haritasi)
        egitim_veri = egitim_veri.drop(columns=[yuksek_kardinalite_sutunu])

    # Hedef ve öznitelik ayrımı
    kodlama_semasi = {"no": 0, "yes": 1}
    y_tren = egitim_veri["no_show"].map(kodlama_semasi)
    X_tren = egitim_veri.drop(columns=["no_show"])

    # 3. StandardScaler parametrelerinin KOŞULLU öğrenilmesi
    # Yalnızca şampiyon model Lojistik Regresyon ise scaler fit edilir.
    # Ağaç tabanlı modeller (RF, XGBoost, LightGBM, CatBoost, Decision Tree)
    # ölçeklendirmeye ihtiyaç duymadığından scaler None olarak paketlenir.
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

    # 4. Nihai Şampiyon Modelin (18. adımla tutarlı, dinamik seçilmiş) Eğitilmesi
    print(f"-> Şampiyon Algoritma: {en_iyi_model_adi}")
    print("-> Model tüm eğitim veri seti üzerinde nihai olarak eğitiliyor...")
    nihai_model = dinamik_model_olustur(en_iyi_model_adi, y_tren)
    nihai_model.fit(X_tren_nihai, y_tren)

    # 5. Canlıya Alma Paketinin (Artifact) Hazırlanması
    # Canlı sistemde (API veya Arayüz) yeni bir hasta geldiğinde, bu bileşenler sırayla çalışacaktır.
    # "scaler" None ise, inference kodu ölçeklendirme adımını atlamalıdır.
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

    # Dosya kaydı
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
    """
    Programın başlangıç noktası ve modüler akış yönetimi.
    """
    # Girdi dosya yolları
    egitim_yolu = Path(__file__).resolve().parent.parent / "veriler" / "medical_appointments_train.csv"
    sonuc_yolu = Path(__file__).resolve().parent.parent / "veriler" / "nihai_dis_dogrulama_sonuclari.csv"

    # Modelin kaydedileceği klasör ve dosya adı
    model_kayit_yolu = Path(__file__).resolve().parent.parent / "modeller" / "nihai_no_show_model_paketi.joblib"

    egitim_veri = veri_setlerini_yukle(egitim_yolu)

    if egitim_veri is None:
        return

    # 18. adımdan gelen gerçek şampiyonu tespit et (bulunamazsa güvenli varsayılan: Random Forest)
    en_iyi_model_adi = en_iyi_modeli_tespit_et(sonuc_yolu)

    # Nihai boru hattını çalıştır ve diske yaz
    nihai_bor_hatti_egit_ve_kaydet(egitim_veri, en_iyi_model_adi, model_kayit_yolu)


if __name__ == "__main__":
    main()