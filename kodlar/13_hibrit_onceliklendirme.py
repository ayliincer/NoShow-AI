import pandas as pd
from pathlib import Path


def egitim_veri_setini_yukle(dosya_yolu: Path) -> pd.DataFrame:
    """
    Bir önceki adımdan (Adım 12) elde edilen ve tabakalı olarak ayrılan 
    eğitim veri setini, üzerinde hiçbir değişiklik yapmadan sisteme yükler.
    """
    try:
        veri = pd.read_csv(dosya_yolu)
        return veri
    except Exception as hata:
        print(f"\nEğitim veri seti yüklenirken hata oluştu:\n{hata}")
        return None


def eksik_veri_profilini_incele(veri: pd.DataFrame):
    """
    ANALİZ ADIMI

    Yalnızca eğitim veri seti (medical_appointments_train.csv) üzerindeki 
    eksik veri dağılımını, oranlarını ve veri tiplerini ampirik olarak hesaplar.
    Olası metodolojik aday stratejileri tablolaştırarak raporlar.
    """
    print("=" * 110)
    print("ANALİZ: İZOLE EDİLMİŞ EĞİTİM VERİ SETİ (medical_appointments_train.csv) EKSİK VERİ PROFİLLEMESİ")
    print("=" * 110)

    toplam_satir = len(veri)
    eksik_analiz_listesi = []

    for sutun in veri.columns:
        # İlgili sütunun ham serisindeki eksik değer (NaN/NaT) sayısı hesaplanır
        eksik_sayisi = veri[sutun].isnull().sum()
        eksik_yuzdesi = (eksik_sayisi / toplam_satir) * 100
        veri_tipi = veri[sutun].dtype

        # Metodolojik esneklik ilkesi gereği, peşin karar vermeden 
        # veri tipine göre değerlendirilebilecek olası aday stratejiler listelenir
        if eksik_sayisi > 0:
            if pd.api.types.is_numeric_dtype(veri_tipi):
                olasi_stratejiler = "Ortalama / Medyan / Model Tabanlı Doldurma Stratejileri Adaydır"
            elif pd.api.types.is_datetime64_any_dtype(veri_tipi) or sutun in ["appointment_date", "entry_service_date"]:
                olasi_stratejiler = "Kronolojik İleri/Geri Doldurma veya Sabit Zaman Atama Stratejileri Adaydır"
            else:
                olasi_stratejiler = "Mod (En Sık Değer) / Bağımsız Sınıf (Unknown) / Kategorik Kodlama Stratejileri Adaydır"
        else:
            olasi_stratejiler = "Eksik Değer Bulunmamaktadır (İşlem Gerekli Değildir)"

        eksik_analiz_listesi.append({
            "Değişken Adı": sutun,
            "Veri Tipi": str(veri_tipi),
            "Eksik Değer Sayısı": eksik_sayisi,
            "Eksik Değer Yüzdesi (%)": eksik_yuzdesi,
            "Değerlendirilebilecek Olası Stratejiler": olasi_stratejiler
        })

    # Elde edilen ampirik sonuçlar DataFrame formatına dönüştürülür
    eksik_tablosu = pd.DataFrame(eksik_analiz_listesi)
    
    # Eksik değer sayısına göre büyükten küçüğe sıralanır
    sirali_eksik_tablosu = eksik_tablosu.sort_values(by="Eksik Değer Sayısı", ascending=False)

    # SCI düzeyinde şeffaf hizalama ve raporlama ayarları
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    
    print(sirali_eksik_tablosu.to_string(index=False))
    print("=" * 110)


def main():
    """
    Programın başlangıç noktası ve modüler akış yönetimi.
    """
    # Adım 12'den gelen eğitim veri setinin güncellenmiş dinamik yolu
    egitim_veri_yolu = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical_appointments_train.csv"
    )

    egitim_verisi = egitim_veri_setini_yukle(egitim_veri_yolu)

    if egitim_verisi is None:
        return

    # Sadece eğitim verisi üzerinde eksik veri profil incelemesinin çalıştırılması
    eksik_veri_profilini_incele(egitim_verisi)


if __name__ == "__main__":
    main()