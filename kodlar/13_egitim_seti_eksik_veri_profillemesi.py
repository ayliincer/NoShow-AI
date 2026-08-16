import pandas as pd
from pathlib import Path


def egitim_veri_setini_yukle(dosya_yolu: Path) -> pd.DataFrame:
    try:
        veri = pd.read_csv(dosya_yolu)
        return veri
    except Exception as hata:
        print(f"\nEğitim veri seti yüklenirken hata oluştu:\n{hata}")
        return None


def eksik_veri_profilini_incele(veri: pd.DataFrame):
    print("=" * 110)
    print("ANALİZ: İZOLE EDİLMİŞ EĞİTİM VERİ SETİ (medical_appointments_train.csv) EKSİK VERİ PROFİLLEMESİ")
    print("=" * 110)

    toplam_satir = len(veri)
    eksik_analiz_listesi = []

    for sutun in veri.columns:
        eksik_sayisi = veri[sutun].isnull().sum()
        eksik_yuzdesi = (eksik_sayisi / toplam_satir) * 100
        veri_tipi = veri[sutun].dtype

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

    eksik_tablosu = pd.DataFrame(eksik_analiz_listesi)
    sirali_eksik_tablosu = eksik_tablosu.sort_values(by="Eksik Değer Sayısı", ascending=False)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    
    print(sirali_eksik_tablosu.to_string(index=False))
    print("\nNot:")
    print("Bu analiz yalnızca eksik veri profillerini ve uygun doldurma stratejilerini belirlemek amacıyla gerçekleştirilmiştir.")
    print("Bu aşamada herhangi bir eksik değer doldurma işlemi uygulanmamıştır.")
    print("Nihai doldurma yöntemi yalnızca eğitim veri seti üzerinde sonraki adımlarda gerçekleştirilecektir.")
    print("=" * 110)


def main():
    egitim_veri_yolu = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical_appointments_train.csv"
    )

    egitim_verisi = egitim_veri_setini_yukle(egitim_veri_yolu)

    if egitim_verisi is None:
        return

    eksik_veri_profilini_incele(egitim_verisi)


if __name__ == "__main__":
    main()