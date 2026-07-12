import pandas as pd
from pathlib import Path


def ham_veri_setini_yukle(dosya_yolu: str) -> pd.DataFrame:
    """
    Belirtilen dizindeki CSV dosyasını, üzerinde hiçbir dönüştürme 
    veya filtreleme işlemi yapmadan ham haliyle yükler.
    """
    try:
        ham_veri = pd.read_csv(dosya_yolu)
        return ham_veri
    except Exception as hata:
        print(f"\nVeri seti yüklenirken bir hata oluştu:\n{hata}")
        return None


def veri_sizintisi_risk_degerlendirmesi(veri: pd.DataFrame):
    """
    ANALİZ ADIMI

    Veri setindeki her bir değişkeni zamanlama mantığına göre listeler.
    Değişkenlerin randevu anındaki erişilebilirlik durumunu ve potansiyel 
    veri sızıntısı risk durumunu (Düşük/Orta/Yüksek) gerekçesiyle raporlar.
    """
    print("=" * 90)
    print("ANALİZ: HAM VERİ KRONOLOJİK VERİ SIZINTISI (DATA LEAKAGE) DEĞERLENDİRMESİ")
    print("=" * 90)

    # Veri setinde ampirik olarak varlığı tescillenmiş tüm sütunlar listelenir.
    # Her değişkenin anlamı ve ham zamanlama mantığı tek tek matris olarak kurgulanmıştır.
    risk_matrisi = [
        {
            "Değişken Adı": "gender",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Hastanın biyolojik cinsiyet bilgisi randevu oluşturulmadan önce sistemde mevcuttur."
        },
        {
            "Değişken Adı": "age",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Hastanın yaş bilgisi randevu kayıt anında sistem tarafından bilinmektedir."
        },
        {
            "Değişken Adı": "date_of_birth",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Hastanın doğum tarihi kaydı randevu kayıt anından önce sistemde yer almaktadır."
        },
        {
            "Değişken Adı": "under_12_years_old",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Yaş bilgisinden üretilen bu gösterge randevu anında teorik olarak hesaplanabilir durumdadir."
        },
        {
            "Değişken Adı": "over_60_years_old",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Yaş bilgisinden üretilen bu gösterge randevu anında teorik olarak hesaplanabilir durumdadir."
        },
        {
            "Değişken Adı": "entry_service_date",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Randevunun sisteme ilk girildiği talep tarihidir; işlem anında tescil edilir."
        },
        {
            "Değişken Adı": "appointment_date",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Randevunun gerçekleşeceği hedef tarihtir; randevu oluşturulurken belirlenir."
        },
        {
            "Değişken Adı": "appointment_month",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Randevu tarihine ait ay bilgisidir; randevu oluşturulurken takvimsel olarak bellidir."
        },
        {
            "Değişken Adı": "appointment_year",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Randevu tarihine ait yıl bilgisidir; randevu oluşturulurken takvimsel olarak bellidir."
        },
        {
            "Değişken Adı": "appointment_time",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Randevunun gerçekleşeceği gün içindeki saattir; randevu planlama anında atanır."
        },
        {
            "Değişken Adı": "appointment_shift",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Randevu saatine ait vardiya dilimidir; randevu planlama anında netleşmektedir."
        },
        {
            "Değişken Adı": "specialty",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Hastanın talep ettiği tıbbi branş bilgisidir; randevu oluşturulurken seçilir."
        },
        {
            "Değişken Adı": "city",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Sağlık merkezinin veya polikliniğin bulunduğu şehir lokasyonu randevu anında sabittir."
        },
        {
            "Değişken Adı": "patient_needs_companion",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Hastanın refakatçi ihtiyacı durumu randevu oluşturulduğu anda sistemde beyan edilmiştir."
        },
        {
            "Değişken Adı": "disability",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Hastanın sistemde kayıtlı engellilik profili randevu öncesinde tıbbi sicilinde yer alır."
        },
        {
            "Değişken Adı": "icd",
            "Erişilebilirlik": "Evet",
            "Risk Durumu": "Düşük",
            "Gerekçe": "Randevuya gerekçe olan veya hastanın geçmiş tıbbi kayıtlarındaki uluslararası tanı kodudur."
        },
        {
            "Değişken Adı": "average_temp_day",
            "Erişilebilirlik": "Hayır (Tahmini)",
            "Risk Durumu": "Orta",
            "Gerekçe": "Randevu gününe ait gerçekleşen hava sıcaklığıdır; randevu oluşturulduğu anda yalnızca tahmini olarak erişilebilir."
        },
        {
            "Değişken Adı": "max_temp_day",
            "Erişilebilirlik": "Hayır (Tahmini)",
            "Risk Durumu": "Orta",
            "Gerekçe": "Randevu gününe ait gerçekleşen maksimum sıcaklıktır; randevu anında yalnızca meteorolojik tahmin olarak mevcuttur."
        },
        {
            "Değişken Adı": "average_rain_day",
            "Erişilebilirlik": "Hayır (Tahmini)",
            "Risk Durumu": "Orta",
            "Gerekçe": "Randevu gününe ait gerçekleşen yağış miktarıdır; randevu oluşturma anında kesin değeri bilinemez."
        },
        {
            "Değişken Adı": "max_rain_day",
            "Erişilebilirlik": "Hayır (Tahmini)",
            "Risk Durumu": "Orta",
            "Gerekçe": "Randevu gününe ait gerçekleşen en yüksek yağış miktarıdır; randevu anında kesin değeri bilinemez."
        },
        {
            "Değişken Adı": "rainy_day_before",
            "Erişilebilirlik": "Duruma Bağlı",
            "Risk Durumu": "Düşük / Orta",
            "Gerekçe": "Randevudan bir gün önceki yağış durumudur. Randevu kaydı ile gerçekleşme günü arasındaki farka göre erişilebilirliği değişir."
        },
        {
            "Değişken Adı": "storm_day_before",
            "Erişilebilirlik": "Duruma Bağlı",
            "Risk Durumu": "Düşük / Orta",
            "Gerekçe": "Randevudan bir gün önceki fırtına durumudur. Randevu kaydı ile gerçekleşme günü arasındaki farka göre erişilebilirliği değişir."
        },
        {
            "Değişken Adı": "rain_intensity",
            "Erişilebilirlik": "Hayır (Tahmini)",
            "Risk Durumu": "Orta",
            "Gerekçe": "Randevu gününe ait gerçekleşen yağış yoğunluğu kategorisidir; randevu anında kesin değeri bilinemez."
        },
        {
            "Değişken Adı": "heat_intensity",
            "Erişilebilirlik": "Hayır (Tahmini)",
            "Risk Durumu": "Orta",
            "Gerekçe": "Randevu gününe ait gerçekleşen sıcaklık yoğunluğu kategorisidir; randevu anında kesin değeri bilinemez."
        },
        {
            "Değişken Adı": "no_show_reason",
            "Erişilebilirlik": "Hayır",
            "Risk Durumu": "Yüksek",
            "Gerekçe": "Hastanın randevuya gelmeme gerekçesidir. Bu bilgi ancak randevu süreci tamamlandıktan ve 'no_show=yes' gerçekleştikten sonra girilebilir."
        }
    ]

    # Sonuçların pandas DataFrame yapısına aktarılması
    rapor_tablosu = pd.DataFrame(risk_matrisi)
    
    # Mevcut veri setindeki sütun isimleri ile eşleşen alanlar doğrulanır
    mevcut_sutunlar = veri.columns.tolist()
    rapor_tablosu["Veri Setinde Mevcut mu?"] = rapor_tablosu["Değişken Adı"].apply(
        lambda x: "Evet" if x in mevcut_sutunlar else "Hayır"
    )

    # Rapor ekran çıktısı biçimlendirmesi
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    
    print(rapor_tablosu.to_string(index=False))
    print("=" * 90)


def main():
    """
    Programın başlangıç noktası ve modüler akış yönetimi.
    """
    # Proje dizin yapısına tam uyumlu dinamik veri yolu tanımı
    veri_yolu = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical-appointments-no-show-en.csv"
    )

    ham_veri = ham_veri_setini_yukle(veri_yolu)

    if ham_veri is None:
        return

    # Veri sızıntısı risk değerlendirme fonksiyonunu çalıştırma
    veri_sizintisi_risk_degerlendirmesi(ham_veri)


if __name__ == "__main__":
    main()