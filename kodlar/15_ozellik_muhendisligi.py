import pandas as pd
import numpy as np
from pathlib import Path


def veri_setlerini_yukle(egitim_yolu: Path, test_yolu: Path) -> tuple:
    try:
        egitim_veri = pd.read_csv(egitim_yolu)
        test_veri = pd.read_csv(test_yolu)
        return egitim_veri, test_veri
    except Exception as hata:
        print(f"\nVeri setleri yüklenirken hata oluştu:\n{hata}")
        return None, None


def ozellikleri_uret(veri: pd.DataFrame) -> tuple:
    yeni_veri = veri.copy()

    yeni_veri["appointment_date"] = pd.to_datetime(yeni_veri["appointment_date"], format="%Y-%m-%d")
    yeni_veri["entry_service_date"] = pd.to_datetime(yeni_veri["entry_service_date"], format="%Y-%m-%d")

    lead_time_serisi = (yeni_veri["appointment_date"] - yeni_veri["entry_service_date"]).dt.days
    
    anomali_sayisi = (lead_time_serisi < 0).sum()
    
    yeni_veri["lead_time"] = lead_time_serisi.clip(lower=0)

    yeni_veri["is_same_day"] = (yeni_veri["lead_time"] == 0).astype(int)

    yeni_veri["appointment_day_of_week"] = yeni_veri["appointment_date"].dt.dayofweek

    HAFTA_SONU_SINIRI = 5
    yeni_veri["is_weekend"] = (yeni_veri["appointment_day_of_week"] >= HAFTA_SONU_SINIRI).astype(int)

    yeni_veri["appointment_hour"] = pd.to_datetime(yeni_veri["appointment_time"], format="%H:%M").dt.hour

    if "max_temp_day" in yeni_veri.columns and "average_temp_day" in yeni_veri.columns:
        yeni_veri["temp_range"] = yeni_veri["max_temp_day"] - yeni_veri["average_temp_day"]

    if "max_rain_day" in yeni_veri.columns and "average_rain_day" in yeni_veri.columns:
        yeni_veri["rain_range"] = yeni_veri["max_rain_day"] - yeni_veri["average_rain_day"]

    if "average_rain_day" in yeni_veri.columns:
        yeni_veri["is_rainy"] = (yeni_veri["average_rain_day"] > 0.0).astype(int)

    return yeni_veri, anomali_sayisi


def akademik_rapor_yazdir(ham_veri: pd.DataFrame, yeni_veri: pd.DataFrame, anomali_sayisi: int, kume_adi: str):
    print("\n" + "=" * 110)
    print(f"AKADEMİK RAPOR: {kume_adi.upper()} VERİ KÜMESİ ÖZELLİK MÜHENDİSLİĞİ (FEATURE ENGINEERING)")
    print("=" * 110)

    eklenen_sutunlar = [
        "lead_time",
        "is_same_day",
        "appointment_day_of_week",
        "is_weekend",
        "appointment_hour",
        "temp_range",
        "rain_range",
        "is_rainy"
    ]

    mevcut_eklenenler = [
        sutun for sutun in eklenen_sutunlar
        if sutun in yeni_veri.columns
    ]

    print(f"Toplam Satır Sayısı                 : {len(yeni_veri):,}")
    print(f"Başlangıç Sütun Sayısı              : {len(ham_veri.columns)}")
    print(f"Nihai Sütun Sayısı                  : {len(yeni_veri.columns)}")
    print(f"Eklenen Yeni Özellik Sayısı         : {len(mevcut_eklenenler)}")
    print(f"Kronolojik Anomali (Düzeltilen)     : {anomali_sayisi} gözlem (Kırpılarak 0 yapıldı)")
    print("-" * 110)
    print("EKLENEN YENİ ÖZELLİKLER VE TİPLERİ:")
    print("-" * 45)
    for sutun in mevcut_eklenenler:
        print(f"- {sutun:<30} | Veri Tipi: {str(yeni_veri[sutun].dtype)}")
    print("=" * 110)


def veriyi_kaydet(veri: pd.DataFrame, dosya_yolu: Path):
    kayit_kopya = veri.copy()
    
    tarih_kolonlar = ["appointment_date", "entry_service_date", "date_of_birth"]
    for sutun in tarih_kolonlar:
        if sutun in kayit_kopya.columns and pd.api.types.is_datetime64_any_dtype(kayit_kopya[sutun]):
            if sutun == "date_of_birth":
                kayit_kopya[sutun] = kayit_kopya[sutun].dt.strftime("%d/%m/%Y")
            else:
                kayit_kopya[sutun] = kayit_kopya[sutun].dt.strftime("%Y-%m-%d")

    kayit_kopya.to_csv(dosya_yolu, index=False, encoding="utf-8-sig")
    print(f"Özellik mühendisliği tamamlanan veri seti başarıyla kaydedildi: {dosya_yolu.name}")
    


def main():
    proje_dizini = Path(__file__).resolve().parent.parent
    veriler_dizini = proje_dizini / "veriler"
    
    egitim_yolu = veriler_dizini / "medical_appointments_train.csv"
    test_yolu = veriler_dizini / "medical_appointments_test.csv"

    egitim_ham, test_ham = veri_setlerini_yukle(egitim_yolu, test_yolu)
    if egitim_ham is None or test_ham is None:
        return

    egitim_yeni, egitim_anomali = ozellikleri_uret(egitim_ham)
    test_yeni, test_anomali = ozellikleri_uret(test_ham)
    akademik_rapor_yazdir(egitim_ham, egitim_yeni, egitim_anomali, "Eğitim (Train)")
    akademik_rapor_yazdir(test_ham, test_yeni, test_anomali, "Test (Test)")
    veriyi_kaydet(egitim_yeni, egitim_yolu)
    veriyi_kaydet(test_yeni, test_yolu)
    print("\nNot:")
    print("Yeni özellikler yalnızca randevu oluşturma anında erişilebilir bilgiler kullanılarak üretilmiştir.")
    print("Hedef değişken (no_show) veya geleceğe ait bilgilerden hiçbir türetilmiş özellik oluşturulmamıştır.")


if __name__ == "__main__":
    main()
