import pandas as pd
import numpy as np
from pathlib import Path


def veri_setlerini_yukle(egitim_yolu: Path, test_yolu: Path) -> tuple:
    """
    Eğitim ve test veri setlerini diske kayıtlı CSV dosyalarından yükler.

    Parametreler
    ------------
    egitim_yolu : Path
        Eğitim veri setinin yolu.
    test_yolu : Path
        Test veri setinin yolu.

    Döndürür
    --------
    tuple (pd.DataFrame, pd.DataFrame)
        Yüklenen eğitim ve test veri setleri.
    """
    try:
        egitim_veri = pd.read_csv(egitim_yolu)
        test_veri = pd.read_csv(test_yolu)
        return egitim_veri, test_veri
    except Exception as hata:
        print(f"\nVeri setleri yüklenirken hata oluştu:\n{hata}")
        return None, None


def ozellikleri_uret(veri: pd.DataFrame) -> tuple:
    """
    Veri seti üzerinde mantıksal ve bilimsel olarak anlamlı yeni bağımsız 
    değişkenler türetir. Kronolojik anomalileri (negatif lead time) 0 değerine 
    yuvarlayarak (clipping) düzeltir.

    Parametreler
    ------------
    veri : pd.DataFrame
        Girdi veri seti.

    Döndürür
    --------
    tuple (pd.DataFrame, int)
        Yeni öznitelikler eklenmiş veri seti ve düzeltilen anomali sayısı.
    """
    yeni_veri = veri.copy()

    # 1. Tarih özniteliklerinin datetime tipine zorlanması
    yeni_veri["appointment_date"] = pd.to_datetime(yeni_veri["appointment_date"], format="%Y-%m-%d")
    yeni_veri["entry_service_date"] = pd.to_datetime(yeni_veri["entry_service_date"], format="%Y-%m-%d")

    # 2. Bekleme Süresi (Lead Time) Hesabı
    lead_time_serisi = (yeni_veri["appointment_date"] - yeni_veri["entry_service_date"]).dt.days
    
    # Kronolojik olarak imkansız olan negatif lead_time durumlarının (örn: entry > appointment) tespiti
    anomali_sayisi = (lead_time_serisi < 0).sum()
    
    # Negatif lead_time değerlerini 0 (aynı gün) olacak şekilde kırpma (clipping)
    yeni_veri["lead_time"] = lead_time_serisi.clip(lower=0)

    # 3. Aynı Gün Rezervasyon Göstergesi (Is Same Day)
    yeni_veri["is_same_day"] = (yeni_veri["lead_time"] == 0).astype(int)

    # 4. Randevu Günü (Day of Week)
    yeni_veri["appointment_day_of_week"] = yeni_veri["appointment_date"].dt.dayofweek

    # 5. Hafta Sonu Göstergesi (Is Weekend)
    # 0: Pazartesi ... 5: Cumartesi, 6: Pazar
    HAFTA_SONU_SINIRI = 5
    yeni_veri["is_weekend"] = (yeni_veri["appointment_day_of_week"] >= HAFTA_SONU_SINIRI).astype(int)

    # 6. Randevu Saati (Appointment Hour)
    yeni_veri["appointment_hour"] = pd.to_datetime(yeni_veri["appointment_time"], format="%H:%M").dt.hour

    # 7. Günlük Sıcaklık Değişim Aralığı (Temp Range)
    if "max_temp_day" in yeni_veri.columns and "average_temp_day" in yeni_veri.columns:
        yeni_veri["temp_range"] = yeni_veri["max_temp_day"] - yeni_veri["average_temp_day"]

    # 8. Günlük Yağış Değişim Aralığı (Rain Range)
    if "max_rain_day" in yeni_veri.columns and "average_rain_day" in yeni_veri.columns:
        yeni_veri["rain_range"] = yeni_veri["max_rain_day"] - yeni_veri["average_rain_day"]

    # 9. Yağmurlu Gün Göstergesi (Is Rainy)
    if "average_rain_day" in yeni_veri.columns:
        yeni_veri["is_rainy"] = (yeni_veri["average_rain_day"] > 0.0).astype(int)

    return yeni_veri, anomali_sayisi


def akademik_rapor_yazdir(ham_veri: pd.DataFrame, yeni_veri: pd.DataFrame, anomali_sayisi: int, kume_adi: str):
    """
    Türetilen yeni değişkenleri ve düzeltilen kronolojik hataları akademik formatta terminale yazdırır.
    """
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
    """
    Türetilen öznitelikleri içeren veri setini, tarih tiplerini metin formatına 
    geri dönüştürerek diske yazar.
    """
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

    # Verileri yükleme
    egitim_ham, test_ham = veri_setlerini_yukle(egitim_yolu, test_yolu)
    if egitim_ham is None or test_ham is None:
        return

    # Özellik mühendisliği işlemlerini gerçekleştirme (Eğitim ve Test için ayrı ayrı)
    egitim_yeni, egitim_anomali = ozellikleri_uret(egitim_ham)
    test_yeni, test_anomali = ozellikleri_uret(test_ham)

    # Akademik raporları terminale basma
    akademik_rapor_yazdir(egitim_ham, egitim_yeni, egitim_anomali, "Eğitim (Train)")
    akademik_rapor_yazdir(test_ham, test_yeni, test_anomali, "Test (Test)")

    # Güncellenmiş veri setlerini kaydetme
    veriyi_kaydet(egitim_yeni, egitim_yolu)
    veriyi_kaydet(test_yeni, test_yolu)


if __name__ == "__main__":
    main()
