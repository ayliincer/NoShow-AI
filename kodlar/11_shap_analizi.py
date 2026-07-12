import pandas as pd
from pathlib import Path


def ararapor_veri_setini_yukle(dosya_yolu: Path) -> pd.DataFrame:
    """
    Bir önceki adımdan (Adım 10) elde edilen ve kaydedilen ara veri setini, 
    üzerinde hiçbir ek manipülasyon yapmadan sisteme yükler.
    """
    try:
        veri = pd.read_csv(dosya_yolu)
        return veri
    except Exception as hata:
        print(f"\nAra veri seti yüklenirken hata oluştu:\n{hata}")
        return None


def zaman_tiplerini_donustur_ve_dogrula(veri: pd.DataFrame) -> pd.DataFrame:
    """
    ANALİZ VE ÖN İŞLEME ADIMI

    Metinsel (string) tipteki tarih sütunlarını datetime formatına dönüştürür.
    Hatalı/eksik yazılmış tarih formatlarını NaT (eksik veri) biçimine zorlar (coerce)
    ve bu durumdaki gözlem sayılarını ampirik olarak raporlar. Satır silme yapmaz.
    """
    print("=" * 90)
    print("İŞLEM: METİNSEL TAKVİMSEL DEĞİŞKENLERİN VERİ TİPİ DÖNÜŞÜMÜ VE DOĞRULANMASI")
    print("=" * 90)

    hedef_zaman_sutunlari = ["appointment_date", "entry_service_date"]
    
    print("[Dönüşüm Öncesi Ham Veri Tipleri]")
    print("-" * 45)
    for sutun in hedef_zaman_sutunlari:
        if sutun in veri.columns:
            print(f"Değişken: {sutun:<20} | Mevcut Tip: {veri[sutun].dtype}")
    print("-" * 45)

    print("\n[Biçim Uyumsuzluğu Kontrolü ve Dönüşüm]")
    print("-" * 45)
    for sutun in hedef_zaman_sutunlari:
        if sutun in veri.columns:
            # Dönüşüm öncesi serideki orijinal eksik veri sayısı
            eski_eksik = veri[sutun].isnull().sum()
            
            # errors="coerce" ile formata uymayan gürültülü metinler NaT biçimine dönüştürülür.
            # Böylece kod durmaz ve veri kaybı (satır silme) yaşanmaz.
            veri[sutun] = pd.to_datetime(veri[sutun], format="%d/%m/%Y", errors="coerce")
            
            # Dönüşüm sonrası oluşan yeni eksik veri sayısı
            yeni_eksik = veri[sutun].isnull().sum()
            format_hatali_sayisi = yeni_eksik - eski_eksik
            
            print(f"Değişken: {sutun:<20} -> Formatı Hatalı Olan Gözlem Sayısı (NaT Yapılan): {format_hatali_sayisi}")

    print("-" * 45)
    print("\n[Dönüşüm Sonrası Akademik Tip Tescili]")
    print("-" * 45)
    for sutun in hedef_zaman_sutunlari:
        if sutun in veri.columns:
            print(f"Değişken: {sutun:<20} | Yeni Tip: {veri[sutun].dtype}")
    print("-" * 45)

    # Kronolojik tutarlılık kontrolü ampirik olarak ekrana basılır (Eksik/NaT veriler hesaplamaya katılmaz)
    if "appointment_date" in veri.columns and "entry_service_date" in veri.columns:
        gecersiz_kronoloji_sayisi = (veri["appointment_date"] < veri["entry_service_date"]).sum()
        print(f"\nAmpirik Durum Kontrolü:")
        print(f"  Randevu Tarihi, Talep Tarihinden Önce Olan Gözlem Sayısı: {gecersiz_kronoloji_sayisi}")
    
    print("=" * 90)
    return veri


def temiz_veriyi_kaydet(veri: pd.DataFrame, dosya_yolu: Path):
    """
    Dönüşümleri tamamlanmış ara veri setini bir sonraki bağımsız adıma 
    (12_veri_seti_bolme.py) aktarmak üzere diske kaydeder.
    """
    veri.to_csv(dosya_yolu, index=False, encoding="utf-8-sig")
    print(f"\nZaman dönüşümleri tescil edilen veri seti başarıyla kaydedildi.")
    print(f"Hedef Dosya: {dosya_yolu}")


def main():
    """
    Programın başlangıç noktası ve modüler akış yönetimi.
    """
    girdi_yolu = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical_appointments_preprocessed_step01.csv"
    )

    veri = ararapor_veri_setini_yukle(girdi_yolu)

    if veri is None:
        return

    donusturulmus_veri = zaman_tiplerini_donustur_ve_dogrula(veri)

    cikiti_yolu = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical_appointments_preprocessed_step02.csv"
    )

    temiz_veriyi_kaydet(donusturulmus_veri, cikiti_yolu)


if __name__ == "__main__":
    main()