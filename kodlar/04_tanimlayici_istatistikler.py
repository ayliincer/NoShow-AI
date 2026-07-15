import pandas as pd
from pathlib import Path


def veri_setini_yukle(dosya_yolu: Path) -> pd.DataFrame:
    """
    Veri setini hiçbir ön işleme veya dönüşüm uygulamadan
    ham haliyle yükler.
    """
    try:
        veri = pd.read_csv(dosya_yolu)
        return veri
    except Exception as hata:
        print(f"\nVeri seti yüklenirken hata oluştu:\n{hata}")
        return None


def yinelenen_kayit_tespiti(veri: pd.DataFrame):
    """
    ANALİZ ADIMI

    Veri setindeki tam mükerrer satır sayısını, oranını ve olası 
    kimlik bazlı tekrarları frekans düzeyinde hesaplayarak raporlar.
    """
    print("=" * 90)
    print("ANALİZ: HAM VERİ YİNELENEN KAYIT (DUPLICATE RECORD) ANALİZİ")
    print("=" * 90)

    toplam_satir = len(veri)
    toplam_sutun = veri.shape[1]

    print(f"Toplam Satır Sayısı    : {toplam_satir:,}")
    print(f"Toplam Değişken Sayısı : {toplam_sutun}")
    print()

    # 1. Tüm sütunlar bazında birebir aynı olan tam mükerrer satırların hesabı
    tam_mukerrer_sayisi = veri.duplicated().sum()
    tam_mukerrer_orani = (tam_mukerrer_sayisi / toplam_satir) * 100

    print(f"Tam Mükerrer Satır Sayısı     : {tam_mukerrer_sayisi:,}")
    print(f"Tam Mükerrer Satır Oranı (%)  : {tam_mukerrer_orani:.4f}")
    print("-" * 90)

    # 2. Veri setinde hasta veya randevu belirteci olabilecek kimlik alanlarının 
    # ham frekans düzeyinde mükerrerlik durumlarının tespiti
    print("\n[Olası Kimlik Değişkenlerinin Tekrar Frekansları]")
    print("-" * 90)
    
    # Ham veri şemasında yer alan olası kimlik öznitelikleri kontrol edilir
    olasi_kimlikler = ["PatientId", "AppointmentID", "patient_id", "appointment_id"]
    
    mevcut_kimlikler = [sutun for sutun in olasi_kimlikler if sutun in veri.columns]
    print(f"Tespit Edilen Kimlik Değişkeni Sayısı : {len(mevcut_kimlikler)}")
    print()

    if mevcut_kimlikler:
        for kimlik in mevcut_kimlikler:
            kimlik_tekrar_sayisi = veri.duplicated(subset=[kimlik]).sum()
            kimlik_tekrar_orani = (kimlik_tekrar_sayisi / toplam_satir) * 100

            benzersiz_kimlik = veri[kimlik].nunique()
            ortalama_kayit = len(veri) / benzersiz_kimlik

            print(f"Değişken: {kimlik}")
            print(f"  Yinelenen Gözlem Sayısı     : {kimlik_tekrar_sayisi:,}")
            print(f"  Yinelenen Gözlem Oranı (%)  : {kimlik_tekrar_orani:.4f}")
            print(f"  Benzersiz Kimlik Sayısı     : {benzersiz_kimlik:,}")
            print(f"  Kimlik Başına Ortalama Kayıt: {ortalama_kayit:.2f}")
            print("." * 45)
        else:
            print("Not: Belirtilen spesifik kimlik sütun isimleri ham veri setinde bulunamamıştır.")
            print("Mevcut tüm sütun listesi üzerinden genel kontrol önerilir.")
        
    print("=" * 90)


def main():
    """
    Programın başlangıç noktası ve akış yönetimi.
    """
    # Proje dizin yapısına tam uyumlu dinamik yol tanımı
    veri_dosyasi = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical-appointments-no-show-en.csv"
    )

    veri = veri_setini_yukle(veri_dosyasi)

    if veri is None:
        return

    # Yinelenen kayıt analiz fonksiyonunu koşturma
    yinelenen_kayit_tespiti(veri)


if __name__ == "__main__":
    main()