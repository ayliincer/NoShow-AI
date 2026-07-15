import pandas as pd
import numpy as np
from pathlib import Path


def ham_veri_setini_yukle(dosya_yolu: Path) -> pd.DataFrame:
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


def aykiri_deger_sinirlarini_hesapla(veri: pd.DataFrame):
    """
    ANALİZ ADIMI

    Sürekli sayısal öznitelikleri otomatik olarak tespit eder.
    Her biri için Q1, Q3, IQR, alt sınır ve üst sınır değerlerini hesaplar.
    Bu sınırların dışında kalan gözlem sayılarını belirleyerek raporlar.
    """
    print("=" * 90)
    print("ANALİZ: SÜREKLİ SAYISAL DEĞİŞKENLER AYKIRI DEĞER SINIR ANALİZİ (IQR YÖNTEMİ)")
    print("=" * 90)

    # İkili (binary) kodlanmış veya kategorik tabanlı sayısal göstergeler
    # sürekli sayısal değişken olmadıkları için bu spesifik analizin dışında tutulur.
    surekli_sayisal_degiskenler = [
        "appointment_year",
        "age",
        "average_temp_day",
        "average_rain_day",
        "max_temp_day",
        "max_rain_day"
    ]

    print(f"İncelenen Sürekli Sayısal Değişken Sayısı : {len(surekli_sayisal_degiskenler)}")
    print()

    aykiri_deger_sonuclari = []

    for degisken in surekli_sayisal_degiskenler:
        if degisken in veri.columns:
            # Eksik değerler (NaN) hesaplamayı etkilememesi için alt seride dropna yapılır, 
            # fakat ana veri setinden hiçbir satır silinmez.
            seri = veri[degisken].dropna()
            
            toplam_gozlem = len(seri)
            birinci_ceyrek = seri.quantile(0.25)
            ucuncu_ceyrek = seri.quantile(0.75)
            ceyrekler_arasi_mesafe = ucuncu_ceyrek - birinci_ceyrek
            
            # IQR formülüne göre alt ve üst sınırlar tanımlanır
            alt_sinir = birinci_ceyrek - (1.5 * ceyrekler_arasi_mesafe)
            ust_sinir = ucuncu_ceyrek + (1.5 * ceyrekler_arasi_mesafe)
            
            # Sınırların dışında kalan gözlem sayıları hesaplanır
            alt_sinir_alti_sayisi = (seri < alt_sinir).sum()
            ust_sinir_usti_sayisi = (seri > ust_sinir).sum()
            toplam_aykiri_sayisi = alt_sinir_alti_sayisi + ust_sinir_usti_sayisi
            aykiri_oran_yuzde = (toplam_aykiri_sayisi / toplam_gozlem) * 100 if toplam_gozlem > 0 else 0
            
            aykiri_deger_sonuclari.append({
                "Değişken": degisken,
                "Gözlem Sayısı": toplam_gozlem,
                "Q1 (%25)": birinci_ceyrek,
                "Q3 (%75)": ucuncu_ceyrek,
                "IQR": ceyrekler_arasi_mesafe,
                "Alt Sınır": alt_sinir,
                "Üst Sınır": ust_sinir,
                "Alt Sınır Altı": alt_sinir_alti_sayisi,
                "Üst Sınır Üstü": ust_sinir_usti_sayisi,
                "Toplam Aykırı": toplam_aykiri_sayisi,
                "Aykırı Oranı (%)": aykiri_oran_yuzde
            })

    # İstatistiksel özet tablo formatına dönüştürme
    rapor_tablosu = pd.DataFrame(aykiri_deger_sonuclari)
    
    # Tüm sütunların terminal ekranında tam hizalı gösterilmesi sağlanır
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(rapor_tablosu.to_string(index=False))

    print("\nNot:")
    print("IQR yöntemine göre aykırı değerler yalnızca raporlanmıştır.")
    print("Bu aşamada veri setinden herhangi bir gözlem silinmemiştir.")

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

    # Aykırı değer analiz fonksiyonunu çalıştırma
    aykiri_deger_sinirlarini_hesapla(ham_veri)


if __name__ == "__main__":
    main()