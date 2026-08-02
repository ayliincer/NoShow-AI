import pandas as pd
import numpy as np
from pathlib import Path


def ham_veri_setini_yukle(dosya_yolu: Path) -> pd.DataFrame:
    try:
        ham_veri = pd.read_csv(dosya_yolu)
        return ham_veri
    except Exception as hata:
        print(f"\nVeri seti yüklenirken bir hata oluştu:\n{hata}")
        return None


def aykiri_deger_sinirlarini_hesapla(veri: pd.DataFrame):
    print("=" * 90)
    print("ANALİZ: SÜREKLİ SAYISAL DEĞİŞKENLER AYKIRI DEĞER SINIR ANALİZİ (IQR YÖNTEMİ)")
    print("=" * 90)

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
            seri = veri[degisken].dropna()
            
            toplam_gozlem = len(seri)
            birinci_ceyrek = seri.quantile(0.25)
            ucuncu_ceyrek = seri.quantile(0.75)
            ceyrekler_arasi_mesafe = ucuncu_ceyrek - birinci_ceyrek
            alt_sinir = birinci_ceyrek - (1.5 * ceyrekler_arasi_mesafe)
            ust_sinir = ucuncu_ceyrek + (1.5 * ceyrekler_arasi_mesafe)
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

    rapor_tablosu = pd.DataFrame(aykiri_deger_sonuclari)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(rapor_tablosu.to_string(index=False))

    print("\nNot:")
    print("IQR yöntemine göre aykırı değerler yalnızca raporlanmıştır.")
    print("Bu aşamada veri setinden herhangi bir gözlem silinmemiştir.")

    print("=" * 90)


def main():
    veri_yolu = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical-appointments-no-show-en.csv"
    )

    ham_veri = ham_veri_setini_yukle(veri_yolu)

    if ham_veri is None:
        return

    aykiri_deger_sinirlarini_hesapla(ham_veri)


if __name__ == "__main__":
    main()