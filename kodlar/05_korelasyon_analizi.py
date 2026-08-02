import pandas as pd
import numpy as np
from pathlib import Path


def veri_setini_yukle(dosya_yolu: Path) -> pd.DataFrame:
    try:
        veri = pd.read_csv(dosya_yolu)
        return veri
    except Exception as hata:
        print(f"\nVeri seti yüklenirken hata oluştu:\n{hata}")
        return None


def sayisal_degisken_istatistikleri(veri: pd.DataFrame):
    print("=" * 90)
    print("ANALİZ: HAM SAYISAL DEĞİŞKEN DAĞILIM VE ŞEKİL PARAMETRELERİ")
    print("=" * 90)

    # Veri setindeki sayısal (sürekli ve kesikli) değişkenler izole edilir
    sayisal_sutunlar = veri.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Toplam Sayısal Değişken Sayısı : {len(sayisal_sutunlar)}")
    print()
    
    ozet_veriler = []

    for sutun in sayisal_sutunlar:
        seri = veri[sutun]

        gozlem_sayisi = seri.count()
        eksik_sayisi = seri.isnull().sum()
        ortalama = seri.mean()
        medyan = seri.median()
        standart_sapma = seri.std()

        if ortalama != 0:
            degisim_katsayisi = standart_sapma / ortalama
        else:
            degisim_katsayisi = np.nan
        minimum_deger = seri.min()

        q1 = seri.quantile(0.25)
        q3 = seri.quantile(0.75)
        iqr = q3 - q1
        maksimum_deger = seri.max()
        carpiklik = seri.skew()
        basiklik = seri.kurtosis()
        

        sutun_ozeti = {
            "Değişken": sutun,
            "Gözlem Sayısı": gozlem_sayisi,
            "Eksik Değer": eksik_sayisi,
            "Ortalama": ortalama,
            "Medyan": medyan,
            "Std Sapma": standart_sapma,
            "CV": degisim_katsayisi,
            "Minimum": minimum_deger,
            "Q1 (%25)": q1,
            "Q3 (%75)": q3,
            "IQR": iqr,
            "Maksimum": maksimum_deger,
            "Skewness": carpiklik,
            "Kurtosis": basiklik
        }
        ozet_veriler.append(sutun_ozeti)

    sayisal_ozet_tablo = pd.DataFrame(ozet_veriler)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(sayisal_ozet_tablo.to_string(index=False))
    print("=" * 90)


def main():
    veri_dosyasi = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical-appointments-no-show-en.csv"
    )

    veri = veri_setini_yukle(veri_dosyasi)

    if veri is None:
        return

    sayisal_degisken_istatistikleri(veri)


if __name__ == "__main__":
    main()