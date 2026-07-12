import pandas as pd
from pathlib import Path


def veri_setini_yukle(dosya_yolu: str) -> pd.DataFrame:
    """
    Veri setini hiçbir ön işleme, dönüşüm veya veri manipülasyonu
    uygulamadan ham haliyle yükler.

    Parametre
    ---------
    dosya_yolu : str
        CSV dosyasının yolu.

    Döndürür
    --------
    pandas.DataFrame
        Ham veri seti.
    """

    try:
        veri = pd.read_csv(dosya_yolu)
        return veri

    except Exception as hata:
        print(f"\nVeri seti yüklenirken hata oluştu:\n{hata}")
        return None


def veri_seti_genel_bakis(veri: pd.DataFrame):
    """
    ADIM 1

    Veri setinin genel yapısını raporlar.
    """

    print("=" * 90)
    print("ADIM 1 - VERİ SETİ GENEL BAKIŞI")
    print("=" * 90)

    satir_sayisi, sutun_sayisi = veri.shape

    bellek_kullanimi = (
        veri.memory_usage(deep=True).sum() / (1024 ** 2)
    )

    print(f"Toplam Satır Sayısı : {satir_sayisi:,}")
    print(f"Toplam Sütun Sayısı : {sutun_sayisi}")
    print(f"Bellek Kullanımı    : {bellek_kullanimi:.2f} MB")

    print("\n" + "-" * 90)
    print("İLK 5 GÖZLEM")
    print("-" * 90)
    print(veri.head())

    print("\n" + "-" * 90)
    print("SON 5 GÖZLEM")
    print("-" * 90)
    print(veri.tail())

    print("\n" + "-" * 90)
    print("RASTGELE 5 GÖZLEM (Seed = 42)")
    print("-" * 90)
    print(veri.sample(n=min(5, len(veri)), random_state=42))


def degisken_yapisini_incele(veri: pd.DataFrame):
    """
    ADIM 2

    Veri setinin yapısal şemasını raporlar.
    """

    print("\n")
    print("=" * 90)
    print("ADIM 2 - DEĞİŞKEN YAPISI")
    print("=" * 90)

    print("\nDataFrame İndeksi")
    print(veri.index)

    print("\nİndeks Tipi")
    print(type(veri.index))

    print("\nSütun İsimleri")
    print(veri.columns.tolist())

    print("\nVeri Tipleri")
    print(veri.dtypes)

    print("\nDataFrame Bilgisi")
    veri.info(memory_usage="deep")


def degisken_ozet_tablosu(veri: pd.DataFrame):
    """
    ADIM 3

    Her değişken için temel özet bilgileri oluşturur.
    """

    print("\n")
    print("=" * 90)
    print("ADIM 3 - DEĞİŞKEN ÖZET TABLOSU")
    print("=" * 90)

    ozet_tablo = pd.DataFrame(
        {
            "Değişken": veri.columns,
            "Veri Tipi": veri.dtypes.astype(str).values,
            "Eksik Değer": veri.isnull().sum().values,
            "Benzersiz Değer": veri.nunique(dropna=False).values,
            "Bellek (KB)": (
                veri.memory_usage(deep=True)
                .drop("Index")
                .values / 1024
            ),
        }
    )

    print(ozet_tablo.to_string(index=False))


def tanimlayici_istatistikler(veri: pd.DataFrame):
    """
    ADIM 4

    Ham veri üzerinde herhangi bir değişiklik yapmadan
    tanımlayıcı istatistikleri raporlar.
    """

    print("\n")
    print("=" * 90)
    print("ADIM 4 - TANIMLAYICI İSTATİSTİKLER")
    print("=" * 90)

    print(veri.describe(include="all").transpose())


def main():
    """
    Programın başlangıç noktası.
    """

    veri_dosyasi = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical-appointments-no-show-en.csv"
    )

    veri = veri_setini_yukle(veri_dosyasi)

    if veri is None:
        return

    veri_seti_genel_bakis(veri)

    degisken_yapisini_incele(veri)

    degisken_ozet_tablosu(veri)

    tanimlayici_istatistikler(veri)


if __name__ == "__main__":
    main()