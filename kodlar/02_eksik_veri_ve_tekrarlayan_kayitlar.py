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


def hedef_degisken_dagilimi(veri: pd.DataFrame):
    """
    Hedef değişkenin (no_show) ham frekans ve yüzde dağılımını raporlar.
    """
    print("=" * 90)
    print("ANALİZ: HEDEF DEĞİŞKEN (no_show) DAĞILIMI")
    print("=" * 90)
    print(f"Toplam Gözlem Sayısı : {len(veri):,}")
    print()

    frekans = veri["no_show"].value_counts(dropna=False)
    yuzde = veri["no_show"].value_counts(normalize=True, dropna=False) * 100

    dagilim_tablosu = pd.DataFrame({"Frekans": frekans, "Yüzde (%)": yuzde})

    print(dagilim_tablosu.to_string())
    print("-" * 90)


def kategorik_degisken_dagilimlari(veri: pd.DataFrame):
    """
    Veri setindeki ana kategorik değişkenlerin benzersiz sınıf listelerini
    	ve frekans yoğunluklarını ampirik olarak raporlar.
    """
    print("\n")
    print("=" * 90)
    print("ANALİZ: KATEGORİK VE SÖZEL DEĞİŞKENLERİN DAĞILIMI")
    print("=" * 90)

    # Analiz edilecek ana kategorik sütunlar listesi
    kategorik_sutunlar = [
        "specialty",
        "gender",
        "appointment_shift",
        "rain_intensity",
        "heat_intensity",
        "city",
    ]

    for sutun in kategorik_sutunlar:
        if sutun in veri.columns:
            print(f"\n[Değişken: {sutun}]")
            print(f"Benzersiz Sınıf Sayısı : {veri[sutun].nunique(dropna=False)}")
            print("-" * 45)

            frekans = veri[sutun].value_counts(dropna=False)
            yuzde = veri[sutun].value_counts(normalize=True, dropna=False) * 100

            sutun_tablo = pd.DataFrame(
                {"Frekans": frekans, "Yüzde (%)": yuzde}
            )
            print(sutun_tablo.to_string())
            print("." * 45)
        else:
            print(f"\n[Uyarı: {sutun} sütunu veri setinde bulunamadı.]")

    print("=" * 90)


def main():
    """
    Programın başlangıç noktası ve akış yönetimi.
    """
    # Proje dizin yapısına uygun dinamik yol tanımı
    veri_dosyasi = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical-appointments-no-show-en.csv"
    )

    veri = veri_setini_yukle(veri_dosyasi)

    if veri is None:
        return

    # Sınıf dengesi ve frekans analizlerini çalıştırma
    hedef_degisken_dagilimi(veri)
    kategorik_degisken_dagilimlari(veri)


if __name__ == "__main__":
    main()