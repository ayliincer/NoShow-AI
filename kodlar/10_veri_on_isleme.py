import pandas as pd
from pathlib import Path


def ham_veri_setini_yukle(dosya_yolu: Path) -> pd.DataFrame:
    """
    Belirtilen dizindeki ham CSV dosyasını yükler.
    Veri üzerinde hiçbir değişiklik yapılmaz.
    """
    try:
        return pd.read_csv(dosya_yolu)
    except Exception as hata:
        print(f"\nVeri seti yüklenirken hata oluştu:\n{hata}")
        return None


def veri_sizintisi_degiskenini_cikar(
    veri: pd.DataFrame,
    hedef_sutun: str
) -> pd.DataFrame:
    """
    Veri sızıntısı riski taşıyan değişkeni veri setinden çıkarır.
    """

    print("\n" + "=" * 90)
    print("İŞLEM 1 : VERİ SIZINTISI (DATA LEAKAGE) KONTROLÜ")
    print("=" * 90)

    if hedef_sutun in veri.columns:

        print(f"'{hedef_sutun}' değişkeni veri setinde bulundu.")

        veri = veri.drop(columns=[hedef_sutun])

        print(
            f"EDA sürecinde yüksek veri sızıntısı riski taşıdığı "
            f"değerlendirilen '{hedef_sutun}' değişkeni veri setinden çıkarıldı."
        )

    else:

        print(f"'{hedef_sutun}' veri setinde bulunamadı. İşlem uygulanmadı.")

    return veri


def tam_mukerrer_kayitlari_temizle(
    veri: pd.DataFrame,
    ham_duplicate_sayisi: int
) -> pd.DataFrame:
    """
    Tam mükerrer kayıtları temizler ve akademik rapor üretir.
    """

    print("\n" + "=" * 90)
    print("İŞLEM 2 : TAM MÜKERRER (DUPLICATE) KAYIT TEMİZLİĞİ")
    print("=" * 90)

    baslangic_satir = len(veri)

    duplicate_sayisi = veri.duplicated().sum()

    print(f"Ham veri setindeki duplicate sayısı                : {ham_duplicate_sayisi:,}")
    print(f"'no_show_reason' kaldırıldıktan sonraki duplicate  : {duplicate_sayisi:,}")
    print(f"Fark                                                : {duplicate_sayisi-ham_duplicate_sayisi:+}")

    print("-" * 90)

    temiz_veri = (
        veri
        .drop_duplicates(keep="first")
        .reset_index(drop=True)
    )

    print("İndeks yapısı yeniden oluşturuldu.")

    kalan_satir = len(temiz_veri)
    korunan_oran = (kalan_satir / baslangic_satir) * 100

    print(f"\nBaşlangıç Gözlem Sayısı      : {baslangic_satir:,}")
    print(f"Silinen Duplicate Sayısı     : {duplicate_sayisi:,}")
    print(f"Kalan Gözlem Sayısı          : {kalan_satir:,}")
    print(f"Korunan Gözlem Oranı (%)     : {korunan_oran:.2f}")

    return temiz_veri


def nihai_veri_raporu(veri: pd.DataFrame):
    """
    Nihai veri yapısını raporlar.
    """

    print("\n" + "=" * 90)
    print("NİHAİ VERİ YAPISI")
    print("=" * 90)

    print(f"Satır Sayısı  : {veri.shape[0]:,}")
    print(f"Sütun Sayısı  : {veri.shape[1]}")

    print("\nKalan sütunlar:\n")

    for sutun in veri.columns:
        print(f"- {sutun}")

    print("=" * 90)


def temiz_veriyi_kaydet(
    veri: pd.DataFrame,
    dosya_yolu: Path
):
    """
    Temizlenmiş veri setini bir sonraki preprocessing
    adımlarında kullanılmak üzere kaydeder.
    """

    veri.to_csv(
        dosya_yolu,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTemizlenmiş veri seti başarıyla kaydedildi.")
    print(f"Dosya : {dosya_yolu}")
    print("\nNot:")
    print("Bu aşamada yalnızca veri sızıntısı riski taşıyan değişken kaldırılmış ve tam mükerrer kayıtlar temizlenmiştir.")
    print("Eksik veri yönetimi, veri dönüşümleri, kodlama ve özellik mühendisliği işlemleri sonraki adımlarda gerçekleştirilecektir.")


def main():

    veri_yolu = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical-appointments-no-show-en.csv"
    )

    ham_veri = ham_veri_setini_yukle(veri_yolu)

    if ham_veri is None:
        return

    print("=" * 90)
    print("HAM VERİ DURUMU")
    print("=" * 90)

    ham_duplicate = ham_veri.duplicated().sum()

    print(f"Ham veri setindeki duplicate sayısı : {ham_duplicate:,}")

    temiz_veri = veri_sizintisi_degiskenini_cikar(
        ham_veri,
        "no_show_reason"
    )

    temiz_veri = tam_mukerrer_kayitlari_temizle(
        temiz_veri,
        ham_duplicate
    )

    nihai_veri_raporu(temiz_veri)

    kayit_yolu = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical_appointments_preprocessed_step01.csv"
    )

    temiz_veriyi_kaydet(
        temiz_veri,
        kayit_yolu
    )


if __name__ == "__main__":
    main()