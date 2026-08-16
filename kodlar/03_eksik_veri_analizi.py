import pandas as pd
from pathlib import Path


def veri_setini_yukle(dosya_yolu: Path) -> pd.DataFrame:
    try:
        veri = pd.read_csv(dosya_yolu)
        return veri
    except Exception as hata:
        print(f"\nVeri seti yüklenirken hata oluştu:\n{hata}")
        return None


def eksik_veri_durum_tespiti(veri: pd.DataFrame):
    print("=" * 90)
    print("ANALİZ: HAM VERİ EKSİK VERİ (MISSING VALUE) ANALİZİ")
    print("=" * 90)
    print(f"Toplam Gözlem Sayısı : {len(veri):,}")
    print(f"Toplam Değişken Sayısı : {veri.shape[1]}")
    print()

    toplam_satir = len(veri)
    eksik_sayisi = veri.isnull().sum()
    eksik_yuzdesi = (veri.isnull().sum() / toplam_satir) * 100

    eksik_veri_tablosu = pd.DataFrame(
        {"Eksik Değer Sayısı": eksik_sayisi, "Eksik Değer Yüzdesi (%)": eksik_yuzdesi}
    )

    sirali_tablo = eksik_veri_tablosu.sort_values(
        by="Eksik Değer Sayısı", ascending=False
    )
    eksik_sutun_sayisi = (eksik_sayisi > 0).sum()

    print(sirali_tablo.to_string())
    print("=" * 90)
    print(f"Eksik Değere Sahip Değişken Sayısı : {eksik_sutun_sayisi}")
    print()


def main():
    veri_dosyasi = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical-appointments-no-show-en.csv"
    )

    veri = veri_setini_yukle(veri_dosyasi)

    if veri is None:
        return
    
    eksik_veri_durum_tespiti(veri)


if __name__ == "__main__":
    main()