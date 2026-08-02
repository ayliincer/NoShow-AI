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


def korelasyon_matrisi_hesapla(veri: pd.DataFrame):
    print("=" * 90)
    print("ANALİZ: HAM SÜREKLİ SAYISAL DEĞİŞKENLER PEARSON KORELASYON ANALİZİ")
    print("=" * 90)

    surekli_degiskenler = [
        "age",
        "average_temp_day",
        "average_rain_day",
        "max_temp_day",
        "max_rain_day"
    ]

    mevcut_sutunlar = [sutun for sutun in surekli_degiskenler if sutun in veri.columns]
    print(f"Analize Dahil Edilen Sürekli Değişken Sayısı : {len(mevcut_sutunlar)}")
    print()
    
    if not mevcut_sutunlar:
        print("Analiz için uygun sürekli sayısal değişken bulunamadı.")
        return

    korelasyon_matrisi = veri[mevcut_sutunlar].corr(method="pearson")

    print("[PEARSON KORELASYON MATRİSİ]")
    print("-" * 90)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(korelasyon_matrisi.round(4))
    print("-" * 90)

    print("\n[EN YÜKSEK MUTLAK KORELASYONA SAHİP DEĞİŞKEN ÇİFTLERİ]")
    print("-" * 90)
    
    korelasyon_serisi = korelasyon_matrisi.abs().unstack()
    sirali_korelasyon = korelasyon_serisi.sort_values(ascending=False)
    ayrik_ciftler = []
    gorulen_ciftler = set()
    
    for indeks, katsayi in sirali_korelasyon.items():
        degisken_1, degisken_2 = indeks
        if degisken_1 != degisken_2 and (degisken_2, degisken_1) not in gorulen_ciftler:
            gorulen_ciftler.add((degisken_1, degisken_2))
            # Orijinal (yönlü) katsayıyı matristen geri çekme
            orijinal_katsayi = korelasyon_matrisi.loc[degisken_1, degisken_2]
            ayrik_ciftler.append({
                "Değişken 1": degisken_1,
                "Değişken 2": degisken_2,
                "Pearson Katsayısı": orijinal_katsayi,
                "Mutlak Değer": katsayi
            })
            
            
    ciftler_tablosu = pd.DataFrame(ayrik_ciftler)

    guclu_korelasyon_sayisi = (
        ciftler_tablosu["Mutlak Değer"] >= 0.70
    ).sum()

    print(f"Güçlü Korelasyon Çifti Sayısı (|r| ≥ 0.70): {guclu_korelasyon_sayisi}")
    print()

    print(ciftler_tablosu.to_string(index=False))

    print("\nNot:")
    print("Bu analiz yalnızca yüksek doğrusal ilişkiye sahip değişken çiftlerini belirlemek amacıyla gerçekleştirilmiştir.")
    print("Bu aşamada herhangi bir değişken veri setinden çıkarılmamıştır.")
    print("Değişken çıkarma kararı modelleme aşamasındaki performans değerlendirmelerine göre verilecektir.")

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

    korelasyon_matrisi_hesapla(ham_veri)


if __name__ == "__main__":
    main()