import pandas as pd
import numpy as np
from pathlib import Path


def ham_veri_setini_yukle(dosya_yolu: str) -> pd.DataFrame:
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


def korelasyon_matrisi_hesapla(veri: pd.DataFrame):
    """
    ANALİZ ADIMI

    Yalnızca gerçekten sürekli sayısal olan değişkenleri seçer.
    Pearson korelasyon matrisini hesaplar ve terminalde raporlar.
    En yüksek doğrusal ilişkiye sahip değişken çiftlerini sıralı listeler.
    """
    print("=" * 90)
    print("ANALİZ: HAM SÜREKLİ SAYISAL DEĞİŞKENLER PEARSON KORELASYON ANALİZİ")
    print("=" * 90)

    # Protokol gereği yalnızca sürekli sayısal değişkenler analize dahil edilir.
    # İkili (binary) bayraklar ve kategorik alanlar dışarıda bırakılmıştır.
    surekli_degiskenler = [
        "age",
        "average_temp_day",
        "average_rain_day",
        "max_temp_day",
        "max_rain_day"
    ]

    # Mevcut olan sütunlar üzerinden veri alt kümesi oluşturulur
    mevcut_sutunlar = [sutun for sutun in surekli_degiskenler if sutun in veri.columns]
    
    if not mevcut_sutunlar:
        print("Analiz için uygun sürekli sayısal değişken bulunamadı.")
        return

    # Eksik değerler silinmez veya doldurulmaz, pandas corr() fonksiyonu 
    # ham veri üzerindeki mevcut çiftleri (pairwise) esas alarak hesaplama yapar.
    korelasyon_matrisi = veri[mevcut_sutunlar].corr(method="pearson")

    print("[PEARSON KORELASYON MATRİSİ]")
    print("-" * 90)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(korelasyon_matrisi.round(4))
    print("-" * 90)

    # Matrisin alt ve üst üçgen tekrarlarını ve köşegen (1.0) değerlerini ayıklama
    print("\n[EN YÜKSEK MUTLAK KORELASYONA SAHİP DEĞİŞKEN ÇİFTLERİ]")
    print("-" * 90)
    
    korelasyon_serisi = korelasyon_matrisi.abs().unstack()
    sirali_korelasyon = korelasyon_serisi.sort_values(ascending=False)
    
    # Kendisiyle olan korelasyonları (1.0) ve mükerrer çiftleri filtreleme
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
    print(ciftler_tablosu.to_string(index=False))
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

    # Korelasyon analiz fonksiyonunu çalıştırma
    korelasyon_matrisi_hesapla(ham_veri)


if __name__ == "__main__":
    main()