import pandas as pd
from pathlib import Path


def veri_setini_yukle(dosya_yolu: str) -> pd.DataFrame:
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


def eksik_veri_durum_tespiti(veri: pd.DataFrame):
    """
    ANALİZ ADIMI

    Veri setindeki her bir sütunun eksik değer sayısını ve yüzdesini hesaplar,
    büyükten küçüğe sıralayarak ampirik olarak raporlar.
    """
    print("=" * 90)
    print("ANALİZ: HAM VERİ EKSİK VERİ (MISSING VALUE) ANALİZİ")
    print("=" * 90)

    # Toplam satır sayısı üzerinden oran hesabı için referans alınır
    toplam_satir = len(veri)

    # Her sütun için eksik değer sayısı ve yüzdesi hesaplanır
    eksik_sayisi = veri.isnull().sum()
    eksik_yuzdesi = (veri.isnull().sum() / toplam_satir) * 100

    # Elde edilen sonuçlar yeni bir DataFrame üzerinde birleştirilir
    eksik_veri_tablosu = pd.DataFrame(
        {"Eksik Değer Sayısı": eksik_sayisi, "Eksik Değer Yüzdesi (%)": eksik_yuzdesi}
    )

    # Tablo sadece eksik değere sahip olan sütunlara göre değil, tüm sütunları gösterecek
    # ve eksik değer sayısına göre büyükten küçüğe sıralanacaktır
    sirali_tablo = eksik_veri_tablosu.sort_values(
        by="Eksik Değer Sayısı", ascending=False
    )

    # Raporlama formatında ekrana basılır
    print(sirali_tablo.to_string())
    print("=" * 90)


def main():
    """
    Programın başlangıç noktası ve akış yönetimi.
    """
    # Proje dizin yapısına ve iskeletinize tam uyumlu dinamik yol tanımı
    veri_dosyasi = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical-appointments-no-show-en.csv"
    )

    veri = veri_setini_yukle(veri_dosyasi)

    if veri is None:
        return

    # Eksik veri durum tespit fonksiyonunu koşturma
    eksik_veri_durum_tespiti(veri)


if __name__ == "__main__":
    main()