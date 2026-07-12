import pandas as pd
import numpy as np
from pathlib import Path


def veri_setini_yukle(dosya_yolu: str) -> pd.DataFrame:
    """
    Veri setini hiçbir ön işleme, dönüşüm veya veri manipülasyonu
    uygulamadan ham haliyle yükler.
    """
    try:
        veri = pd.read_csv(dosya_yolu)
        return veri
    except Exception as hata:
        print(f"\nVeri seti yüklenirken hata oluştu:\n{hata}")
        return None


def sayisal_degisken_istatistikleri(veri: pd.DataFrame):
    """
    ANALİZ ADIMI

    Veri setindeki tüm sayısal değişkenleri tespit eder ve her biri için
    gözlem, eksiklik, merkezi eğilim, çeyreklikler, IQR, çarpıklık ve basıklık
    değerlerini içeren kapsamlı bir özet tablo raporlar.
    """
    print("=" * 90)
    print("ANALİZ: HAM SAYISAL DEĞİŞKEN DAĞILIM VE ŞEKİL PARAMETRELERİ")
    print("=" * 90)

    # Veri setindeki sayısal (sürekli ve kesikli) değişkenler izole edilir
    sayisal_sutunlar = veri.select_dtypes(include=[np.number]).columns.tolist()

    # İlerleyen süreçte ID veya ikili (binary) kodlanmış bayrakları bu aşamada 
    # matematiksel olarak dışarıda bırakmak adına ham listeden filtreleme yapılabilir.
    # Ancak ham veri durum tespiti ilkesi gereği tüm sayısal alanlar listelenir.
    
    ozet_veriler = []

    for sutun in sayisal_sutunlar:
        # İlgili sütunun ham serisi alınır
        seri = veri[sutun]
        
        # Temel tanımlayıcı istatistikler hesaplanır
        gozlem_sayisi = seri.count() # Eksik olmayan satır sayısı
        eksik_sayisi = seri.isnull().sum()
        ortalama = seri.mean()
        medyan = seri.median()
        standart_sapma = seri.std()
        minimum_deger = seri.min()
        
        # Çeyreklikler hesaplanır
        q1 = seri.quantile(0.25)
        q3 = seri.quantile(0.75)
        iqr = q3 - q1
        maksimum_deger = seri.max()
        
        # Dağılımın şekil parametreleri (Çarpıklık ve Basıklık) hesaplanır
        carpiklik = seri.skew()
        basiklik = seri.kurtosis()

        # Sonuçlar sözlük yapısında toplanır
        sutun_ozeti = {
            "Değişken": sutun,
            "Gözlem Sayısı": gozlem_sayisi,
            "Eksik Değer": eksik_sayisi,
            "Ortalama": ortalama,
            "Medyan": medyan,
            "Std Sapma": standart_sapma,
            "Minimum": minimum_deger,
            "Q1 (%25)": q1,
            "Q3 (%75)": q3,
            "IQR": iqr,
            "Maksimum": maksimum_deger,
            "Skewness": carpiklik,
            "Kurtosis": basiklik
        }
        ozet_veriler.append(sutun_ozeti)

    # Elde edilen özet veriler pandas DataFrame yapısına dönüştürülür
    sayisal_ozet_tablo = pd.DataFrame(ozet_veriler)

    # Bilimsel raporlama formatında tüm sütunların görünmesi sağlanarak ekrana basılır
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(sayisal_ozet_tablo.to_string(index=False))
    print("=" * 90)


def main():
    """
    Programın başlangıç noktası ve akış yönetimi.
    """
    # Proje dizin yapısına tam uyumlu dinamik yol tanımı
    veri_dosyasi = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical-appointments-no-show-en.csv"
    )

    veri = veri_setini_yukle(veri_dosyasi)

    if veri is None:
        return

    # Sayısal değişken dağılım analiz fonksiyonunu koşturma
    sayisal_degisken_istatistikleri(veri)


if __name__ == "__main__":
    main()