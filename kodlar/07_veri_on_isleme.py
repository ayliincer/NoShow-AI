import pandas as pd
from pathlib import Path


def ham_veri_setini_yukle(dosya_yolu: str) -> pd.DataFrame:
    """
    Belirtilen dizindeki CSV dosyasını, üzerinde hiçbir dönüştürme 
    veya filtreleme işlemi yapmadan ham haliyle yükler.

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
        ham_veri = pd.read_csv(dosya_yolu)
        return ham_veri
    except Exception as hata:
        print(f"\nVeri seti yüklenirken bir hata oluştu:\n{hata}")
        return None


def dusuk_varyans_durum_tespiti(veri: pd.DataFrame):
    """
    ANALİZ ADIMI

    Veri setindeki tüm değişkenler için benzersiz değer sayısını,
    en sık görülen değerin frekansını ve bu değerin yüzde oranını hesaplar.
    Sonuçları en yüksek yüzde oranına göre sıralayarak raporlar.
    """
    print("=" * 90)
    print("ANALİZ: HAM VERİ SABİT VE DÜŞÜK VARYANSLI DEĞİŞKEN ANALİZİ")
    print("=" * 90)

    toplam_satir = len(veri)
    varyans_ozet_verileri = []

    for sutun in veri.columns:
        # İlgili sütunun ham serisi alınır
        seri = veri[sutun]
        
        # Benzersiz değer sayısı hesaplanır (eksik değerler dahil edilerek tam çeşitlilik ölçülür)
        benzersiz_sayisi = seri.nunique(dropna=False)
        
        # En sık görülen değerin tespiti için frekans tablosu çıkarılır
        frekans_serisi = seri.value_counts(dropna=False)
        
        if not frekans_serisi.empty:
            en_sik_deger = frekans_serisi.index[0]
            en_sik_frekans = frekans_serisi.iloc[0]
            en_sik_oran_yuzde = (en_sik_frekans / toplam_satir) * 100
        else:
            en_sik_deger = None
            en_sik_frekans = 0
            en_sik_oran_yuzde = 0.0

        varyans_ozet_verileri.append({
            "Değişken": sutun,
            "Benzersiz Değer Sayısı": benzersiz_sayisi,
            "En Sık Değer": str(en_sik_deger),
            "En Sık Değer Frekansı": en_sik_frekans,
            "En Sık Değer Oranı (%)": en_sik_oran_yuzde
        })

    # Elde edilen özet veriler pandas DataFrame yapısına dönüştürülür
    varyans_ozet_tablo = pd.DataFrame(varyans_ozet_verileri)

    # En yüksek yüzde oranına göre büyükten küçüğe sıralanır
    sirali_tablo = varyans_ozet_tablo.sort_values(
        by="En Sık Değer Oranı (%)", ascending=False
    )

    # Tüm sütunların terminal ekranında tam hizalı gösterilmesi sağlanır
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(sirali_tablo.to_string(index=False))
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

    # Sabit ve düşük varyans analiz fonksiyonunu çalıştırma
    dusuk_varyans_durum_tespiti(ham_veri)


if __name__ == "__main__":
    main()