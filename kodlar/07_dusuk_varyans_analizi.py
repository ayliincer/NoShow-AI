import pandas as pd
from pathlib import Path


def ham_veri_setini_yukle(dosya_yolu: Path) -> pd.DataFrame:
    try:
        ham_veri = pd.read_csv(dosya_yolu)
        return ham_veri
    except Exception as hata:
        print(f"\nVeri seti yüklenirken bir hata oluştu:\n{hata}")
        return None


def dusuk_varyans_durum_tespiti(veri: pd.DataFrame):
    print("=" * 90)
    print("ANALİZ: HAM VERİ SABİT VE DÜŞÜK VARYANSLI DEĞİŞKEN ANALİZİ")
    print("=" * 90)
    print(f"Toplam Değişken Sayısı : {veri.shape[1]}")
    print()

    toplam_satir = len(veri)
    varyans_ozet_verileri = []

    for sutun in veri.columns:
        seri = veri[sutun]
        benzersiz_sayisi = seri.nunique(dropna=False)
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

    varyans_ozet_tablo = pd.DataFrame(varyans_ozet_verileri)

    sirali_tablo = varyans_ozet_tablo.sort_values(
        by="En Sık Değer Oranı (%)", ascending=False
    )
    
    dusuk_varyans_sayisi = (
    sirali_tablo["En Sık Değer Oranı (%)"] >= 95
    ).sum()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(f"Düşük Varyans Adayı Değişken Sayısı (≥ %95): {dusuk_varyans_sayisi}")
    print()

    print(sirali_tablo.to_string(index=False))

    print("\nNot:")
    print("Bu analiz yalnızca düşük varyanslı değişken adaylarını belirlemek amacıyla gerçekleştirilmiştir.")
    print("Bu aşamada herhangi bir değişken veri setinden çıkarılmamıştır.")
    print("Değişken çıkarma kararı, modelleme öncesindeki özellik seçimi aşamasında verilecektir.")

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

    dusuk_varyans_durum_tespiti(ham_veri)


if __name__ == "__main__":
    main()