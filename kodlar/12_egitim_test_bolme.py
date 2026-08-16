import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path


def ara_veri_setini_yukle(dosya_yolu: Path) -> pd.DataFrame:
    try:
        veri = pd.read_csv(dosya_yolu)
        return veri
    except Exception as hata:
        print(f"\nAra veri seti yüklenirken hata oluştu:\n{hata}")
        return None


def tabakali_veri_seti_bolme(veri: pd.DataFrame, hedef_kolon: str) -> tuple:
    print("=" * 90)
    print("İŞLEM: TABAKALI ÖRNEKLEME İLE VERİ SETİNİN EĞİTİM VE TEST OLARAK BÖLÜNMESİ")
    print("=" * 90)

    if hedef_kolon not in veri.columns:
        print(f"Hata: Hedef değişken '{hedef_kolon}' veri setinde bulunamadı.")
        return None, None

    baslangic_satir, baslangic_sutun = veri.shape
    print(f"Girdi Veri Seti Boyutu: {baslangic_satir:,} satır, {baslangic_sutun} sütun")
    print("-" * 90)

    bolucu = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)

    for egitim_indeks, test_indeks in bolucu.split(veri, veri[hedef_kolon]):
        egitim_seti = veri.iloc[egitim_indeks].reset_index(drop=True)
        test_seti = veri.iloc[test_indeks].reset_index(drop=True)

    print("[KÜME BOYUTLARI VE ORANLARI]")
    print(f"Eğitim Seti (Train Set) : {len(egitim_seti):,} satır | Oran: %{(len(egitim_seti)/baslangic_satir)*100:.2f}")
    print(f"Test Seti (Test Set)     : {len(test_seti):,} satır | Oran: %{(len(test_seti)/baslangic_satir)*100:.2f}")
    print("-" * 90)

    print("\n[HEDEF DEĞİŞKEN (no_show) SINIF DAĞILIM KONTROLÜ]")
    print("." * 45)
    
    for isim, alt_kume in [("Eğitim Seti", egitim_seti), ("Test Seti", test_seti)]:
        frekans = alt_kume[hedef_kolon].value_counts(dropna=False)
        yuzde = alt_kume[hedef_kolon].value_counts(normalize=True, dropna=False) * 100
        dagilim_tablosu = pd.DataFrame({"Frekans": frekans, "Yüzde (%)": yuzde})
        print(f"\n{isim} Dağılımı:")
        print(dagilim_tablosu.to_string())
        print("." * 45)

    print("=" * 90)
    return egitim_seti, test_seti


def alt_kumeleri_kaydet(egitim_veri: pd.DataFrame, test_veri: pd.DataFrame, egitim_yolu: Path, test_yolu: Path):
    egitim_veri.to_csv(egitim_yolu, index=False, encoding="utf-8-sig")
    test_veri.to_csv(test_yolu, index=False, encoding="utf-8-sig")
    
    print("\nEğitim ve Test alt kümeleri başarıyla izole edilerek kaydedildi.")
    print(f"  Eğitim Seti: {egitim_yolu.name}")
    print(f"  Test Seti  : {test_yolu.name}")

    print("\nNot:")
    print("Tabakalı örnekleme (Stratified Sampling) kullanılarak hedef değişken dağılımı korunmuştur.")
    print("Bu aşamada herhangi bir ölçekleme, kodlama veya SMOTE işlemi uygulanmamıştır.")
    print("Bu işlemler yalnızca eğitim seti üzerinde sonraki adımlarda gerçekleştirilecektir.")


def main():
    girdi_yolu = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical_appointments_preprocessed_step02.csv"
    )

    veri = ara_veri_setini_yukle(girdi_yolu)

    if veri is None:
        return

    egitim_kumesi, test_kumesi = tabakali_veri_seti_bolme(veri, hedef_kolon="no_show")

    if egitim_kumesi is None or test_kumesi is None:
        return

    egitim_cikti_yolu = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical_appointments_train.csv"
    )
    test_cikti_yolu = (
        Path(__file__).resolve().parent.parent
        / "veriler"
        / "medical_appointments_test.csv"
    )

    alt_kumeleri_kaydet(egitim_kumesi, test_kumesi, egitim_cikti_yolu, test_cikti_yolu)


if __name__ == "__main__":
    main()