import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path


def veri_setlerini_yukle(egitim_yolu: Path, test_yolu: Path) -> tuple:
    try:
        egitim = pd.read_csv(egitim_yolu)
        test = pd.read_csv(test_yolu)
        return egitim, test
    except Exception as hata:
        print(f"\nVeri setleri yüklenirken hata oluştu:\n{hata}")
        return None, None


def kardinalite_profilini_raporla(veri: pd.DataFrame, haric_tutulacaklar: list) -> list:
    print("=" * 110)
    print("ANALİZ: KATEGORİK ÖZNİTELİKLERİN KARDİNALİTE VE ENCODING ADAYLIK PROFİLLEMESİ")
    print("=" * 110)


    kategorik_adaylar = []
    
    kategorik_sutunlar = veri.select_dtypes(include=["object"]).columns

    for sutun in kategorik_sutunlar:
        if sutun in haric_tutulacaklar:
            continue
            
        benzersiz_sayisi = veri[sutun].nunique()
        eksik_sayisi = veri[sutun].isnull().sum()
        veri_tipi = str(veri[sutun].dtype)
        
        # SCI Hakem Kriteri: Kardinalite eşik değeri 15 olarak belirlenmiştir.
        if benzersiz_sayisi > 15:
            adaylik_durumu = "HAYIR (Yüksek Kardinalite Uyarısı)"
            strateji = "Target / Frequency Encoding Adayıdır"
        else:
            adaylik_durumu = "EVET"
            strateji = "One-Hot Encoding Adayıdır"

        kategorik_adaylar.append({
            "Sütun Adı": sutun,
            "Veri Tipi": veri_tipi,
            "Benzersiz Sınıf": benzersiz_sayisi,
            "Eksik Değer": eksik_sayisi,
            "Encoding Adayı": adaylik_durumu,
            "Önerilen Metodoloji": strateji
        })

    rapor_tablosu = pd.DataFrame(kategorik_adaylar)
    print(rapor_tablosu.to_string(index=False))
    print("=" * 110)
    
    return kategorik_adaylar


def kategorik_kodlama_uygula(egitim: pd.DataFrame, test: pd.DataFrame, profil_listesi: list, ham_zaman_sutunlari: list) -> tuple:
    print("\n" + "-" * 90)
    print("İŞLEM 1: HAM ZAMAN SÜTUNLARININ BİLİMSEL GEREKÇEYLE KALDIRILMASI")
    print("-" * 90)
    for sutun in ham_zaman_sutunlari:
        for veri_kumesi, isim in [(egitim, "Eğitim Seti"), (test, "Test Seti")]:
            if sutun in veri_kumesi.columns:
                veri_kumesi.drop(columns=[sutun], inplace=True)
        print(f"Bilgi: Saatsel/Takvimsel örüntüleri türetildiği için ham '{sutun}' özniteliği matris dışı bırakılmıştır.")

    one_hot_sutunlari = [o["Sütun Adı"] for o in profil_listesi if o["Encoding Adayı"] == "EVET" and o["Sütun Adı"] in egitim.columns]
    print(f"\nOne-Hot Encoding uygulanan sütun sayısı : {len(one_hot_sutunlari)}")
    print(f"One-Hot Encoding uygulanan sütunlar : {one_hot_sutunlari}")
    yuksek_kardinalite_sutunlari = [o["Sütun Adı"] for o in profil_listesi if o["Encoding Adayı"] != "EVET" and o["Sütun Adı"] in egitim.columns]

    if yuksek_kardinalite_sutunlari:
        print("\n" + "!" * 90)
        print("METODOLOJİK UYARI: YÜKSEK KARDİNALİTELİ DEĞİŞKEN TESPİTİ")
        print("!" * 90)
        for sutun in yuksek_kardinalite_sutunlari:
            print(f"-> '{sutun}' değişkeni yüksek sınıf çeşitliliği nedeniyle One-Hot Encoding kapsamı dışına alınmıştır.")
            print(f"   Bu değişken ileriki Feature Selection veya model mimarisine göre Target/Frequency Encoding ile değerlendirilecektir.")
        print("!" * 90)

    if not one_hot_sutunlari:
        print("\nOne-Hot Encoding uygulanacak uygun kategorik değişken bulunamadı.")
        return egitim, test

    print("\n" + "-" * 90)
    print("İŞLEM 2: İZOLASYON UYUMLU ONE-HOT ENCODING UYGULAMASI")
    print("-" * 90)

    try:
        kodlayici = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        kodlayici = OneHotEncoder(sparse=False, handle_unknown="ignore")

    kodlayici.fit(egitim[one_hot_sutunlari])

    yeni_sutun_isimleri = kodlayici.get_feature_names_out(one_hot_sutunlari)

    egitim_donusum = pd.DataFrame(kodlayici.transform(egitim[one_hot_sutunlari]), columns=yeni_sutun_isimleri, index=egitim.index)
    test_donusum = pd.DataFrame(kodlayici.transform(test[one_hot_sutunlari]), columns=yeni_sutun_isimleri, index=test.index)
    egitim_nihai = pd.concat([egitim.drop(columns=one_hot_sutunlari), egitim_donusum], axis=1)
    test_nihai = pd.concat([test.drop(columns=one_hot_sutunlari), test_donusum], axis=1)

    egitim_nihai, test_nihai = egitim_nihai.align(test_nihai, join="left", axis=1, fill_value=0)

    print("\n" + "=" * 90)
    print("NİHAİ MATRİS BOYUT VE SÜTUN YAPISI DOĞRULAMASI")
    print("=" * 90)
    print(f"Eğitim Seti Son Boyut : {egitim_nihai.shape[0]:,} satır | {egitim_nihai.shape[1]} sütun")
    print(f"Test Seti Son Boyut   : {test_nihai.shape[0]:,} satır | {test_nihai.shape[1]} sütun")
    
    sutun_uyumu = list(egitim_nihai.columns) == list(test_nihai.columns)

    print(f"Eğitim ve Test Sütun Yapıları Birebir Eşit mi? : {sutun_uyumu}")
    print(f"Oluşturulan Dummy Sütun Sayısı : {len(yeni_sutun_isimleri)}")
    print("=" * 90)

    return egitim_nihai, test_nihai


def verileri_diske_kaydet(egitim: pd.DataFrame, test: pd.DataFrame, egitim_yolu: Path, test_yolu: Path):
    egitim.to_csv(egitim_yolu, index=False, encoding="utf-8-sig")
    test.to_csv(test_yolu, index=False, encoding="utf-8-sig")
    print(f"\nKodlama işlemleri tescil edilen veri setleri başarıyla güncellendi:")
    print(f"  Eğitim Seti Çıktısı : {egitim_yolu.name}")
    print(f"  Test Seti Çıktısı   : {test_yolu.name}")


def main():
    egitim_yolu = Path(__file__).resolve().parent.parent / "veriler" / "medical_appointments_train.csv"
    test_yolu = Path(__file__).resolve().parent.parent / "veriler" / "medical_appointments_test.csv"

    egitim_veri, test_veri = veri_setlerini_yukle(egitim_yolu, test_yolu)

    if egitim_veri is None or test_veri is None:
        return

    haric_tutulacaklar = ["no_show"]
    ham_zaman_sutunlari = ["appointment_date", "entry_service_date", "date_of_birth"]

    profil_listesi = kardinalite_profilini_raporla(egitim_veri, haric_tutulacaklar)

    yeni_egitim, yeni_test = kategorik_kodlama_uygula(egitim_veri, test_veri, profil_listesi, ham_zaman_sutunlari)

    verileri_diske_kaydet(yeni_egitim, yeni_test, egitim_yolu, test_yolu)


if __name__ == "__main__":
    main()