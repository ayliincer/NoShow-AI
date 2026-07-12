import pandas as pd
import numpy as np
from pathlib import Path


def veri_setlerini_yukle(egitim_yolu: Path, test_yolu: Path) -> tuple:
    """
    Eğitim ve test veri setlerini diske kayıtlı CSV dosyalarından yükler.

    Parametreler
    ------------
    egitim_yolu : Path
        Eğitim veri setinin yolu.
    test_yolu : Path
        Test veri setinin yolu.

    Döndürür
    --------
    tuple (pd.DataFrame, pd.DataFrame)
        Yüklenen eğitim ve test veri setleri.
    """
    try:
        egitim_veri = pd.read_csv(egitim_yolu)
        test_veri = pd.read_csv(test_yolu)
        return egitim_veri, test_veri
    except Exception as hata:
        print(f"\nVeri setleri yüklenirken hata oluştu:\n{hata}")
        return None, None


def veri_tiplerini_ve_bosluklari_duzenle(veri: pd.DataFrame) -> pd.DataFrame:
    """
    Tarih değişkenlerini datetime nesnesine dönüştürür ve kategorik sütunlardaki
    boşluk (whitespace) karakterlerini ampirik olarak NaN değerine çevirir.

    Parametreler
    ------------
    veri : pd.DataFrame
        İşlem görecek veri seti.

    Döndürür
    --------
    pd.DataFrame
        Düzeltilmiş veri seti.
    """
    # Kategorik sütunlardaki boşlukları NaN ile değiştirme
    kategorik_sutunlar = veri.select_dtypes(include=["object"]).columns
    for sutun in kategorik_sutunlar:
        if sutun not in ["appointment_date", "entry_service_date", "date_of_birth"]:
            # Sadece whitespace içeren hücreleri NaN yapar
            veri[sutun] = veri[sutun].astype(str).str.strip().replace("", np.nan)
            veri[sutun] = veri[sutun].replace("nan", np.nan)

    # Tarih dönüşümleri
    # Önceden datetime olarak kaydedilen sütunları tekrar datetime tipine dönüştürür
    if "appointment_date" in veri.columns:
        veri["appointment_date"] = pd.to_datetime(veri["appointment_date"], format="%Y-%m-%d", errors="coerce")
    if "entry_service_date" in veri.columns:
        veri["entry_service_date"] = pd.to_datetime(veri["entry_service_date"], format="%Y-%m-%d", errors="coerce")
    if "date_of_birth" in veri.columns:
        veri["date_of_birth"] = pd.to_datetime(veri["date_of_birth"], format="%d/%m/%Y", errors="coerce")

    return veri


def eksiklik_gostergelerini_ekle(veri: pd.DataFrame, aday_sutunlar: list) -> pd.DataFrame:
    """
    Eksik değere sahip sürekli sayısal veya kritik değişkenler için
    ikili (binary) eksiklik gösterge (nan indicator) sütunları oluşturur.

    Parametreler
    ------------
    veri : pd.DataFrame
        Veri seti.
    aday_sutunlar : list
        Gösterge eklenecek sütun listesi.

    Döndürür
    --------
    pd.DataFrame
        Göstergeler eklenmiş veri seti.
    """
    for sutun in aday_sutunlar:
        if sutun in veri.columns:
            yeni_ad = f"{sutun}_nan"
            veri[yeni_ad] = veri[sutun].isnull().astype(int)
    return veri


def egitim_parametrelerini_hesapla(egitim_veri: pd.DataFrame) -> dict:
    """
    Veri sızıntısını önlemek amacıyla yalnızca eğitim seti üzerinden
    doldurma (imputation) parametrelerini hesaplar.

    Parametreler
    ------------
    egitim_veri : pd.DataFrame
        Eğitim veri seti.

    Döndürür
    --------
    dict
        Hesaplanan istatistiksel parametreler (medyanlar, modlar vb.).
    """
    parametreler = {}

    # Yaş değişkeni için medyan hesabı
    if "age" in egitim_veri.columns:
        parametreler["age_median"] = egitim_veri["age"].median()

    # Meteorolojik sayısal değişkenler için medyan hesabı
    meteorolojik_sutunlar = ["average_temp_day", "average_rain_day", "max_temp_day", "max_rain_day"]
    for sutun in meteorolojik_sutunlar:
        if sutun in egitim_veri.columns:
            parametreler[f"{sutun}_median"] = egitim_veri[sutun].median()

    # entry_service_date için randevu tarihi ile arasındaki medyan gün farkı hesabı
    if "appointment_date" in egitim_veri.columns and "entry_service_date" in egitim_veri.columns:
        fark_gunler = (egitim_veri["appointment_date"] - egitim_veri["entry_service_date"]).dt.days
        # Negatif farkları veya geçersiz farkları dışarıda tutarak medyan hesaplama
        gecerli_farklar = fark_gunler[fark_gunler >= 0]
        if not gecerli_farklar.empty:
            parametreler["lead_time_median_days"] = gecerli_farklar.median()
        else:
            parametreler["lead_time_median_days"] = 0.0  # Varsayılan değer

    return parametreler


def eksik_verileri_doldur(veri: pd.DataFrame, parametreler: dict) -> pd.DataFrame:
    """
    Eğitim setinden elde edilen parametreleri kullanarak veri setindeki
    eksik gözlemleri metodolojik kurallara uygun olarak doldurur.

    Parametreler
    ------------
    veri : pd.DataFrame
        Doldurulacak veri seti.
    parametreler : dict
        Doldurma parametreleri.

    Döndürür
    --------
    pd.DataFrame
        Doldurulmuş veri seti.
    """
    veri_dolu = veri.copy()

    # 1. Yaş (age) ve Doğum Tarihi (date_of_birth) Doldurulması
    if "age" in veri_dolu.columns:
        # Öncelikli olarak: date_of_birth bilgisi olup age bilgisi olmayanları hesaplama (örn: 20 gözlem)
        maske_age_hesapla = veri_dolu["age"].isnull() & veri_dolu["date_of_birth"].notnull()
        if maske_age_hesapla.any():
            veri_dolu.loc[maske_age_hesapla, "age"] = (
                (veri_dolu.loc[maske_age_hesapla, "appointment_date"] - 
                 veri_dolu.loc[maske_age_hesapla, "date_of_birth"]).dt.days / 365.25
            ).round(0)

        # Kalan eksik yaşları eğitim medyanı ile doldurma
        yas_medyani = parametreler.get("age_median", 11.0)
        veri_dolu["age"] = veri_dolu["age"].fillna(yas_medyani)

    if "date_of_birth" in veri_dolu.columns:
        # Eksik date_of_birth değerlerini, doldurulmuş age ve appointment_date üzerinden hesaplama
        maske_dob_hesapla = veri_dolu["date_of_birth"].isnull()
        if maske_dob_hesapla.any():
            veri_dolu.loc[maske_dob_hesapla, "date_of_birth"] = (
                veri_dolu.loc[maske_dob_hesapla, "appointment_date"] - 
                pd.to_timedelta(veri_dolu.loc[maske_dob_hesapla, "age"] * 365.25, unit="D")
            )

    # 2. Servis Kayıt Tarihi (entry_service_date) Doldurulması
    if "entry_service_date" in veri_dolu.columns:
        medyan_gun_farki = parametreler.get("lead_time_median_days", 4.0)
        maske_entry_hesapla = veri_dolu["entry_service_date"].isnull()
        if maske_entry_hesapla.any():
            veri_dolu.loc[maske_entry_hesapla, "entry_service_date"] = (
                veri_dolu.loc[maske_entry_hesapla, "appointment_date"] - 
                pd.to_timedelta(medyan_gun_farki, unit="D")
            )

    # 3. Meteorolojik Sayısal Değişkenlerin Medyan ile Doldurulması
    meteorolojik_sutunlar = ["average_temp_day", "average_rain_day", "max_temp_day", "max_rain_day"]
    for sutun in meteorolojik_sutunlar:
        if sutun in veri_dolu.columns:
            medyan_deger = parametreler.get(f"{sutun}_median", 0.0)
            veri_dolu[sutun] = veri_dolu[sutun].fillna(medyan_deger)

    # 4. Kategorik Değişkenlerin Bilinmiyor (Unknown) ile Doldurulması
    kategorik_doldurulacak = ["icd", "specialty", "city", "disability"]
    for sutun in kategorik_doldurulacak:
        if sutun in veri_dolu.columns:
            veri_dolu[sutun] = veri_dolu[sutun].fillna("Bilinmiyor")

    return veri_dolu


def akademik_rapor_yazdir(ham_veri: pd.DataFrame, dolu_veri: pd.DataFrame, kume_adi: str):
    """
    Eksik veri doldurma öncesi ve sonrasındaki ampirik değişimleri
    akademik standartlara uygun bir rapor halinde terminale yansıtır.

    Parametreler
    ------------
    ham_veri : pd.DataFrame
        Doldurma öncesi ham veri seti.
    dolu_veri : pd.DataFrame
        Doldurma sonrası veri seti.
    kume_adi : str
        Raporlanan veri kümesinin ismi (Eğitim/Test).
    """
    print("\n" + "=" * 110)
    print(f"AKADEMİK RAPOR: {kume_adi.upper()} VERİ KÜMESİ EKSİK VERİ DOLDURMA (IMPUTATION) SONUÇLARI")
    print("=" * 110)

    toplam_satir = len(ham_veri)
    rapor_satirlari = []

    for sutun in ham_veri.columns:
        onceki_eksik = ham_veri[sutun].isnull().sum()
        sonraki_eksik = dolu_veri[sutun].isnull().sum()
        degisim = sonraki_eksik - onceki_eksik
        doldurulan_oran = (onceki_eksik / toplam_satir) * 100

        # Raporlama satırı oluşturma
        rapor_satirlari.append({
            "Öznitelik": sutun,
            "Başlangıç Eksik": onceki_eksik,
            "Nihai Eksik": sonraki_eksik,
            "Değişim": degisim,
            "Doldurulan Oran (%)": doldurulan_oran
        })

    rapor_df = pd.DataFrame(rapor_satirlari)
    # Sadece değişim yaşanan (eksik verisi doldurulan) satırları göstermek akademik okumayı kolaylaştırır
    degisenler = rapor_df[rapor_df["Başlangıç Eksik"] > 0].sort_values(by="Başlangıç Eksik", ascending=False)
    
    if not degisenler.empty:
        print(degisenler.to_string(index=False))
    else:
        print("Veri kümesinde eksik değere sahip öznitelik bulunmamaktadır.")
        
    print("-" * 110)
    print(f"Toplam Gözlem (Satır) Sayısı: {toplam_satir:,}")
    print(f"Toplam Öznitelik (Sütun) Sayısı (Başlangıç): {ham_veri.shape[1]}")
    print(f"Toplam Öznitelik (Sütun) Sayısı (Nihai): {dolu_veri.shape[1]}")
    print("=" * 110)


def veriyi_kaydet(veri: pd.DataFrame, dosya_yolu: Path):
    """
    İşlenmiş veri setini, tarih sütunlarını string tipine geri döndürerek
    belirtilen dosya yoluna kaydeder.

    Parametreler
    ------------
    veri : pd.DataFrame
        Kaydedilecek veri seti.
    dosya_yolu : Path
        Kayıt adresi.
    """
    kayit_kopya = veri.copy()
    
    # Tarih formatlarını kaydederken standart metin yapısına dönüştürme
    tarih_kolonlar = ["appointment_date", "entry_service_date", "date_of_birth"]
    for sutun in tarih_kolonlar:
        if sutun in kayit_kopya.columns and pd.api.types.is_datetime64_any_dtype(kayit_kopya[sutun]):
            # date_of_birth formatını korumak için %d/%m/%Y, diğerlerini %Y-%m-%d formatında kaydetme
            if sutun == "date_of_birth":
                kayit_kopya[sutun] = kayit_kopya[sutun].dt.strftime("%d/%m/%Y")
            else:
                kayit_kopya[sutun] = kayit_kopya[sutun].dt.strftime("%Y-%m-%d")

    kayit_kopya.to_csv(dosya_yolu, index=False, encoding="utf-8-sig")
    print(f"Veri başarıyla kaydedildi: {dosya_yolu.name}")


def main():
    # Dosya yollarının dinamik tanımlanması
    proje_dizini = Path(__file__).resolve().parent.parent
    veriler_dizini = proje_dizini / "veriler"
    
    egitim_yolu = veriler_dizini / "medical_appointments_train.csv"
    test_yolu = veriler_dizini / "medical_appointments_test.csv"

    # Verileri yükleme
    egitim_ham, test_ham = veri_setlerini_yukle(egitim_yolu, test_yolu)
    if egitim_ham is None or test_ham is None:
        return

    # Veri tipleri ve boşluk karakterlerinin NaN ile değiştirilmesi
    # Raporlama öncesi karşılaştırma için ham kopya üzerinde işlem yapılır
    egitim_ham_kopya = egitim_ham.copy()
    test_ham_kopya = test_ham.copy()

    egitim_analiz = veri_tiplerini_ve_bosluklari_duzenle(egitim_ham)
    test_analiz = veri_tiplerini_ve_bosluklari_duzenle(test_ham)

    # Eksiklik göstergelerinin ekleneceği sütunlar listesi
    gosterge_sutunlari = [
        "age", 
        "entry_service_date", 
        "average_temp_day", 
        "average_rain_day", 
        "max_temp_day", 
        "max_rain_day"
    ]

    # Göstergelerin eklenmesi
    egitim_gostergeli = eksiklik_gostergelerini_ekle(egitim_analiz, gosterge_sutunlari)
    test_gostergeli = eksiklik_gostergelerini_ekle(test_analiz, gosterge_sutunlari)

    # Doldurma parametrelerinin yalnızca EĞİTİM seti üzerinden hesaplanması (Leakage Önlemi)
    doldurma_parametreleri = egitim_parametrelerini_hesapla(egitim_gostergeli)

    # Verilerin doldurulması
    egitim_dolu = eksik_verileri_doldur(egitim_gostergeli, doldurma_parametreleri)
    test_dolu = eksik_verileri_doldur(test_gostergeli, doldurma_parametreleri)

    # Akademik raporların üretilmesi
    akademik_rapor_yazdir(egitim_ham_kopya, egitim_dolu, "Eğitim (Train)")
    akademik_rapor_yazdir(test_ham_kopya, test_dolu, "Test (Test)")

    # Doldurulmuş ve gösterge eklenmiş veri setlerinin kaydedilmesi
    veriyi_kaydet(egitim_dolu, egitim_yolu)
    veriyi_kaydet(test_dolu, test_yolu)


if __name__ == "__main__":
    main()
