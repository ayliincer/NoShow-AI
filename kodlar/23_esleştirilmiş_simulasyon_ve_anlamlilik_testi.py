import numpy as np
import pandas as pd
import simpy
import joblib
from pathlib import Path
from scipy import stats

RASTGELE_TOHUM = 42
KLINIK_SURESI_DK = 300
SLOT_ARALIGI_DK = 15
SLOT_SAYISI = KLINIK_SURESI_DK // SLOT_ARALIGI_DK
ORTALAMA_MUAYENE_SURESI_DK = 12
YEDEK_HASTA_GECIKME_DK = 5
YUKSEK_RISK_ESIGI = 0.40
GUNLUK_SIMULASYON_TEKRARI = 300
KOK = Path(__file__).resolve().parent.parent


def sampiyon_modelin_gercek_olasiliklarini_yukle(kok: Path) -> np.ndarray:
    # Madde 2 (danışman): simülasyonun istatistiksel testi de sonuç tablosundaki
    # modelle aynı olasılıkları kullanmalı -> v4 "tam adil" paket (scaler'sız).
    paket = joblib.load(kok / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib")
    test = pd.read_csv(kok / "veriler" / "medical_appointments_test.csv")
    if "appointment_time" in test.columns:
        test = test.drop(columns=["appointment_time"])
    if paket["icd_frekans_haritasi"] is not None and "icd" in test.columns:
        test["icd_frekans"] = test["icd"].map(paket["icd_frekans_haritasi"]).fillna(0.0)
        test = test.drop(columns=["icd"])
    X_test = test.drop(columns=["no_show"])
    for c in paket["sutun_siralamasi"]:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[paket["sutun_siralamasi"]]
    return paket["model"].predict_proba(X_test)[:, 1]


def gunluk_hasta_populasyonu_uret(rng: np.random.Generator, tum_olasiliklar: np.ndarray) -> pd.DataFrame:
    """TEK bir popülasyon üretir; hem A hem B bu popülasyona karşı simüle edilecek (CRN)."""
    secilen_olasiliklar = rng.choice(tum_olasiliklar, size=SLOT_SAYISI, replace=True)
    gercek_gelis = rng.random(SLOT_SAYISI) > secilen_olasiliklar
    return pd.DataFrame({
        "slot_no": np.arange(SLOT_SAYISI),
        "slot_zamani_dk": np.arange(SLOT_SAYISI) * SLOT_ARALIGI_DK,
        "no_show_olasiligi": secilen_olasiliklar,
        "geldi_mi": gercek_gelis,
    })


def hasta_sureci(env, isim, gelis_zamani, hekim, muayene_sureleri_rng, sonuclar):
    yield env.timeout(max(0, gelis_zamani - env.now))
    varis_ani = env.now
    with hekim.request() as istek:
        yield istek
        bekleme_suresi = env.now - varis_ani
        muayene_suresi = max(3.0, muayene_sureleri_rng.exponential(ORTALAMA_MUAYENE_SURESI_DK))
        yield env.timeout(muayene_suresi)
        sonuclar["bekleme_sureleri"].append(bekleme_suresi)
        sonuclar["gorulen_hasta_sayisi"] += 1
        sonuclar["klinik_bitis_ani"] = max(sonuclar["klinik_bitis_ani"], env.now)


def tek_gun_simule_et(hasta_df: pd.DataFrame, politika: str, muayene_tohumu: int) -> dict:
    """AYNI hasta_df (aynı gün) hem 'statik' hem 'hibrit' için kullanılır (CRN);
    muayene süresi rng'si de politikalar arasında AYNI tohumla eşleştirilir."""
    env = simpy.Environment()
    hekim = simpy.Resource(env, capacity=1)
    muayene_rng = np.random.default_rng(muayene_tohumu)

    sonuclar = {"bekleme_sureleri": [], "gorulen_hasta_sayisi": 0, "klinik_bitis_ani": 0.0,
                "atil_slot_sayisi": 0, "yedek_kullanilan_slot_sayisi": 0}
    yedek_rng = np.random.default_rng(muayene_tohumu + 999_983)

    for _, satir in hasta_df.iterrows():
        if satir["geldi_mi"]:
            env.process(hasta_sureci(
                env, f"Hasta-{satir['slot_no']}", satir["slot_zamani_dk"], hekim, muayene_rng, sonuclar
            ))
        elif politika == "statik":
            sonuclar["atil_slot_sayisi"] += 1

        if politika == "hibrit" and satir["no_show_olasiligi"] > YUKSEK_RISK_ESIGI:
            if yedek_rng.random() < 0.85:
                yedek_gelis_zamani = satir["slot_zamani_dk"] + YEDEK_HASTA_GECIKME_DK
                env.process(hasta_sureci(
                    env, f"Yedek-{satir['slot_no']}", yedek_gelis_zamani, hekim, muayene_rng, sonuclar
                ))
                sonuclar["yedek_kullanilan_slot_sayisi"] += 1

    env.run()
    ortalama_bekleme = float(np.mean(sonuclar["bekleme_sureleri"])) if sonuclar["bekleme_sureleri"] else 0.0
    hekim_mesai_asimi = max(0.0, sonuclar["klinik_bitis_ani"] - KLINIK_SURESI_DK)
    atil_sure_dk = sonuclar["atil_slot_sayisi"] * SLOT_ARALIGI_DK
    return {
        "ortalama_hasta_bekleme_dk": ortalama_bekleme,
        "hekim_atil_sure_dk": atil_sure_dk,
        "hekim_mesai_asimi_dk": hekim_mesai_asimi,
        "gorulen_hasta_sayisi": sonuclar["gorulen_hasta_sayisi"],
    }


def esleştirilmiş_simulasyonu_yurut(tum_olasiliklar: np.ndarray) -> pd.DataFrame:
    populasyon_rng = np.random.default_rng(RASTGELE_TOHUM)
    kayitlar = []
    for gun in range(GUNLUK_SIMULASYON_TEKRARI):
        hasta_df = gunluk_hasta_populasyonu_uret(populasyon_rng, tum_olasiliklar)
        muayene_tohumu = int(populasyon_rng.integers(0, 1_000_000))
        for politika in ("statik", "hibrit"):
            sonuc = tek_gun_simule_et(hasta_df, politika, muayene_tohumu)
            sonuc["gun"] = gun
            sonuc["politika"] = politika
            kayitlar.append(sonuc)
    return pd.DataFrame(kayitlar)


def anlamlilik_testleri_uygula(df: pd.DataFrame) -> pd.DataFrame:
    statik = df[df["politika"] == "statik"].sort_values("gun").reset_index(drop=True)
    hibrit = df[df["politika"] == "hibrit"].sort_values("gun").reset_index(drop=True)

    metrikler = {
        "ortalama_hasta_bekleme_dk": "Ort. Hasta Bekleme Süresi (dk)",
        "hekim_atil_sure_dk": "Hekim Atıl Süresi (dk/gün)",
        "hekim_mesai_asimi_dk": "Hekim Mesai Aşımı (dk/gün)",
        "gorulen_hasta_sayisi": "Görülen Hasta Sayısı (gün)",
    }

    satirlar = []
    for kod, ad in metrikler.items():
        a = statik[kod].values
        b = hibrit[kod].values
        fark = b - a
        t_stat, t_p = stats.ttest_rel(a, b)
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        cohen_d_z = fark.mean() / fark.std(ddof=1)
        iyilesme_pct = (a.mean() - b.mean()) / a.mean() * 100 if a.mean() != 0 else np.nan

        satirlar.append({
            "Metrik": ad,
            "Statik Ort.": round(a.mean(), 2),
            "Hibrit Ort.": round(b.mean(), 2),
            "İyileşme (%)": round(iyilesme_pct, 2),
            "Paired t-test p": f"{t_p:.2e}",
            "Wilcoxon p": f"{w_p:.2e}" if not np.isnan(w_p) else "N/A",
            "Cohen's d_z (eşleştirilmiş)": round(cohen_d_z, 3),
            "İstatistiksel Anlamlılık (p<0.05)": "EVET" if t_p < 0.05 else "HAYIR",
        })
    return pd.DataFrame(satirlar)


def main():
    print("=" * 110)
    print("EŞLEŞTİRİLMİŞ (ORTAK RASTGELE SAYILAR / CRN) SİMÜLASYON VE ANLAMLILIK TESTİ")
    print("=" * 110)

    olasiliklar = sampiyon_modelin_gercek_olasiliklarini_yukle(KOK)
    df = esleştirilmiş_simulasyonu_yurut(olasiliklar)
    df.to_csv(KOK / "veriler" / "simulasyon_esleştirilmiş_gunluk_sonuclar.csv", index=False, encoding="utf-8-sig")

    rapor = anlamlilik_testleri_uygula(df)
    rapor.to_csv(KOK / "veriler" / "simulasyon_anlamlilik_testi_sonuclari.csv", index=False, encoding="utf-8-sig")

    print("\n" + rapor.to_string(index=False))
    print("\n-> Kaydedildi: simulasyon_esleştirilmiş_gunluk_sonuclar.csv")
    print("-> Kaydedildi: simulasyon_anlamlilik_testi_sonuclari.csv")
    print("=" * 110)


if __name__ == "__main__":
    main()