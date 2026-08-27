import numpy as np
import pandas as pd
import simpy
import joblib
from pathlib import Path
RASTGELE_TOHUM = 42
KLINIK_SURESI_DK = 300
SLOT_ARALIGI_DK = 15
SLOT_SAYISI = KLINIK_SURESI_DK // SLOT_ARALIGI_DK
ORTALAMA_MUAYENE_SURESI_DK = 12
GRACE_PERIOD_DK = 5
YEDEK_CAGRI_SONRASI_VARIS_DK = 12
YUKSEK_RISK_ESIGI = 0.40
GUNLUK_SIMULASYON_TEKRARI = 2500


def sampiyon_modelin_gercek_olasiliklarini_yukle(kok: Path) -> np.ndarray:
    model_yolu = kok / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib"
    test_yolu = kok / "veriler" / "medical_appointments_test.csv"

    paket = joblib.load(model_yolu)
    test = pd.read_csv(test_yolu)

    if "appointment_time" in test.columns:
        test = test.drop(columns=["appointment_time"])

    if paket["icd_frekans_haritasi"] is not None and "icd" in test.columns:
        test["icd_frekans"] = test["icd"].map(paket["icd_frekans_haritasi"]).fillna(0.0)
        test = test.drop(columns=["icd"])

    y_test = test["no_show"].map({"no": 0, "yes": 1})
    X_test = test.drop(columns=["no_show"])
    for c in paket["sutun_siralamasi"]:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[paket["sutun_siralamasi"]]

    olasiliklar = paket["model"].predict_proba(X_test)[:, 1]
    print(f"Bilgi: Şampiyon modelden {len(olasiliklar):,} gerçek no-show olasılık skoru yüklendi.")
    print(f"Bilgi: Ortalama tahmini no-show olasılığı: {olasiliklar.mean():.4f} "
          f"(Gerçek gözlenen no-show oranı: {y_test.mean():.4f})")
    return olasiliklar


def gunluk_hasta_populasyonu_uret(rng: np.random.Generator, tum_olasiliklar: np.ndarray) -> pd.DataFrame:
    secilen_olasiliklar = rng.choice(tum_olasiliklar, size=SLOT_SAYISI, replace=True)
    gercek_gelis = rng.random(SLOT_SAYISI) > secilen_olasiliklar  # True = geldi

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


def tek_gun_simule_et(hasta_df: pd.DataFrame, politika: str, rng: np.random.Generator) -> dict:
    env = simpy.Environment()
    hekim = simpy.Resource(env, capacity=1)
    muayene_rng = np.random.default_rng(rng.integers(0, 1_000_000))

    sonuclar = {"bekleme_sureleri": [], "gorulen_hasta_sayisi": 0, "klinik_bitis_ani": 0.0,
                "atil_slot_sayisi": 0, "yedek_kullanilan_slot_sayisi": 0}

    for _, satir in hasta_df.iterrows():
        if satir["geldi_mi"]:
            env.process(hasta_sureci(
                env, f"Hasta-{satir['slot_no']}", satir["slot_zamani_dk"], hekim, muayene_rng, sonuclar
            ))
        elif politika == "statik":
            sonuclar["atil_slot_sayisi"] += 1

        if politika == "hibrit" and satir["no_show_olasiligi"] > YUKSEK_RISK_ESIGI:
            yedek_gelis_ihtimali = rng.random() < 0.85
            if yedek_gelis_ihtimali:
                yedek_gelis_zamani = satir["slot_zamani_dk"] + 5
                env.process(hasta_sureci(
                    env, f"Yedek-{satir['slot_no']}", yedek_gelis_zamani, hekim, muayene_rng, sonuclar
                ))
                sonuclar["yedek_kullanilan_slot_sayisi"] += 1

        if politika == "kosullu" and satir["no_show_olasiligi"] > YUKSEK_RISK_ESIGI and not satir["geldi_mi"]:
            yedek_gelis_ihtimali = rng.random() < 0.85
            if yedek_gelis_ihtimali:
                yedek_gelis_zamani = satir["slot_zamani_dk"] + GRACE_PERIOD_DK + YEDEK_CAGRI_SONRASI_VARIS_DK
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


def politikayi_coklu_gun_simule_et(tum_olasiliklar: np.ndarray, politika: str, tekrar: int) -> pd.DataFrame:
    tohum_haritasi = {"statik": RASTGELE_TOHUM, "hibrit": RASTGELE_TOHUM + 1, "kosullu": RASTGELE_TOHUM + 1}
    rng = np.random.default_rng(tohum_haritasi[politika])
    gunluk_sonuclar = []
    for gun in range(tekrar):
        hasta_df = gunluk_hasta_populasyonu_uret(rng, tum_olasiliklar)
        sonuc = tek_gun_simule_et(hasta_df, politika, rng)
        sonuc["gun"] = gun
        gunluk_sonuclar.append(sonuc)
    return pd.DataFrame(gunluk_sonuclar)


def main():
    from scipy import stats
    kok = Path(__file__).resolve().parent.parent

    print("=" * 110)
    print("KOŞULLU (GERÇEK-ZAMANLI) AKTİVASYON POLİTİKASI TESTİ")
    print(f"Simüle Edilen Klinik Günü Sayısı: {GUNLUK_SIMULASYON_TEKRARI} | Slot/Gün: {SLOT_SAYISI}")
    print("=" * 110)

    tum_olasiliklar = sampiyon_modelin_gercek_olasiliklarini_yukle(kok)

    print("\n-> Politika A (Statik 1:1 Randevu) simüle ediliyor...")
    statik_df = politikayi_coklu_gun_simule_et(tum_olasiliklar, "statik", GUNLUK_SIMULASYON_TEKRARI)

    print("-> Politika C (Koşullu/Gerçek-Zamanlı Aktivasyon) simüle ediliyor...")
    kosullu_df = politikayi_coklu_gun_simule_et(tum_olasiliklar, "kosullu", GUNLUK_SIMULASYON_TEKRARI)

    print("\n" + "=" * 110)
    print(f"SONUÇ TABLOSU (Ortalama, N={GUNLUK_SIMULASYON_TEKRARI} klinik günü) — Statik (A) vs Koşullu (C)")
    print("=" * 110)
    metrikler = ["ortalama_hasta_bekleme_dk", "hekim_atil_sure_dk", "hekim_mesai_asimi_dk", "gorulen_hasta_sayisi"]
    isimler = {
        "ortalama_hasta_bekleme_dk": "Ort. Hasta Bekleme Süresi (dk)",
        "hekim_atil_sure_dk": "Hekim Atıl Süresi (dk/gün)",
        "hekim_mesai_asimi_dk": "Hekim Mesai Aşımı (dk/gün)",
        "gorulen_hasta_sayisi": "Görülen Hasta Sayısı (gün)",
    }
    satirlar = []
    for m in metrikler:
        a = statik_df[m].values
        c = kosullu_df[m].values
        t_stat, t_p = stats.ttest_rel(c, a)
        fark = (c - a).mean()
        ci = stats.t.interval(0.95, len(a) - 1, loc=fark, scale=stats.sem(c - a))
        satir = {
            "Metrik": isimler[m],
            "Statik (A)": round(a.mean(), 2),
            "Koşullu (C)": round(c.mean(), 2),
            "Fark (C-A)": round(fark, 3),
            "%95 GA": f"({ci[0]:+.3f}, {ci[1]:+.3f})",
            "p-değeri": round(t_p, 6),
            "Anlamlı mı (a=0.05)": "Evet" if t_p < 0.05 else "Hayır",
        }
        satirlar.append(satir)
        print(f"\n{isimler[m]}:")
        print(f"  Statik={a.mean():.2f}  Koşullu={c.mean():.2f}  Fark={fark:+.3f}  %95GA={ci}  p={t_p:.6f}")

    kok_out = kok / "veriler"
    pd.DataFrame(satirlar).to_csv(kok_out / "simulasyon_kosullu_politika_karsilastirma.csv", index=False, encoding="utf-8-sig")
    statik_df["politika"] = "Statik"; kosullu_df["politika"] = "Kosullu"
    pd.concat([statik_df, kosullu_df], ignore_index=True).to_csv(
        kok_out / "simulasyon_kosullu_gunluk_sonuclar.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n-> Kaydedildi: simulasyon_kosullu_politika_karsilastirma.csv")


if __name__ == "__main__":
    main()

