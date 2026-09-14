import numpy as np
import pandas as pd
import joblib
import simpy
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

KOK = Path(__file__).resolve().parent.parent
GUN_SAYISI = 1500
SLOT_SAYISI = 20
SLOT_ARALIGI_DK = 20
MUAYENE_SURESI_DK = 18
YUKSEK_RISK_ESIGI = 0.40
YEDEK_GECIKME_DK = 5
RF = dict(n_estimators=300, max_depth=15, min_samples_leaf=5, max_features=0.5,
          random_state=42, n_jobs=-1)


def _hazirla_X(df, paket):
    df = df.copy()
    if "appointment_time" in df.columns:
        df = df.drop(columns=["appointment_time"])
    if paket["icd_frekans_haritasi"] is not None and "icd" in df.columns:
        df["icd_frekans"] = df["icd"].map(paket["icd_frekans_haritasi"]).fillna(0.0)
        df = df.drop(columns=["icd"])
    X = df.drop(columns=["no_show"]) if "no_show" in df.columns else df
    for c in paket["sutun_siralamasi"]:
        if c not in X.columns:
            X[c] = 0
    return X[paket["sutun_siralamasi"]]


def iyimser_olasiliklar():
    paket = joblib.load(KOK / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib")
    test = pd.read_csv(KOK / "veriler" / "medical_appointments_test.csv")
    X = _hazirla_X(test, paket)
    y_gercek = test["no_show"].map({"no": 0, "yes": 1}).to_numpy()
    return paket["model"].predict_proba(X)[:, 1], y_gercek


def gercekci_olasiliklar():
    paket = joblib.load(KOK / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib")
    train = pd.read_csv(KOK / "veriler" / "medical_appointments_train.csv")
    test = pd.read_csv(KOK / "veriler" / "medical_appointments_test.csv")
    train_kron = train[train["appointment_year"] <= 2020].copy()
    Xtr = _hazirla_X(train_kron, paket)
    ytr = train_kron["no_show"].map({"no": 0, "yes": 1})
    Xte = _hazirla_X(test, paket)
    model = RandomForestClassifier(**RF)
    model.fit(Xtr, ytr)
    y_gercek = test["no_show"].map({"no": 0, "yes": 1}).to_numpy()
    return model.predict_proba(Xte)[:, 1], y_gercek


def hasta_sureci(env, hekim, gelis, kayit):
    yield env.timeout(max(0, gelis - env.now))
    varis = env.now
    with hekim.request() as istek:
        yield istek
        kayit["bekleme"].append(env.now - varis)
        yield env.timeout(MUAYENE_SURESI_DK)
        kayit["klinik_bitis_ani"] = max(kayit["klinik_bitis_ani"], env.now)


def tek_gun(olasiliklar, gunun_slotlari, gercek_gelme, politika, rng):
    env = simpy.Environment()
    hekim = simpy.Resource(env, capacity=1)
    kayit = {"bekleme": [], "klinik_bitis_ani": 0.0}
    for i, idx in enumerate(gunun_slotlari):
        slot = i * SLOT_ARALIGI_DK
        if gercek_gelme[idx]:
            env.process(hasta_sureci(env, hekim, slot, kayit))

        if politika == "overbooking" and olasiliklar[idx] > YUKSEK_RISK_ESIGI:
            if rng.random() < 0.85:
                env.process(hasta_sureci(env, hekim, slot + YEDEK_GECIKME_DK, kayit))
    env.run()

    toplam = SLOT_SAYISI * SLOT_ARALIGI_DK
    bekleme = kayit["bekleme"]
    calisilan = sum(bekleme) + len(bekleme) * MUAYENE_SURESI_DK
    mesai_asimi = max(0.0, kayit["klinik_bitis_ani"] - toplam)
    return {
        "gorulen": len(bekleme),
        "atil": max(0, toplam - calisilan),
        "ort_bekleme": np.mean(bekleme) if bekleme else 0.0,
        "mesai_asimi": mesai_asimi,
    }


def senaryo_calistir(olasiliklar, gunluk_veri, etiket, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    (atil_statik, atil_over, gor_statik, gor_over,
     bekleme_statik, bekleme_over, mesai_statik, mesai_over) = ([], [], [], [], [], [], [], [])
    for slotlar, gg in gunluk_veri:
        s = tek_gun(olasiliklar, slotlar, gg, "statik", rng)
        o = tek_gun(olasiliklar, slotlar, gg, "overbooking", rng)
        atil_statik.append(s["atil"]); atil_over.append(o["atil"])
        gor_statik.append(s["gorulen"]); gor_over.append(o["gorulen"])
        bekleme_statik.append(s["ort_bekleme"]); bekleme_over.append(o["ort_bekleme"])
        mesai_statik.append(s["mesai_asimi"]); mesai_over.append(o["mesai_asimi"])
    return {
        "etiket": etiket,
        "atil_azalma": np.mean(atil_statik) - np.mean(atil_over),
        "gor_artis": np.mean(gor_over) - np.mean(gor_statik),
        "bekleme_degisim": np.mean(bekleme_over) - np.mean(bekleme_statik),
        "mesai_degisim": np.mean(mesai_over) - np.mean(mesai_statik),
        "bekleme_statik_ort": np.mean(bekleme_statik), "bekleme_over_ort": np.mean(bekleme_over),
        "mesai_statik_ort": np.mean(mesai_statik), "mesai_over_ort": np.mean(mesai_over),
    }


def main():
    print("=" * 100)
    print("SİMÜLASYON GENİŞLETME: BEKLEME SÜRESİ + MESAİ AŞIMI DEĞİŞİMİ")
    print("(script 40 ile aynı simülasyon; script 21/23'ten taşınan maliyet metrikleri)")
    print("=" * 100)

    print("\nOlasılıklar üretiliyor...")
    p_iyimser, y_gercek_iy = iyimser_olasiliklar()
    p_gercekci, y_gercek = gercekci_olasiliklar()

    print("\nSimülasyon çalışıyor (iki rejim)...")
    gr = np.random.default_rng(999)
    n = len(p_iyimser)
    gelmedi = (y_gercek == 1)
    gunluk_veri = []
    for _ in range(GUN_SAYISI):
        slotlar = gr.choice(n, size=SLOT_SAYISI, replace=False)
        gelme = np.ones(n, dtype=bool)
        gelme[slotlar] = ~gelmedi[slotlar]
        gunluk_veri.append((slotlar, gelme))

    r_iyimser = senaryo_calistir(p_iyimser, gunluk_veri, "İyimser (rastgele)")
    r_gercekci = senaryo_calistir(p_gercekci, gunluk_veri, "Gerçekçi (kronolojik)")

    print("\n" + "=" * 100)
    print(f"{'Metrik':<35} {'İyimser':>14} {'Gerçekçi':>14}")
    print("-" * 100)
    for ad, key in [("Atıl zaman azalması (dk)", "atil_azalma"),
                    ("Görülen hasta artışı", "gor_artis"),
                    ("Bekleme süresi değişimi (dk)", "bekleme_degisim"),
                    ("Mesai aşımı değişimi (dk)", "mesai_degisim")]:
        print(f"{ad:<35} {r_iyimser[key]:>14.3f} {r_gercekci[key]:>14.3f}")

    print("\n(Ham ortalamalar, referans için)")
    for r in [r_iyimser, r_gercekci]:
        print(f"  {r['etiket']}: bekleme statik={r['bekleme_statik_ort']:.2f} dk, "
              f"overbooking={r['bekleme_over_ort']:.2f} dk | "
              f"mesai statik={r['mesai_statik_ort']:.2f} dk, overbooking={r['mesai_over_ort']:.2f} dk")

    nihai = pd.DataFrame([
        {
            "Ölçüt/Rejim": "İyimser (rastgele)",
            "Atıl Zaman Azalması (dk)": round(r_iyimser["atil_azalma"], 2),
            "Görülen Hasta Artışı": round(r_iyimser["gor_artis"], 2),
            "Bekleme Süresi Değişimi (dk)": round(r_iyimser["bekleme_degisim"], 2),
            "Mesai Aşımı Değişimi (dk)": round(r_iyimser["mesai_degisim"], 2),
        },
        {
            "Ölçüt/Rejim": "Gerçekçi (kronolojik)",
            "Atıl Zaman Azalması (dk)": round(r_gercekci["atil_azalma"], 2),
            "Görülen Hasta Artışı": round(r_gercekci["gor_artis"], 2),
            "Bekleme Süresi Değişimi (dk)": round(r_gercekci["bekleme_degisim"], 2),
            "Mesai Aşımı Değişimi (dk)": round(r_gercekci["mesai_degisim"], 2),
        },
    ])

    print("\n" + "=" * 100)
    print("NİHAİ TABLO (mevcut fayda tablosu + 2 yeni maliyet sütunu)")
    print("=" * 100)
    print(nihai.to_string(index=False))

    cikti = KOK / "veriler" / "simulasyon_bekleme_mesai_genisletme.csv"
    nihai.to_csv(cikti, index=False, encoding="utf-8-sig")
    print(f"\n-> Kaydedildi: {cikti.name}")


if __name__ == "__main__":
    main()