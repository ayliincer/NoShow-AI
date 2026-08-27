import numpy as np
import pandas as pd
import joblib
import simpy
from pathlib import Path
from scipy import stats
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
    cols = paket["sutun_siralamasi"]

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
        kayit.append(env.now - varis)
        yield env.timeout(MUAYENE_SURESI_DK)


def tek_gun(olasiliklar, gunun_slotlari, gercek_gelme, politika, rng):
    env = simpy.Environment()
    hekim = simpy.Resource(env, capacity=1)
    bekleme = []
    for i, idx in enumerate(gunun_slotlari):
        slot = i * SLOT_ARALIGI_DK
        if gercek_gelme[idx]:
            env.process(hasta_sureci(env, hekim, slot, bekleme))

        if politika == "overbooking" and olasiliklar[idx] > YUKSEK_RISK_ESIGI:
            if rng.random() < 0.85:
                env.process(hasta_sureci(env, hekim, slot + YEDEK_GECIKME_DK, bekleme))
    env.run()
    toplam = SLOT_SAYISI * SLOT_ARALIGI_DK
    calisilan = sum(bekleme) + len(bekleme) * MUAYENE_SURESI_DK
    return {"gorulen": len(bekleme), "atil": max(0, toplam - calisilan),
            "ort_bekleme": np.mean(bekleme) if bekleme else 0.0}


def senaryo_calistir(olasiliklar, gunluk_veri, etiket, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    atil_statik, atil_over, gor_statik, gor_over = [], [], [], []
    for slotlar, gg in gunluk_veri:
        s = tek_gun(olasiliklar, slotlar, gg, "statik", rng)
        o = tek_gun(olasiliklar, slotlar, gg, "overbooking", rng)
        atil_statik.append(s["atil"]); atil_over.append(o["atil"])
        gor_statik.append(s["gorulen"]); gor_over.append(o["gorulen"])
    return {
        "etiket": etiket,
        "atil_statik": np.mean(atil_statik), "atil_over": np.mean(atil_over),
        "gor_statik": np.mean(gor_statik), "gor_over": np.mean(gor_over),
        "atil_azalma": np.mean(atil_statik) - np.mean(atil_over),
        "gor_artis": np.mean(gor_over) - np.mean(gor_statik),
        "_atil_over_dizi": atil_over, "_atil_statik_dizi": atil_statik,
    }


def main():
    print("=" * 78)
    print("SİMÜLASYON İKİLİ RAPORLAMA: İyimser (rastgele) vs Gerçekçi (kronolojik) model")
    print("=" * 78)

    print("\nOlasılıklar üretiliyor...")
    p_iyimser, y_gercek_iy = iyimser_olasiliklar()
    p_gercekci, y_gercek = gercekci_olasiliklar()
    print(f"  İyimser  (v4/rastgele): ort={p_iyimser.mean():.3f}, "
          f">{YUKSEK_RISK_ESIGI} olan slot oranı={np.mean(p_iyimser>YUKSEK_RISK_ESIGI):.3f}")
    print(f"  Gerçekçi (kronolojik):  ort={p_gercekci.mean():.3f}, "
          f">{YUKSEK_RISK_ESIGI} olan slot oranı={np.mean(p_gercekci>YUKSEK_RISK_ESIGI):.3f}")

    print("\nSimülasyon çalışıyor (iki senaryo)...")
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

    print("\n" + "=" * 78)
    print(f"{'Metrik':<30} {'İyimser':>14} {'Gerçekçi':>14} {'Fark':>12}")
    print("-" * 78)
    for ad, key in [("Atıl zaman azalması (dk)", "atil_azalma"),
                    ("Görülen hasta artışı", "gor_artis")]:
        iy, ge = r_iyimser[key], r_gercekci[key]
        print(f"{ad:<30} {iy:>14.2f} {ge:>14.2f} {iy-ge:>12.2f}")

    t_iy, p_iy = stats.ttest_rel(r_iyimser["_atil_statik_dizi"], r_iyimser["_atil_over_dizi"])
    t_ge, p_ge = stats.ttest_rel(r_gercekci["_atil_statik_dizi"], r_gercekci["_atil_over_dizi"])
    print(f"\nOverbooking atıl azaltıyor mu? (eşleştirilmiş t-testi)")
    print(f"  İyimser rejim:  p={p_iy:.4g}")
    print(f"  Gerçekçi rejim: p={p_ge:.4g}")

    kayit = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in [r_iyimser, r_gercekci]])
    kayit.to_csv(KOK / "veriler" / "simulasyon_ikili_raporlama.csv", index=False, encoding="utf-8-sig")

    if r_iyimser["gor_artis"] != 0:
        oran = r_gercekci["gor_artis"] / r_iyimser["gor_artis"]
        print("\n" + "=" * 78)
        print("YORUM:")
        print(f"Gerçekçi (kronolojik) model altında görülen-hasta kazancı, iyimser")
        print(f"rejimdekinin ~{oran:.0%}'i kadardır. Yani rastgele-split olasılıklarıyla")
        print(f"raporlanan operasyonel fayda, gerçek prospektif koşulda ölçülenden")
        print(f"belirgin biçimde büyüktür; iyimser rejim faydayı abartmaktadır.")
    print("-> Kaydedildi: veriler/simulasyon_ikili_raporlama.csv")


if __name__ == "__main__":
    main()
