import numpy as np
import pandas as pd
import simpy
import joblib
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent

GUN_SAYISI = 2500
SLOT_SAYISI = 20
SLOT_ARALIGI_DK = 20
MUAYENE_SURESI_DK = 18
YUKSEK_RISK_ESIGI = 0.40
YEDEK_HASTA_GECIKME_DK = 5
RNG_SEED = 42


def sampiyon_modelin_gercek_olasiliklarini_yukle(kok: Path) -> np.ndarray:
    paket = joblib.load(kok / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib")
    model = paket["model"]
    cols = paket["sutun_siralamasi"]
    test = pd.read_csv(kok / "veriler" / "medical_appointments_test.csv")
    if "appointment_time" in test.columns:
        test = test.drop(columns=["appointment_time"])
    if paket["icd_frekans_haritasi"] is not None and "icd" in test.columns:
        test["icd_frekans"] = test["icd"].map(paket["icd_frekans_haritasi"]).fillna(0.0)
        test = test.drop(columns=["icd"])
    X = test.drop(columns=["no_show"]) if "no_show" in test.columns else test
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    X = X[cols]
    return model.predict_proba(X)[:, 1]


def hasta_sureci(env, hekim, gelis_zamani, bekleme_kaydi):
    yield env.timeout(max(0, gelis_zamani - env.now))
    varis = env.now
    with hekim.request() as istek:
        yield istek
        bekleme_kaydi.append(env.now - varis)
        yield env.timeout(MUAYENE_SURESI_DK)


def tek_gun_simule_et(olasiliklar, politika, rng):
    env = simpy.Environment()
    hekim = simpy.Resource(env, capacity=1)
    bekleme_kaydi = []
    secili = rng.choice(len(olasiliklar), size=SLOT_SAYISI, replace=False)
    for i, idx in enumerate(secili):
        p_noshow = olasiliklar[idx]
        slot_zamani = i * SLOT_ARALIGI_DK
        geldi = rng.random() > p_noshow
        if geldi:
            env.process(hasta_sureci(env, hekim, slot_zamani, bekleme_kaydi))
        if politika == "hibrit" and p_noshow > YUKSEK_RISK_ESIGI:
            if rng.random() < 0.85:
                env.process(hasta_sureci(env, hekim, slot_zamani + YEDEK_HASTA_GECIKME_DK, bekleme_kaydi))
    env.run()
    toplam_sure = SLOT_SAYISI * SLOT_ARALIGI_DK
    calisilan = sum(bekleme_kaydi) + len(bekleme_kaydi) * MUAYENE_SURESI_DK
    atil = max(0, toplam_sure - calisilan)
    return {"ortalama_bekleme": np.mean(bekleme_kaydi) if bekleme_kaydi else 0.0,
            "hekim_atil": atil, "gorulen_hasta": len(bekleme_kaydi)}


def main():
    olasiliklar = sampiyon_modelin_gercek_olasiliklarini_yukle(KOK)
    rng = np.random.default_rng(RNG_SEED)
    kayit = []
    for gun in range(GUN_SAYISI):
        for pol in ["statik", "hibrit"]:
            s = tek_gun_simule_et(olasiliklar, pol, rng)
            kayit.append({"gun": gun, "politika": pol, **s})
    df = pd.DataFrame(kayit)
    print("=" * 70)
    print("KUYRUK SİMÜLASYONU ÖZET (Politika B - koşulsuz, v4 adil model)")
    print("=" * 70)
    print(df.groupby("politika").mean(numeric_only=True))
    df.to_csv(KOK / "veriler" / "simulasyon_gunluk_sonuclar.csv", index=False, encoding="utf-8-sig")
    print("-> Kaydedildi: veriler/simulasyon_gunluk_sonuclar.csv")


if __name__ == "__main__":
    main()
