import numpy as np
import pandas as pd
import joblib
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent


def olasiliklari_al():
    p = joblib.load(KOK / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib")
    cols = p["sutun_siralamasi"]
    test = pd.read_csv(KOK / "veriler" / "medical_appointments_test.csv")
    if "appointment_time" in test.columns:
        test = test.drop(columns=["appointment_time"])
    test["icd_frekans"] = test["icd"].map(p["icd_frekans_haritasi"]).fillna(0.0)
    test = test.drop(columns=["icd"])
    y = test["no_show"].map({"no": 0, "yes": 1}).values
    X = test.drop(columns=["no_show"])
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    X = X[cols]
    return p["model"].predict_proba(X)[:, 1], y


def net_fayda(y, proba, pt):
    n = len(y)
    pred = (proba >= pt).astype(int)
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    return tp / n - (fp / n) * (pt / (1 - pt))


def net_fayda_hepsi(y, pt):
    prevalence = y.mean()
    return prevalence - (1 - prevalence) * (pt / (1 - pt))


def main():
    proba, y = olasiliklari_al()
    prevalence = y.mean()
    print("=" * 70)
    print("KARAR EĞRİSİ ANALİZİ (Decision Curve Analysis)")
    print("=" * 70)
    print(f"Test seti N={len(y)}, no-show prevalansı={prevalence:.3f}\n")
    print(f"{'Eşik(pt)':>8} | {'Model':>10} | {'Hepsi':>10} | {'Hiçbiri':>8} | En iyi")
    print("-" * 60)

    kayit = []
    for pt in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        nb_model = net_fayda(y, proba, pt)
        nb_hepsi = net_fayda_hepsi(y, pt)
        nb_hicbiri = 0.0
        en_iyi = max([("Model", nb_model), ("Hepsi", nb_hepsi), ("Hiçbiri", nb_hicbiri)],
                     key=lambda x: x[1])[0]
        print(f"{pt:>8.2f} | {nb_model:>10.4f} | {nb_hepsi:>10.4f} | {nb_hicbiri:>8.4f} | {en_iyi}")
        kayit.append({"esik_pt": pt, "nb_model": nb_model, "nb_hepsi": nb_hepsi,
                      "nb_hicbiri": nb_hicbiri, "en_iyi_strateji": en_iyi})

    df = pd.DataFrame(kayit)
    df.to_csv(KOK / "veriler" / "karar_egrisi_analizi.csv", index=False, encoding="utf-8-sig")

    model_kazanan = df[df["en_iyi_strateji"] == "Model"]
    print("\n" + "=" * 70)
    print("YORUM:")
    if len(model_kazanan) > 0:
        aralik = f"{model_kazanan['esik_pt'].min():.2f}-{model_kazanan['esik_pt'].max():.2f}"
        print(f"Model, eşik olasılık aralığı [{aralik}] içinde hem 'hepsine müdahale'")
        print("hem de 'hiçbirine müdahale' stratejisinden daha yüksek net fayda sağlar.")
        print("Bu, modelin bu maliyet-fayda aralığında klinik olarak faydalı olduğunu gösterir.")
    else:
        print("Model, hiçbir eşikte basit stratejilerden anlamlı üstünlük sağlamıyor;")
        print("bu, mütevazı ayrım gücüyle (kronolojik) tutarlı bir dürüst bulgudur.")
    print("-> Kaydedildi: veriler/karar_egrisi_analizi.csv")


if __name__ == "__main__":
    main()
