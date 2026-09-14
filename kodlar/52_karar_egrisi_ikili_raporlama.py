"""
Script 37'yi (karar_egrisi_analizi.py) ikili raporlamaya genişletir.

Kontrol sonucu (madde 1): Script 37, SADECE "nihai_no_show_model_paketi_v4_tam_adil.joblib"
(v4 şampiyon model) ile medical_appointments_test.csv üzerindeki olasılıkları kullanıyor
- yani bu, İYİMSER (satır-rastgele) rejimi temsil ediyor. Gerçekçi (kronolojik) rejim
hiç hesaplanmamıştı.

Bu script, script 40/51 ile birebir aynı "gerçekçi" model kurulumunu (kronolojik alt
kümede (train, appointment_year<=2020) yeniden eğitilen RF, aynı test setinde
değerlendirilir) kullanarak aynı karar eğrisi analizini (Model/Hepsi/Hiçbiri net fayda,
script 37'nin kullandığı pt ızgarası: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
her iki rejim için ayrı ayrı hesaplar.

Girdi: modeller/nihai_no_show_model_paketi_v4_tam_adil.joblib,
       veriler/medical_appointments_train.csv, medical_appointments_test.csv (dokunulmadı)
Çıktı (YENİ dosya): veriler/karar_egrisi_ikili_raporlama.csv
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

KOK = Path(__file__).resolve().parent.parent
PT_IZGARASI = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]  # script 37 ile birebir ayni
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
    """Script 37 ile birebir ayni: v4 sampiyon model, medical_appointments_test.csv."""
    paket = joblib.load(KOK / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib")
    test = pd.read_csv(KOK / "veriler" / "medical_appointments_test.csv")
    X = _hazirla_X(test, paket)
    y = test["no_show"].map({"no": 0, "yes": 1}).to_numpy()
    return paket["model"].predict_proba(X)[:, 1], y


def gercekci_olasiliklar():
    """Script 40/51 ile birebir ayni: kronolojik alt kumede (<=2020) yeniden egitilen RF."""
    paket = joblib.load(KOK / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib")
    train = pd.read_csv(KOK / "veriler" / "medical_appointments_train.csv")
    test = pd.read_csv(KOK / "veriler" / "medical_appointments_test.csv")
    train_kron = train[train["appointment_year"] <= 2020].copy()
    Xtr = _hazirla_X(train_kron, paket)
    ytr = train_kron["no_show"].map({"no": 0, "yes": 1})
    Xte = _hazirla_X(test, paket)
    model = RandomForestClassifier(**RF)
    model.fit(Xtr, ytr)
    y = test["no_show"].map({"no": 0, "yes": 1}).to_numpy()
    return model.predict_proba(Xte)[:, 1], y


def net_fayda(y, proba, pt):
    n = len(y)
    pred = (proba >= pt).astype(int)
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    return tp / n - (fp / n) * (pt / (1 - pt))


def net_fayda_hepsi(y, pt):
    prevalence = y.mean()
    return prevalence - (1 - prevalence) * (pt / (1 - pt))


def karar_egrisi_hesapla(y, proba, etiket):
    kayit = []
    for pt in PT_IZGARASI:
        nb_model = net_fayda(y, proba, pt)
        nb_hepsi = net_fayda_hepsi(y, pt)
        nb_hicbiri = 0.0
        en_iyi = max([("Model", nb_model), ("Hepsi", nb_hepsi), ("Hiçbiri", nb_hicbiri)],
                     key=lambda x: x[1])[0]
        kayit.append({"rejim": etiket, "esik_pt": pt, "nb_model": nb_model,
                      "nb_hepsi": nb_hepsi, "nb_hicbiri": nb_hicbiri, "en_iyi_strateji": en_iyi})
    return kayit


def main():
    print("=" * 100)
    print("KARAR EĞRİSİ ANALİZİ - İKİLİ RAPORLAMA (İyimser vs Gerçekçi)")
    print("=" * 100)

    print("\nOlasılıklar üretiliyor...")
    p_iyimser, y_iyimser = iyimser_olasiliklar()
    p_gercekci, y_gercekci = gercekci_olasiliklar()
    print(f"  İyimser  (v4/rastgele): N={len(y_iyimser)}, prevalans={y_iyimser.mean():.4f}")
    print(f"  Gerçekçi (kronolojik):  N={len(y_gercekci)}, prevalans={y_gercekci.mean():.4f}")

    kayit_iy = karar_egrisi_hesapla(y_iyimser, p_iyimser, "İyimser (rastgele)")
    kayit_ge = karar_egrisi_hesapla(y_gercekci, p_gercekci, "Gerçekçi (kronolojik)")

    df_uzun = pd.DataFrame(kayit_iy + kayit_ge)

    # --- 3) Karsilastirma tablosu: pt satir, Iyimser/Gercekci Model NB sutun ---
    karsilastirma = []
    for pt, r_iy, r_ge in zip(PT_IZGARASI, kayit_iy, kayit_ge):
        nb_iy, nb_ge = r_iy["nb_model"], r_ge["nb_model"]
        fark = nb_ge - nb_iy
        yuzde = (fark / nb_iy * 100) if nb_iy != 0 else float("nan")
        karsilastirma.append({
            "Eşik (pt)": pt,
            "İyimser Model Net Fayda": round(nb_iy, 4),
            "Gerçekçi Model Net Fayda": round(nb_ge, 4),
            "Fark (Gerçekçi - İyimser)": round(fark, 4),
            "Düşüş (%)": round(-yuzde, 1),
        })
    df_karsilastirma = pd.DataFrame(karsilastirma)

    print("\n" + "=" * 100)
    print("KARŞILAŞTIRMA TABLOSU: MODEL NET FAYDASI, İYİMSER vs GERÇEKÇİ")
    print("=" * 100)
    print(df_karsilastirma.to_string(index=False))

    ort_dusus_mutlak = df_karsilastirma["Fark (Gerçekçi - İyimser)"].mean()
    ort_dusus_yuzde = df_karsilastirma["Düşüş (%)"].mean()
    print("\n" + "-" * 100)
    print(f"ORTALAMA (pt=0.05-0.50 üzerinden): mutlak düşüş={ort_dusus_mutlak:+.4f}, "
          f"yüzde düşüş=%{ort_dusus_yuzde:.1f}")

    cikti_uzun = KOK / "veriler" / "karar_egrisi_ikili_raporlama_uzun.csv"
    cikti_karsilastirma = KOK / "veriler" / "karar_egrisi_ikili_raporlama.csv"
    df_uzun.to_csv(cikti_uzun, index=False, encoding="utf-8-sig")
    df_karsilastirma.to_csv(cikti_karsilastirma, index=False, encoding="utf-8-sig")
    print(f"\n-> Kaydedildi: {cikti_uzun.name}")
    print(f"-> Kaydedildi: {cikti_karsilastirma.name}")


if __name__ == "__main__":
    main()
