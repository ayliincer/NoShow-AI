import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score

KOK = Path(__file__).resolve().parent.parent
ESIK = 0.171
N_BOOT = 200


def hazirla():
    p = joblib.load(KOK / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib")
    cols = p["sutun_siralamasi"]
    test = pd.read_csv(KOK / "veriler" / "medical_appointments_test.csv")
    ham = test.copy()
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
    proba = p["model"].predict_proba(X)[:, 1]
    return ham, y, proba


def fpr_fnr(y_true, y_pred):
    """FP oranı (gerçek negatifler arasında) ve FN oranı (gerçek pozitifler arasında)."""
    neg = y_true == 0
    pos = y_true == 1
    fpr = np.nan if neg.sum() == 0 else (y_pred[neg] == 1).sum() / neg.sum()
    fnr = np.nan if pos.sum() == 0 else (y_pred[pos] == 0).sum() / pos.sum()
    return fpr, fnr


def bootstrap_ga(y, proba, rng_seed=42, n_boot=N_BOOT):
    rng = np.random.default_rng(rng_seed)
    y_arr = np.asarray(y)
    n = len(y_arr)
    roc_list, fpr_list, fnr_list = [], [], []
    pred = (proba >= ESIK).astype(int)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb, predb = y_arr[idx], proba[idx], pred[idx]
        if len(np.unique(yb)) >= 2:
            roc_list.append(roc_auc_score(yb, pb))
        f_p, f_n = fpr_fnr(yb, predb)
        if not np.isnan(f_p):
            fpr_list.append(f_p)
        if not np.isnan(f_n):
            fnr_list.append(f_n)

    def ga(lst):
        if len(lst) < 10:
            return (np.nan, np.nan)
        return tuple(np.percentile(lst, [2.5, 97.5]))

    return ga(roc_list), ga(fpr_list), ga(fnr_list)


def alt_grup_metrikleri(y, proba, maske, ad):
    yy, pp = y[maske], proba[maske]
    n = int(maske.sum())
    if n < 30:
        return {"Alt Grup": ad, "N": n, "not": "yetersiz örneklem (N<30)"}

    pred = (pp >= ESIK).astype(int)
    roc = roc_auc_score(yy, pp) if len(np.unique(yy)) >= 2 else np.nan
    fpr, fnr = fpr_fnr(yy, pred)
    (roc_lo, roc_hi), (fpr_lo, fpr_hi), (fnr_lo, fnr_hi) = bootstrap_ga(yy, pp)

    return {
        "Alt Grup": ad, "N": n,
        "ROC-AUC": round(roc, 4), "ROC-AUC GA_alt": round(roc_lo, 4), "ROC-AUC GA_ust": round(roc_hi, 4),
        "FPR": round(fpr, 4), "FPR GA_alt": round(fpr_lo, 4), "FPR GA_ust": round(fpr_hi, 4),
        "FNR": round(fnr, 4), "FNR GA_alt": round(fnr_lo, 4), "FNR GA_ust": round(fnr_hi, 4),
        "not": "",
    }


def main():
    ham, y, proba = hazirla()
    print("=" * 100)
    print(f"ALT-GRUP FPR/FNR ANALİZİ (v4 şampiyon model, dış test, eşik={ESIK})")
    print("=" * 100)

    kayit = []
    print("\n--- Cinsiyet ---")
    for kolon, ad in [("gender_F", "Kadın"), ("gender_M", "Erkek")]:
        if kolon in ham.columns:
            maske = ham[kolon].astype(bool).values
            r = alt_grup_metrikleri(y, proba, maske, ad)
            kayit.append(r)
            print(f"  {ad:8s}: N={r['N']:5d}  ROC-AUC={r.get('ROC-AUC','-')}  "
                  f"FPR={r.get('FPR','-')}  FNR={r.get('FNR','-')}")

    print("\n--- Yaş grubu ---")
    if "age" in ham.columns:
        yas = ham["age"].values
        for lo, hi, ad in [(0, 18, "Yaş 0-17"), (18, 40, "Yaş 18-39"), (40, 65, "Yaş 40-64"), (65, 200, "Yaş 65+")]:
            maske = (yas >= lo) & (yas < hi)
            r = alt_grup_metrikleri(y, proba, maske, ad)
            kayit.append(r)
            print(f"  {ad:10s}: N={r['N']:5d}  ROC-AUC={r.get('ROC-AUC','-')}  "
                  f"FPR={r.get('FPR','-')}  FNR={r.get('FNR','-')}")

    df = pd.DataFrame(kayit)


    def fmt_ga(row, ad):
        return f"{row[ad]:.4f} [{row[ad+' GA_alt']:.4f}-{row[ad+' GA_ust']:.4f}]"

    okunur = []
    for _, row in df.iterrows():
        okunur.append({
            "Alt Grup": row["Alt Grup"], "N": row["N"],
            "ROC-AUC [GA]": fmt_ga(row, "ROC-AUC"),
            "FPR [GA]": fmt_ga(row, "FPR"),
            "FNR [GA]": fmt_ga(row, "FNR"),
        })
    df_okunur = pd.DataFrame(okunur)

    print("\n" + "=" * 100)
    print("SONUÇ TABLOSU")
    print("=" * 100)
    print(df_okunur.to_string(index=False))


    def en_buyuk_fark(metrik):
        en_iyi = None
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                a, b = df.iloc[i], df.iloc[j]
                fark = abs(a[metrik] - b[metrik])
                if en_iyi is None or fark > en_iyi[0]:
                    en_iyi = (fark, a["Alt Grup"], a[metrik], b["Alt Grup"], b[metrik])
        return en_iyi

    fpr_fark = en_buyuk_fark("FPR")
    fnr_fark = en_buyuk_fark("FNR")

    print("\n" + "-" * 100)
    print("EN BÜYÜK İKİLİ FARK:")
    print(f"  FPR: {fpr_fark[1]} ({fpr_fark[2]:.4f}) vs {fpr_fark[3]} ({fpr_fark[4]:.4f})  -> fark={fpr_fark[0]:.4f}")
    print(f"  FNR: {fnr_fark[1]} ({fnr_fark[2]:.4f}) vs {fnr_fark[3]} ({fnr_fark[4]:.4f})  -> fark={fnr_fark[0]:.4f}")

    cikti = KOK / "veriler" / "alt_grup_fpr_fnr_analizi.csv"
    df.to_csv(cikti, index=False, encoding="utf-8-sig")
    print(f"\n-> Kaydedildi: {cikti.name}")


if __name__ == "__main__":
    main()