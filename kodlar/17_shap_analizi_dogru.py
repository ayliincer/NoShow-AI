import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent


def main(ornek_boyutu: int = 1000, random_state: int = 42):
    paket = joblib.load(KOK / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib")
    model = paket["model"]
    model_adi = paket["model_adi"]
    cols = paket["sutun_siralamasi"]

    print(f"Yüklenen gerçek şampiyon model: {model_adi}")
    print(f"Öznitelik sayısı: {len(cols)}")

    test = pd.read_csv(KOK / "veriler" / "medical_appointments_test.csv")
    if "appointment_time" in test.columns:
        test.drop(columns=["appointment_time"], inplace=True)
    test["icd_frekans"] = test["icd"].map(paket["icd_frekans_haritasi"]).fillna(0.0)
    test.drop(columns=["icd"], inplace=True)
    X_test = test.drop(columns=["no_show"])[cols]

    n = min(ornek_boyutu, len(X_test))
    X_sample = X_test.sample(n=n, random_state=random_state)
    print(f"SHAP için {n} hastalık örneklem (saklı dış test setinden, train'e hiç dokunulmadan)")

    if "Logistic" in model_adi:
        arka_plan = X_test.sample(n=min(200, len(X_test)), random_state=random_state)
        explainer = shap.LinearExplainer(model, arka_plan)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_sample, check_additivity=False)
    vals = shap_values.values
    if len(vals.shape) == 3:
        vals = vals[:, :, 1]
        shap_values_gorsel = shap_values[:, :, 1]
    else:
        shap_values_gorsel = shap_values

    mean_abs = np.abs(vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15]
    print(f"\n=== {model_adi} — Top 15 Öznitelik (Mean |SHAP|) ===")
    for rank, i in enumerate(order, 1):
        print(f"{rank:2d}. {cols[i]:35s} {mean_abs[i]:.5f}")

    gorseller_klasoru = KOK / "gorseller"
    gorseller_klasoru.mkdir(exist_ok=True)

    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values_gorsel, max_display=20, show=False)
    plt.title(f"SHAP Summary Plot ({model_adi})")
    plt.tight_layout()
    plt.savefig(gorseller_klasoru / "shap_summary_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values_gorsel, max_display=20, show=False)
    plt.title(f"Global Feature Importance ({model_adi} - SHAP)")
    plt.tight_layout()
    plt.savefig(gorseller_klasoru / "shap_bar_plot.png", dpi=150)
    plt.close()

    pd.DataFrame({"ozellik": cols, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    ).to_csv(KOK / "veriler" / "shap_nihai_model_siralamasi.csv", index=False, encoding="utf-8-sig")

    print(f"\n-> Kaydedildi: gorseller/shap_summary_plot.png, shap_bar_plot.png (başlıklar '{model_adi}' olarak dinamik)")
    print("-> Kaydedildi: veriler/shap_nihai_model_siralamasi.csv")


if __name__ == "__main__":
    main()
