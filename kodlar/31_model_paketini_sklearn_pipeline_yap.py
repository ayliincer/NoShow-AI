import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

KOK = Path(__file__).resolve().parent.parent

HAVA_DURUMU_SUTUNLARI = [
    "average_temp_day", "average_rain_day", "max_temp_day", "max_rain_day",
    "rainy_day_before", "storm_day_before",
    "average_temp_day_nan", "average_rain_day_nan", "max_temp_day_nan", "max_rain_day_nan",
    "temp_range", "rain_range", "is_rainy",
    "rain_intensity_heavy", "rain_intensity_moderate", "rain_intensity_no_rain", "rain_intensity_weak",
    "heat_intensity_cold", "heat_intensity_heavy_cold", "heat_intensity_heavy_warm",
    "heat_intensity_mild", "heat_intensity_warm",
]


class NoShowOnIslemci(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.icd_frekans_haritasi_ = None
        self.age_medyan_ = None
        self.sutun_sirasi_ = None

    def fit(self, X, y=None):
        X = X.copy()
        self.icd_frekans_haritasi_ = (
            X["icd"].value_counts(normalize=True).to_dict() if "icd" in X.columns else {})
        if "age" in X.columns:
            self.age_medyan_ = float(X["age"].median())
        self.sutun_sirasi_ = list(self._temel_donusum(X).columns)
        return self

    def _temel_donusum(self, X):
        X = X.copy()
        if "appointment_time" in X.columns:
            X = X.drop(columns=["appointment_time"])
        if "icd" in X.columns:
            X["icd_frekans"] = X["icd"].map(self.icd_frekans_haritasi_).fillna(0.0)
            X = X.drop(columns=["icd"])
        if "age" in X.columns and self.age_medyan_ is not None:
            X["age"] = X["age"].fillna(self.age_medyan_)
        sil = [c for c in HAVA_DURUMU_SUTUNLARI if c in X.columns]
        return X.drop(columns=sil)

    def transform(self, X):
        Xd = self._temel_donusum(X)
        for s in self.sutun_sirasi_:
            if s not in Xd.columns:
                Xd[s] = 0
        return Xd[self.sutun_sirasi_]


def main():
    eski = joblib.load(KOK / "modeller" / "nihai_no_show_model_paketi_v4_tam_adil.joblib")
    model = eski["model"]
    train_ham = pd.read_csv(KOK / "veriler" / "medical_appointments_train.csv")
    on = NoShowOnIslemci()
    on.fit(train_ham.drop(columns=["no_show"]))
    if list(on.sutun_sirasi_) == list(eski["sutun_siralamasi"]):
        print("Dogrulama: on islemci sutun sirasi model ile birebir ayni.")
    pipe = Pipeline([("on_isleme", on), ("model", model)])
    test_ham = pd.read_csv(KOK / "veriler" / "medical_appointments_test.csv")
    y_test = test_ham["no_show"].map({"no": 0, "yes": 1})
    from sklearn.metrics import roc_auc_score
    proba = pipe.predict_proba(test_ham.drop(columns=["no_show"]))[:, 1]
    print(f"Standalone Pipeline ROC-AUC: {roc_auc_score(y_test, proba):.4f} (referans: 0.7753)")
    joblib.dump({"surum": "v5-standalone-sklearn-pipeline", "model_adi": eski["model_adi"],
                 "pipeline": pipe}, KOK / "modeller" / "nihai_no_show_model_paketi_pipeline_STANDALONE.joblib")
    print("-> Kaydedildi: modeller/nihai_no_show_model_paketi_pipeline_STANDALONE.joblib")


if __name__ == "__main__":
    main()
