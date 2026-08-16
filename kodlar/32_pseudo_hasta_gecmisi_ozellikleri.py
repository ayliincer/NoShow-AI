import pandas as pd
import numpy as np
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent


def pseudo_kimlik_olustur(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pseudo_id"] = (
        df["date_of_birth"].astype(str) + "_"
        + df["gender"].astype(str) + "_"
        + df["city"].astype(str)
    )
    return df


def vekil_kimlik_kalitesini_olc(df: pd.DataFrame) -> dict:
    rapor = {}
    n_pseudo = df["pseudo_id"].nunique()
    rapor["benzersiz_pseudo_hasta"] = n_pseudo
    rapor["ortalama_randevu_per_hasta"] = len(df) / n_pseudo

    if "entry_service_date" in df.columns:
        entry_cesit = df.groupby("pseudo_id")["entry_service_date"].nunique()
        rapor["cakisma_sayisi"] = int((entry_cesit >= 2).sum())
        rapor["cakisma_orani"] = float((entry_cesit >= 2).sum() / n_pseudo)

    dtg = df["date_of_birth"].astype(str) + "_" + df["gender"].astype(str)
    sehir_cesit = df.groupby(dtg)["city"].nunique()
    rapor["bolunme_sayisi"] = int((sehir_cesit >= 2).sum())
    rapor["bolunme_orani"] = float((sehir_cesit >= 2).sum() / dtg.nunique())
    return rapor


def gecmis_ozellikleri_turet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["appointment_date"] = pd.to_datetime(df["appointment_date"], errors="coerce")
    df = df.sort_values(["pseudo_id", "appointment_date"]).reset_index(drop=True)
    df["no_show_bin"] = (df["no_show"] == "yes").astype(int)

    df["gecmis_randevu_sayisi"] = df.groupby("pseudo_id").cumcount()

    df["gecmis_no_show_sayisi"] = (
        df.groupby("pseudo_id")["no_show_bin"].cumsum() - df["no_show_bin"]
    )

    df["gecmis_no_show_orani"] = (
        df["gecmis_no_show_sayisi"] / df["gecmis_randevu_sayisi"]
    ).fillna(0.0)

    df["ilk_ziyaret_mi"] = (df["gecmis_randevu_sayisi"] == 0).astype(int)
    return df


def main():
    ham = pd.read_csv(KOK / "veriler" / "medical-appointments-no-show-en.csv")
    print(f"Ham veri: {len(ham):,} satır")

    ham = pseudo_kimlik_olustur(ham)
    kalite = vekil_kimlik_kalitesini_olc(ham)
    print("\n=== VEKİL KİMLİK KALİTE RAPORU (makalede Sınırlamalar bölümü için) ===")
    for k, v in kalite.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    ham = gecmis_ozellikleri_turet(ham)
    print("\n=== TÜRETİLEN ÖZNİTELİKLER (özet istatistik) ===")
    print(ham[["gecmis_randevu_sayisi", "gecmis_no_show_sayisi",
               "gecmis_no_show_orani", "ilk_ziyaret_mi"]].describe())

    tekrar = ham[ham["gecmis_randevu_sayisi"] > 0].copy()
    tekrar["grup"] = pd.cut(tekrar["gecmis_no_show_orani"],
                            bins=[-0.01, 0, 0.25, 0.5, 0.75, 1.0],
                            labels=["0%", "0-25%", "25-50%", "50-75%", "75-100%"])
    iliski = tekrar.groupby("grup", observed=True)["no_show_bin"].agg(["mean", "count"])
    print("\n=== GEÇMİŞ NO-SHOW ORANI -> GELECEKTEKİ NO-SHOW ORANI ===")
    print(iliski)

    cikti = KOK / "veriler" / "pseudo_hasta_gecmisi_ozellikleri.csv"
    ham.to_csv(cikti, index=False, encoding="utf-8-sig")

    pd.DataFrame([kalite]).to_csv(
        KOK / "veriler" / "vekil_kimlik_kalite_raporu.csv",
        index=False, encoding="utf-8-sig")
    print(f"\n-> Kaydedildi: {cikti.name}, vekil_kimlik_kalite_raporu.csv")


if __name__ == "__main__":
    main()
