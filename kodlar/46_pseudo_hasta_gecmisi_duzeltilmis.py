import pandas as pd
import numpy as np
from pathlib import Path


KOK = Path(__file__).resolve().parent.parent

def pseudo_kimlik_olustur_ESKI(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pseudo_id"] = (
        df["date_of_birth"].astype(str) + "_"
        + df["gender"].astype(str) + "_"
        + df["city"].astype(str)
    )
    return df


def pseudo_kimlik_olustur_DUZELTILMIS(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dob_var = df["date_of_birth"].notna()
    pseudo_id = pd.Series(index=df.index, dtype=object)
    pseudo_id.loc[dob_var] = (
        df.loc[dob_var, "date_of_birth"].astype(str) + "_"
        + df.loc[dob_var, "gender"].astype(str) + "_"
        + df.loc[dob_var, "city"].astype(str)
    )
    pseudo_id.loc[~dob_var] = "unique_" + df.index[~dob_var].astype(str)
    df["pseudo_id"] = pseudo_id
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


def gecmis_gelecek_iliskisi(ham: pd.DataFrame) -> pd.DataFrame:
    tekrar = ham[ham["gecmis_randevu_sayisi"] > 0].copy()
    tekrar["grup"] = pd.cut(tekrar["gecmis_no_show_orani"],
                            bins=[-0.01, 0, 0.25, 0.5, 0.75, 1.0],
                            labels=["0%", "0-25%", "25-50%", "50-75%", "75-100%"])
    return tekrar.groupby("grup", observed=True)["no_show_bin"].agg(["mean", "count"])


def main():
    ham_orijinal = pd.read_csv(KOK / "veriler" / "medical-appointments-no-show-en.csv")
    print(f"Ham veri: {len(ham_orijinal):,} satır")
    n_dob_yok = ham_orijinal["date_of_birth"].isna().sum()
    print(f"date_of_birth eksik satır sayısı: {n_dob_yok:,} ({n_dob_yok/len(ham_orijinal):.1%})")


    ham_eski = pseudo_kimlik_olustur_ESKI(ham_orijinal.copy())
    kalite_eski = vekil_kimlik_kalitesini_olc(ham_eski)

    ham_yeni = pseudo_kimlik_olustur_DUZELTILMIS(ham_orijinal.copy())
    kalite_yeni = vekil_kimlik_kalitesini_olc(ham_yeni)

    print("\n" + "=" * 100)
    print("VEKİL KİMLİK KALİTE RAPORU: ESKİ (bozuk) vs YENİ (düzeltilmiş)")
    print("=" * 100)
    karsilastirma = []
    for k in kalite_eski.keys():
        karsilastirma.append({
            "Metrik": k,
            "Eski (bozuk pseudo_id)": kalite_eski[k],
            "Düzeltilmiş pseudo_id": kalite_yeni[k],
        })
        print(f"{k:32s} | eski={kalite_eski[k]!s:>14s} | yeni={kalite_yeni[k]!s:>14s}")
    df_kalite = pd.DataFrame(karsilastirma)
    kalite_yolu = KOK / "veriler" / "vekil_kimlik_kalite_raporu_karsilastirma.csv"
    df_kalite.to_csv(kalite_yolu, index=False, encoding="utf-8-sig")

    ham_eski_g = gecmis_ozellikleri_turet(ham_eski)
    ham_yeni_g = gecmis_ozellikleri_turet(ham_yeni)

    iliski_eski = gecmis_gelecek_iliskisi(ham_eski_g)
    iliski_yeni = gecmis_gelecek_iliskisi(ham_yeni_g)

    print("\n" + "=" * 100)
    print("GEÇMİŞ NO-SHOW ORANI -> GELECEKTEKİ NO-SHOW ORANI (Şekil 7): ESKİ vs YENİ")
    print("=" * 100)
    print("\n--- ESKİ (bozuk pseudo_id) ---")
    print(iliski_eski)
    print("\n--- YENİ (düzeltilmiş pseudo_id) ---")
    print(iliski_yeni)

    df_iliski = pd.DataFrame({
        "Eski_ortalama": iliski_eski["mean"], "Eski_N": iliski_eski["count"],
        "Yeni_ortalama": iliski_yeni["mean"], "Yeni_N": iliski_yeni["count"],
    })
    iliski_yolu = KOK / "veriler" / "gecmis_gelecek_iliskisi_karsilastirma.csv"
    df_iliski.to_csv(iliski_yolu, encoding="utf-8-sig")

    cikti = KOK / "veriler" / "pseudo_hasta_gecmisi_ozellikleri_duzeltilmis.csv"
    ham_yeni_g.to_csv(cikti, index=False, encoding="utf-8-sig")

    print(f"\n-> Kaydedildi: {cikti.name}")
    print(f"-> Kaydedildi: {kalite_yolu.name}")
    print(f"-> Kaydedildi: {iliski_yolu.name}")


if __name__ == "__main__":
    main()