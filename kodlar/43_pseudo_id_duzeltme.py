import pandas as pd
import numpy as np
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent


def main():
    s2 = pd.read_csv(KOK / "veriler" / "medical_appointments_preprocessed_step02.csv")
    print(f"step02 yüklendi: {len(s2):,} satır (resmi kohort ile aynı olmalı: 46.641)")

    s2 = s2.reset_index(drop=False).rename(columns={"index": "_orijinal_sira"})

    eski_pseudo_id = (s2["date_of_birth"].astype(str) + "_"
                       + s2["gender"].astype(str) + "_"
                       + s2["city"].astype(str))
    eski_benzersiz = eski_pseudo_id.nunique()
    eski_en_buyuk_grup = eski_pseudo_id.value_counts().iloc[0]

    dob_var = s2["date_of_birth"].notna()
    yeni_pseudo_id = pd.Series(index=s2.index, dtype=object)
    yeni_pseudo_id.loc[dob_var] = (
        s2.loc[dob_var, "date_of_birth"].astype(str) + "_"
        + s2.loc[dob_var, "gender"].astype(str) + "_"
        + s2.loc[dob_var, "city"].astype(str)
    )
    yeni_pseudo_id.loc[~dob_var] = "unique_" + s2.loc[~dob_var, "_orijinal_sira"].astype(str)
    s2["pseudo_id"] = yeni_pseudo_id

    n_dob_yok = (~dob_var).sum()
    yeni_benzersiz = s2["pseudo_id"].nunique()
    yeni_en_buyuk_grup = s2["pseudo_id"].value_counts().iloc[0]

    print("\n=== PSEUDO_ID DÜZELTME RAPORU ===")
    print(f"date_of_birth eksik satır sayısı: {n_dob_yok:,} ({n_dob_yok/len(s2):.1%})")
    print(f"  ESKİ mantık -> benzersiz pseudo_id: {eski_benzersiz:,} | en büyük grup: {eski_en_buyuk_grup:,} satır")
    print(f"  YENİ mantık -> benzersiz pseudo_id: {yeni_benzersiz:,} | en büyük grup: {yeni_en_buyuk_grup:,} satır")
    unique_maske = s2["pseudo_id"].str.startswith("unique_")
    unique_grup_buyuklukleri = s2.loc[unique_maske, "pseudo_id"].value_counts()
    assert (unique_grup_buyuklukleri == 1).all(), \
        "Beklenmedik: 'unique_' önekli bir pseudo_id birden fazla satırda tekrarlanıyor!"

    tmp = s2.copy()
    tmp["appointment_date"] = pd.to_datetime(tmp["appointment_date"], errors="coerce")
    tmp["no_show_bin"] = (tmp["no_show"] == "yes").astype(int)
    tmp = tmp.sort_values(["pseudo_id", "appointment_date"])

    tmp["gecmis_randevu_sayisi"] = tmp.groupby("pseudo_id").cumcount()
    tmp["gecmis_no_show_sayisi"] = (
        tmp.groupby("pseudo_id")["no_show_bin"].cumsum() - tmp["no_show_bin"])
    tmp["gecmis_no_show_orani"] = (
        tmp["gecmis_no_show_sayisi"] / tmp["gecmis_randevu_sayisi"]).fillna(0.0)
    tmp["ilk_ziyaret_mi"] = (tmp["gecmis_randevu_sayisi"] == 0).astype(int)

    tmp = tmp.sort_values("_orijinal_sira")
    gecmis_kolon = ["gecmis_randevu_sayisi", "gecmis_no_show_sayisi",
                    "gecmis_no_show_orani", "ilk_ziyaret_mi"]
    for k in gecmis_kolon:
        s2[k] = tmp[k].values

    assert (s2["_orijinal_sira"] == np.arange(len(s2))).all(), "Sıra bozuldu!"
    print("\nSıra doğrulaması: OK (split ile birebir hizalanacak)")

    cikti = KOK / "veriler" / "step02_pseudo_gecmis_duzeltilmis.csv"
    s2.to_csv(cikti, index=False, encoding="utf-8-sig")
    print(f"\nGeçmiş öznitelik özeti (düzeltilmiş pseudo_id ile):")
    print(s2[gecmis_kolon].describe().round(3))
    print(f"\n-> Kaydedildi (YENİ dosya, orijinale dokunulmadı): {cikti.name}")


if __name__ == "__main__":
    main()
