import pandas as pd
import numpy as np
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent


def main():
    s2 = pd.read_csv(KOK / "veriler" / "medical_appointments_preprocessed_step02.csv")
    print(f"step02 yüklendi: {len(s2):,} satır (resmi kohort ile aynı olmalı: 46.641)")

    s2 = s2.reset_index(drop=False).rename(columns={"index": "_orijinal_sira"})

    # Pseudo kimlik
    s2["pseudo_id"] = (s2["date_of_birth"].astype(str) + "_"
                       + s2["gender"].astype(str) + "_"
                       + s2["city"].astype(str))

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
    print("Sıra doğrulaması: OK (split ile birebir hizalanacak)")

    cikti = KOK / "veriler" / "step02_pseudo_gecmis_dahil.csv"
    s2.to_csv(cikti, index=False, encoding="utf-8-sig")
    print(f"\nGeçmiş öznitelik özeti:")
    print(s2[gecmis_kolon].describe().round(3))
    print(f"\n-> Kaydedildi: {cikti.name}")


if __name__ == "__main__":
    main()
