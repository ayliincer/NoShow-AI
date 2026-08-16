"""
05_sayisal_degisken_dagilim_analizi.py

Ham sayısal değişkenlerin dağılım ve şekil parametrelerini (ortalama, medyan,
std, çarpıklık, basıklık) raporlayan keşifçi veri analizi. Salt-okunurdur.
"""
import pandas as pd
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent


def main():
    veri = pd.read_csv(KOK / "veriler" / "medical-appointments-no-show-en.csv")
    print("=" * 90)
    print("ANALİZ: HAM SAYISAL DEĞİŞKEN DAĞILIM VE ŞEKİL PARAMETRELERİ")
    print("=" * 90)
    sayisal = veri.select_dtypes(include=["number"])
    for kolon in sayisal.columns:
        s = sayisal[kolon].dropna()
        print(f"\nDeğişken: {kolon}")
        print(f"  N={len(s):,}  Ortalama={s.mean():.3f}  Medyan={s.median():.3f}  Std={s.std():.3f}")
        print(f"  Min/Maks={s.min():.3f}/{s.max():.3f}  Çarpıklık={s.skew():.3f}  Basıklık={s.kurtosis():.3f}")
    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
