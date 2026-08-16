# Hakem Gözüyle İnceleme — Eklenen Analizler

Danışman (hoca) düzeltmelerine EK olarak, bağımsız bir dergi hakeminin
sorabileceği eksikler tespit edilip giderilmiştir. Hocanın onayladığı
düzeltmelere (Madde 1-6) DOKUNULMAMIŞTIR; bunlar yeni, bağımsız script'lerdir.

## EKLENEN 1 — Karar Eğrisi Analizi (script 37)
Hakem sorusu: "ROC-AUC klinik fayda demek değil. Model gerçekten faydalı mı?"
Yöntem: Decision Curve Analysis (Vickers & Elkin, 2006) — net fayda vs
"hepsine müdahale" vs "hiçbirine müdahale".
DOĞRULANMIŞ SONUÇ: Model, eşik olasılık aralığı [0.05-0.50] boyunca her iki
naive stratejiden daha yüksek net fayda sağlıyor. Yani model bu maliyet-fayda
aralığında klinik olarak faydalı. (veriler/karar_egrisi_analizi.csv)

## EKLENEN 2 — Alt-Grup / Adalet Analizi (script 38)
Hakem sorusu: "Model farklı hasta gruplarında eşit mi çalışıyor? (fairness)"
Yöntem: v4 modelin dış test AUC'si cinsiyet ve yaş grubuna göre ayrıştırıldı.
DOĞRULANMIŞ SONUÇ:
  Cinsiyet: Kadın AUC=0.769, Erkek AUC=0.777
  Yaş: 0-17=0.778, 18-39=0.766, 40-64=0.750, 65+=0.764
  Gruplar arası fark yalnızca 0.029 (<0.10). Model kaba adalet açısından
  tutarlı; belirgin bir alt-grup dezavantajı yok. (veriler/alt_grup_adalet_analizi.csv)

## EKLENEN 3 — Naive Baseline + Overfitting Kontrolü (script 39)
Hakem sorusu: "Model önemsiz bir tahminciden iyi mi? Eğitimi ezberliyor mu?"
DOĞRULANMIŞ SONUÇ:
  (1) Model PR-AUC=0.295, prevalans tahmincisi PR-AUC=0.100 -> 3.0 kat üstün.
  (2) Eğitim AUC=0.982, Test AUC=0.775, genelleme boşluğu=0.207.
      DÜRÜST NOT: Bu boşluk RF'nin doğası gereği yüksektir (eğitim setini
      neredeyse ezberler); asıl ölçüt test performansıdır. Yine de makalede
      bu durum ve regularizasyon dengesi açıkça belirtilmelidir.
  (veriler/baseline_ve_overfitting.csv)

## İNCELENİP EKLENMEYENLER (gerekçeli)
- Öğrenme eğrisi: 46K örneklem zaten yeterli; düşük değer.
- Permutation importance: SHAP zaten var; tekrar olur.
- Çok merkezli / prospektif doğrulama: veri elvermiyor; Sınırlamalar'da yazılı.

## EKLENEN 4 — Simülasyon İkili Raporlama (script 40, danışman uyarısı)
Danışman uyarısı: "Simülasyon faydası hem iyimser (rastgele) hem gerçekçi
(kronolojik ~0.55) model altında raporlanmalı; gerçek dünyada kazanç küçülür."
DOĞRULANMIŞ SONUÇ (aynı gerçek dünya, iki farklı model kararı):
  Atıl zaman azalması:  İyimser 24.2 dk/gün -> Gerçekçi 6.5 dk/gün (-73%)
  Görülen hasta artışı: İyimser 0.52/gün    -> Gerçekçi 0.13/gün   (-76%)
  Kronolojik modelin ">0.40 riskli" diyebildiği slot oranı yalnızca %0.7
  (iyimser modelde %3.1). Her iki rejimde overbooking anlamlı (p<0.001) ama
  gerçekçi fayda, iyimserin ~%24'ü kadar. (veriler/simulasyon_ikili_raporlama.csv)
