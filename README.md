# NoShow-AI

Hasta Randevu Devamsızlığı (No-Show) Tahmininde Performans Düşüşünün Ayrıştırılması: Hasta Sızıntısı ve Zamansal Kayma Bileşenleri Üzerine Bir Vaka Çalışması

Bu depo, poliklinik randevu devamsızlığı (no-show) tahmini için geliştirilen bir makine öğrenmesi hattının (pipeline) kodlarını ve analiz sonuçlarını içerir. Çalışmanın merkezinde, aynı modelin farklı doğrulama stratejileriyle test edildiğinde performansının nasıl ve neden değiştiği sorusu yer alır.

## Özet

Sağlık hizmetlerinde randevu devamsızlığı, hekim atıl zamanına ve kaynak israfına yol açan önemli bir sorundur. Bu projede, çok yıllı (2016–2022) bir rehabilitasyon merkezi verisi üzerinde bir devamsızlık tahmin modeli geliştirilmiş ve model performansı üç farklı doğrulama stratejisiyle karşılaştırılmıştır. Amaç, raporlanan başarının ne kadarının gerçek genelleme, ne kadarının veri sızıntısı kaynaklı olduğunu ayrıştırmaktır.

## Ana Bulgu

Aynı model (Random Forest), doğrulama stratejisine göre farklı performanslar göstermektedir:

| Doğrulama Stratejisi | ROC-AUC (%95 GA) | Ne gösterir |
|---|---|---|
| Satır-rastgele (sızıntı var) | 0,775 [0,760–0,791] | Tipik, iyimser doğrulama |
| Hasta-ayrık (GroupShuffleSplit) | 0,677 [0,660–0,693] | Hasta sızıntısı temizlenince |
| Kronolojik (2016–2020'den 2021–2022'ye) | 0,548 [0,532–0,563] | Gerçek prospektif performans |

Toplam 0,227'lik düşüşün %43,2'si (−0,098) hasta sızıntısına, %56,8'i (−0,129) zamansal kaymayla ilişkilendirilmektedir. Bu ayrıştırmanın hesaplama sırasına duyarlılığı ayrıca test edilmiş; yalnızca sızıntısız ara adımlarla ilerleyen sıranın geçerli bir ayrıştırma ürettiği doğrulanmıştır.

## Metodolojik İlkeler

- **Sızıntısız ön işleme:** Tüm parametreler (imputation medyanları, ICD frekans kodlaması, kategorik kodlama) yalnızca eğitim setinden öğrenilir.
- **İzole model seçimi:** Şampiyon model yalnızca eğitim seti ve çapraz doğrulama ile seçilir (5 katlı CV, ortalama standart sapma raporlanır); saklı dış test setine tek kez dokunulur.
- **Sızıntısız eşik seçimi:** Sınıflandırma eşiği eğitim setinin out-of-fold tahminlerinden seçilip test setine sabit uygulanır.
- **Adil karşılaştırma:** Dört topluluk algoritması (Random Forest, XGBoost, LightGBM, CatBoost) eşit koşullarda ve ayrı ayrı optimize edilir.
- **Zamansal sızıntı önleme:** Randevu gününün gerçekleşmiş hava durumu değişkenleri (randevu anında bilinemez) modelden çıkarılır.
- **Vekil kimlik bütünlüğü:** Doğum tarihi eksik olan gözlemlere benzersiz kimlik atanarak hasta-ayrık bölmedeki yapay grup çakışması giderilmiştir.
- **Çoklu sağlamlık testi:** Ayrıştırma sırası, çapraz test seti değerlendirmesi, ablasyon analizleri ve çoklu rastgele tohum karşılaştırmalarıyla bulguların kararlılığı doğrulanmıştır.

## Depo Yapısı

```
NoShow-AI/
├── gorseller/    # Grafikler ve şekiller
├── kodlar/       # Numaralı, sıralı pipeline script'leri (01–53)
├── modeller/     # Model üretim README'si (ağır .joblib dosyaları hariç)
├── veriler/      # Analiz sonuç dosyaları (CSV) — ham veri hariç (bkz. Veri Erişimi)
├── requirements.txt
└── README.md
```

## Kurulum

```bash
pip install -r requirements.txt
```

Başlıca bağımlılıklar: `scikit-learn`, `pandas`, `numpy`, `shap`, `lightgbm`, `catboost`, `xgboost`, `simpy`, `scipy`, `matplotlib`.

## Pipeline Çalıştırma Sırası

### Aşama 1 — Keşifçi Analiz (01–09)
Veri genel bakış, eksik veri, aykırı değer, düşük varyans, korelasyon ve veri sızıntısı risk değerlendirmesi.

### Aşama 2 — Ön İşleme (10–16)
Sızıntısız temizlik: tekrarlayan kayıt/sızıntı riski taşıyan değişken çıkarma (10), tarih dönüşümü (11), eğitim/test bölme (12), eksik veri profilleme ve doldurma (yalnızca eğitim setinden) (13–14), öznitelik mühendisliği (15), kategorik kodlama (16).

### Aşama 3 — Model Seçimi ve Optimizasyon (17–31)
SHAP analizi (17), MLP denemesi ve hiperparametre araması (18–19), erken simülasyon prototipleri (20–23), SMOTE varyant karşılaştırması (22), kronolojik genelleme keşfi (24), hava durumu değişkenleri çıkarılarak adil karşılaştırma (25), dört topluluk algoritmasının ayrı ayrı optimizasyonu (26–29), nihai tam-adil dış test değerlendirmesi ve şampiyon model seçimi (30), üretim-hazır pipeline paketleme (31).

### Aşama 4 — Doğrulama ve Derin Analiz (32–40)
Vekil hasta kimliği türetimi (32–33), altı-senaryo doğrulama ızgarası — ana bulgunun üretildiği yer (34), zaman-stabil kısmi kurtarma (35), çöküş mekanizmasının ayrıştırılması: taban oranı kayması, kalibrasyon bozulması, takvim bağımlılığı (36), karar eğrisi analizi (37), alt-grup adalet analizi (38), baseline ve genelleme boşluğu kontrolü (39), operasyonel simülasyonun ikili raporlanması (40).

### Aşama 5 — Ek Doğrulama ve Sağlamlık Analizleri (41–53)
- **41** — Şampiyon model seçiminin çapraz doğrulama ile gerekçelendirilmesi (CV ROC-AUC ortalama standart sapma, fold bazlı tutarlılık).
- **42, 47** — Takvim özniteliklerinin (appointment_year vd.) ablasyon analizi.
- **43–46** — Vekil kimlik türetiminde doğum tarihi eksik gözlemlerin yanlışlıkla ortak bir kategori altında toplandığının tespiti ve düzeltilmesi; çoklu-tohum kararlılık karşılaştırması.
- **48** — Naif prevalans temelli baseline karşılaştırmasının üç doğrulama stratejisine genişletilmesi.
- **49** — Farklı eğitim rejimlerinin ortak (kronolojik) test setinde çapraz değerlendirilmesi.
- **50** — Alt-grup adalet analizinin operasyonel eşikte (F1-optimal) yanlış pozitif/negatif oranlarına (FPR/FNR) ve bootstrap güven aralıklarına genişletilmesi.
- **51** — Simülasyonun maliyet tarafının (hasta bekleme süresi, hekim mesai aşımı) fayda ölçütleriyle birlikte raporlanması.
- **52** — Karar eğrisi analizinin iyimser/gerçekçi model olarak ikili raporlanması.
- **53** — Geçmiş no-show oranının tek-değişkenli ve çok-değişkenli katkısı arasındaki görünen farkın incelenmesi.

## Ek Analizler ve Sağlamlık Kontrolleri

- **Model seçim sağlamlığı:** Random Forest, 5 katlı çapraz doğrulamada hem en yüksek ortalamaya (0,7455) hem en düşük varyansa (std=0,0010) sahiptir.
- **Takvim özniteliği ablasyonu:** Takvim özniteliklerinin çıkarılması satır-rastgele ve hasta-ayrık bölmede performansı düşürürken (−0,051, −0,081), kronolojik bölmede etkisi sınırlıdır (−0,006).
- **Çapraz test seti doğrulaması:** Satır-rastgele ve hasta-ayrık modellerinin ortak kronolojik test setinde 0,93 ve 0,91 gibi yüksek skorlara ulaşması, bu iki rejimin eğitim setlerinin kronolojik test dönemiyle örtüştüğünü (gizli sızıntı) bağımsız olarak kanıtlamaktadır.
- **Alt-grup adalet:** Cinsiyet ve yaş grupları arası ROC-AUC farkı 0,03'ün altında; operasyonel eşikte FPR/FNR farkları da sınırlı düzeydedir.
- **Vekil kimlik duyarlılık analizi:** Doğum tarihi eksik gözlemlerin düzeltilmiş kimlik ataması, hasta-ayrık performansını ve kararlılığını artırmıştır.
- **Karar eğrisi analizi:** Model, makul eşik aralığında (pt=0,05–0,50) "hepsine/hiçbirine müdahale" stratejilerinden daha yüksek klinik net fayda sağlar; gerçekçi (kronolojik) model altında bu fayda ortalama %44 daha düşük olsa da tüm eşiklerde pozitif kalmaya devam eder.
- **Simülasyon ikili raporlama:** Overbooking faydası hem iyimser hem gerçekçi model altında, gerçek (gözlenmiş) test etiketleri referans ölçüt alınarak raporlanır; gerçekçi rejimdeki kazanç iyimserin yaklaşık dörtte biri kadardır. Maliyet tarafı (bekleme süresi, mesai aşımı) faydaya kıyasla ~23–26 kat küçüktür.

## Veri Erişimi

Ham ve işlenmiş hasta verisi, gizlilik (veri koruma / etik kurul) ve boyut nedeniyle bu depoya dahil edilmemiştir. Depo yalnızca analiz sonuç dosyalarını (CSV) içerir. Ham veri, ilgili kurumun etik onayı çerçevesinde talep üzerine sağlanabilir; elde edildikten sonra ön işleme script'leri (10–16) ile işlenmiş dosyalar yeniden üretilebilir.

## Sınırlamalar

- Çalışma tek bir merkeze ait veriye dayanmaktadır.
- Hasta geçmişi öznitelikleri, doğrudan hasta kimliği bulunmadığından doğum tarihi–cinsiyet–şehir tabanlı bir vekil kimlik üzerinden geçmişe dönük olarak türetilmiştir.
- Öznitelik setinde kullanılan yaş (age) değişkeni, hastanın randevu anındaki değil, ham veri setinin kendi tasarımından kaynaklanan sabit bir referans yılına (2022) göre hesaplanmış yaşı yansıtmaktadır.
- Kronolojik değerlendirme tek bir bölünme noktasına dayanır ve model gelecek dönem verileri üzerinde henüz doğrulanmamıştır.

## Yazarlar

- Aylin Cer
- Beyza Nur Dinçer

Yönetim Bilişim Sistemleri, Karadeniz Teknik Üniversitesi

## Lisans ve Atıf

Bu depo akademik bir çalışmanın parçasıdır. Kullanım ve atıf için lütfen depo sahipleriyle iletişime geçiniz.
