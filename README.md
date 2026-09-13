# NoShow-AI

Hasta Randevu Devamsızlığı (No-Show) Tahmininde Performans Düşüşünün Ayrıştırılması: Hasta Sızıntısı ve Zamansal Kayma Bileşenleri Üzerine Bir Vaka Çalışması

Bu depo, poliklinik randevu devamsızlığı (no-show) tahmini için geliştirilen makine öğrenmesi modelinin kodlarını, analiz sonuçlarını ve akademik makale taslağını içerir. Çalışmanın merkezinde, literatürde yaygın olarak kullanılan rastgele doğrulamanın performansı nasıl abarttığı ve bu abartının hangi bileşenlerden kaynaklandığı sorusu yer alır.

## Özet

Sağlık hizmetlerinde randevu devamsızlığı, hekim atıl zamanına ve kaynak israfına yol açan önemli bir operasyonel sorundur. Bu çalışmada, çok yıllı (2016–2022) bir rehabilitasyon merkezi verisi üzerinde bir devamsızlık tahmin modeli geliştirilmiş ve model performansı üç farklı doğrulama stratejisiyle karşılaştırılmıştır. Amaç, raporlanan başarının ne kadarının gerçek genelleme, ne kadarının veri sızıntısı kaynaklı olduğunu ayrıştırmaktır.

## Ana Bulgu

Aynı model (Random Forest), doğrulama stratejisine göre çarpıcı biçimde farklı performans göstermektedir:

| Doğrulama Stratejisi | ROC-AUC (%95 GA) | Ne gösterir |
|---|---|---|
| Satır-rastgele (sızıntı var) | 0,775 [0,760–0,791] | Literatürün tipik, iyimser raporu |
| Hasta-ayrık (GroupShuffleSplit) | 0,677 [0,660–0,693] | Hasta sızıntısı temizlenince |
| Kronolojik (2016–2020'den 2021–2022'ye) | 0,548 [0,532–0,563] | Gerçek prospektif performans |

Toplam 0,227'lik düşüşün %43,2'si (−0,098) hasta sızıntısına, %56,8'i (−0,129) zamansal kaymaya atfedilmektedir. Bu ayrıştırma, no-show tahmini literatüründeki bir boşluğu doldurmaktadır: performans düşüşü genellikle yalnızca "zamansal" olarak açıklanırken, bu çalışma düşüşün önemli bir kısmının hasta-düzeyi ezberlemeden (grup sızıntısı) kaynaklandığını, ancak baskın bileşenin zamansal kayma olduğunu göstermektedir.

## Metodolojik İlkeler

- **Sızıntısız ön işleme:** Tüm parametreler (imputation medyanları, ICD frekans kodlaması, kategorik kodlama) yalnızca eğitim setinden öğrenilir.
- **İzole model seçimi:** Şampiyon model yalnızca eğitim seti ve çapraz doğrulama ile seçilir; saklı dış test setine tek kez dokunulur.
- **Sızıntısız eşik seçimi:** Sınıflandırma eşiği eğitim setinde seçilip test setine sabit uygulanır.
- **Adil karşılaştırma:** Dört topluluk algoritması (Random Forest, XGBoost, LightGBM, CatBoost) eşit koşullarda ve ayrı ayrı optimize edilir.
- **Zamansal sızıntı önleme:** Randevu gününün gerçekleşmiş hava durumu değişkenleri (randevu anında bilinemez) modelden çıkarılır.
- **Vekil kimlik doğrulaması:** Doğum tarihi eksik olan gözlemlere benzersiz kimlik atanarak hasta-ayrık bölmedeki yapay grup çakışması giderilmiştir.

## Depo Yapısı
NoShow-AI/
├── gorseller/ # Grafikler ve şekiller
├── kodlar/ # Numaralı, sıralı pipeline script'leri (01–46)
├── modeller/ # Model üretim README'si (ağır .joblib dosyaları hariç)
├── veriler/ # Analiz sonuç dosyaları (CSV) — ham veri hariç (bkz. Veri Erişimi)
├── requirements.txt
└── README.md


## Kurulum

```bash
pip install -r requirements.txt
```

Başlıca bağımlılıklar: `scikit-learn`, `pandas`, `numpy`, `shap`, `lightgbm`, `catboost`, `xgboost`, `simpy`, `scipy`, `matplotlib`.

## Pipeline Çalıştırma Sırası

Ayrıntılı sıra için `kodlar/PIPELINE_SIRASI.md` dosyasına bakınız. Özetle:

1. **Keşifçi analiz (01–09):** Veri genel bakış, eksik veri, korelasyon, sızıntı riski.
2. **Ön işleme (10–16):** Bölme, tarih dönüşümü, imputation, öznitelik mühendisliği, kodlama.
3. **Model seçimi ve optimizasyon (25–31):** Adil karşılaştırma, hiperparametre optimizasyonu, şampiyon model ve dış test, pipeline paketi.
4. **Doğrulama ve analiz (32–40):** Hasta geçmişi öznitelikleri, altı-senaryo ızgarası, çöküş mekanizması, karar eğrisi, alt-grup adalet, simülasyon ikili raporlama.
5. **Duyarlılık ve düzeltme (41–46):** CV skorlarının standart sapması, takvim özniteliği ablasyonu, vekil kimlik türetme düzeltmesi ve çok-tohumlu kararlılık analizi.

## Ek Analizler

- **Karar eğrisi analizi (script 37):** Model, kabul edilebilir eşik aralığında "hepsine/hiçbirine müdahale" stratejilerinden daha yüksek klinik net fayda sağlar.
- **Alt-grup adalet (script 38):** Cinsiyet ve yaş grupları arası ROC-AUC farkı yalnızca 0,029; model gruplar arası tutarlıdır.
- **Takvim özniteliği ablasyonu (script 42):** Takvim özniteliklerinin çıkarılması satır-rastgele ve hasta-ayrık bölmede performansı belirgin biçimde düşürürken, kronolojik bölmede etkisi marjinaldir; bu, kronolojik senaryoda takvim sinyalinin zaten genellenemediğini doğrulamaktadır.
- **Vekil kimlik duyarlılık analizi (script 43–44):** Doğum tarihi eksik gözlemlerin tek bir ortak kimlik altında gruplanması, hasta-ayrık bölmede yapay bir kararsızlığa yol açtığı tespit edilmiş ve düzeltilmiştir; düzeltme sonrası çakışma oranı %2,11'den %0,09'a düşmüştür.
- **Simülasyon ikili raporlama (script 40):** Overbooking faydası hem iyimser (rastgele) hem gerçekçi (kronolojik) model altında, gerçek (gözlenmiş) test etiketleri referans ölçüt alınarak raporlanır; gerçekçi rejimdeki kazanç iyimserin yaklaşık dörtte biri kadardır.

## Veri Erişimi

Ham ve işlenmiş hasta verisi, gizlilik (veri koruma / etik kurul) ve boyut nedeniyle bu depoya dahil edilmemiştir. Depo yalnızca analiz sonuç dosyalarını (CSV) içerir. Ham veri, ilgili kurumun etik onayı çerçevesinde talep üzerine sağlanabilir; elde edildikten sonra ön işleme script'leri (10–16) ile işlenmiş dosyalar yeniden üretilebilir.

## Sınırlamalar

Çalışma tek bir merkeze ait veriye dayanmaktadır. Hasta geçmişi öznitelikleri, doğrudan hasta kimliği bulunmadığından doğum tarihi–cinsiyet–şehir tabanlı bir vekil kimlik üzerinden geçmişe dönük olarak türetilmiştir. Kronolojik değerlendirme tek bir bölünme noktasına dayanır ve çalışma prospektif bir dağıtımla henüz doğrulanmamıştır. Ayrıntılar için makale taslağının Sınırlamalar bölümüne bakınız.

## Yazarlar

- Aylin Cer
- Beyza Nur Dinçer

Yönetim Bilişim Sistemleri, Karadeniz Teknik Üniversitesi
