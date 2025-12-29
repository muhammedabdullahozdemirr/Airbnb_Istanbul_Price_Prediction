# Airbnb Fiyat Tahmin Projesi - Tam Dokümantasyon

## Proje Özeti
**Amaç:** Airbnb listelerinin fiyatlarını tahmin etmek (RMSLE metriği)  
**En İyi CV Skoru:** 0.42477  
**Kullanılan Model:** XGBoost Regressor  

---

## Dosya Yapısı

### Notebooks Klasörü (Pipeline Sırası)

| # | Dosya | Açıklama | Durum |
|---|-------|----------|-------|
| 1 | `01_data_loading.py` | Ham veri yükleme | ✅ Aktif |
| 2 | `02_preprocessing.py` | Veri temizleme, tip dönüşümleri | ✅ Aktif |
| 3 | `03_feature_engineering.py` | Temel özellik oluşturma | ✅ Aktif |
| 4 | `04_data_analysis.py` | Keşifsel veri analizi (EDA) | ✅ Aktif |
| 5 | `05_feature_selection.py` | Korelasyon analizi + Popularity özellikleri | ✅ Aktif |
| 6 | `06_model_training.py` | Genel model eğitim fonksiyonları | ✅ Aktif |
| 7 | `07_run_xgboost.py` | **ANA MODEL - Final Submission** | ✅ **KULLANILIYOR** |
| 8 | `09_calendar_features.py` | Calendar.csv'den özellik çıkarma | ✅ Aktif |
| 9 | `11_text_features.py` | Text Mining özellikleri | ✅ Aktif |

### Data Klasörü

| Dosya | Açıklama |
|-------|----------|
| `data/raw/train.csv` | Ham eğitim verisi |
| `data/raw/test.csv` | Ham test verisi |
| `data/raw/calendar.csv` | Müsaitlik takvimi (10.8M satır) |
| `data/raw/reviews.csv` | Yorumlar |
| `data/processed/train_processed.csv` | İşlenmiş eğitim verisi (59 özellik) |
| `data/processed/test_processed.csv` | İşlenmiş test verisi |
| `data/processed/submission.csv` | **Final tahminler** |

---

## Özellik Mühendisliği Detayları

### 1. Temel Özellikler (03_feature_engineering.py)

| Özellik | Formül | Açıklama |
|---------|--------|----------|
| `host_experience_days` | `reference_date - host_since` | Ev sahibinin deneyimi (gün) |
| `bathrooms_clean` | Regex ile `bathrooms_text`'ten çıkarıldı | Temiz banyo sayısı |
| `amenities_count` | Amenities string'inden sayım | Toplam olanak sayısı |
| `has_wifi`, `has_kitchen`, `has_ac` | Keyword arama | Binary özellikler |

### 2. Kapasite Kombinasyonları (07_run_xgboost.py)

| Özellik | Formül | Önem |
|---------|--------|------|
| `accommodates_per_bedroom` | `accommodates / (bedrooms + 1)` | **#2** (%8.5) |
| `accommodates_per_bathroom` | `accommodates / (bathrooms + 1)` | Yüksek |
| `total_rooms` | `bedrooms + bathrooms` | **#3** (%7.2) |
| `space_efficiency` | `accommodates / (total_rooms + 1)` | Orta |

**Neden:** Kişi başına düşen oda/banyo sayısı fiyatı doğrudan etkiler. Büyük evler daha pahalıdır.

### 3. Review Composite (07_run_xgboost.py)

| Özellik | Formül | Açıklama |
|---------|--------|----------|
| `review_composite` | `mean(all 7 review scores)` | Tüm puanların ortalaması |
| `review_variance` | `std(all 7 review scores)` | Puan tutarsızlığı |

**Neden:** 7 ayrı review skoru yerine tek bir bileşik skor kullanmak multicollinearity'yi azaltır ve modelin genelleme yeteneğini artırır.

### 4. Host Quality Score (07_run_xgboost.py)

```python
host_quality = (
    host_is_superhost * 3 +
    host_identity_verified * 2 +
    host_has_profile_pic * 1 +
    (host_response_rate / 100) +
    (host_acceptance_rate / 100)
)
```

**Neden:** Ev sahibinin güvenilirliği fiyatı etkiler. Superhost'lar genellikle daha yüksek fiyat talep edebilir.

### 5. Calendar Özellikleri (09_calendar_features.py)

10.8 milyon satırlık calendar.csv'den çıkarılan özellikler:

| Özellik | Formül | Açıklama |
|---------|--------|----------|
| `availability_rate` | `mean(is_available)` | Yıllık müsaitlik oranı |
| `booking_rate` | `1 - availability_rate` | Doluluk oranı |
| `avg_min_nights` | `mean(minimum_nights)` | Ortalama minimum gece |
| `max_nights_std` | `std(maximum_nights)` | Gece limiti varyasyonu |
| `nights_flexibility` | `avg_max - avg_min` | Rezervasyon esnekliği |

**Neden:** Yüksek doluluk oranı = popüler listing = daha yüksek fiyat talep edilebilir.

### 6. Popularity Özellikleri (05_feature_selection.py)

reviews.csv'den çıkarılan özellikler:

| Özellik | Formül | Açıklama |
|---------|--------|----------|
| `total_reviews` | `count(reviews)` | Toplam yorum sayısı |
| `total_reviews_log` | `log1p(total_reviews)` | Log transformasyonu |
| `popularity_score` | `MinMaxScaler(total_reviews_log)` | Normalize popülerlik (0-1) |
| `popularity_level` | `qcut(total_reviews, 3)` | Kategorik: low/medium/high |

**Neden:** Çok yorum alan evler daha popülerdir ve genellikle daha yüksek fiyatlıdır.

### 7. Text Mining Özellikleri (11_text_features.py) 🆕

Ham veri içindeki `name` ve `description` kolonlarından:

| Özellik | Formül | Önem |
|---------|--------|------|
| `name_len` | `len(name)` | İsim uzunluğu |
| `desc_len` | `len(description)` | Açıklama uzunluğu |
| `name_word_count` | `len(name.split())` | İsim kelime sayısı |
| `desc_word_count` | `len(description.split())` | Açıklama kelime sayısı |
| `has_view` | `'view' in description` | **#9** (%2.3) |
| `has_luxury` | `'luxury' in description` | **#10** (%2.2) |
| `has_heart` | `'heart' in description` | Merkezi konum göstergesi |

**Neden:** 
- "View" (manzara) kelimesi genellikle daha pahalı evlerde geçer
- "Luxury" (lüks) kelimesi premium fiyatlandırma göstergesidir
- Uzun açıklamalar profesyonel ev sahiplerini gösterir

---

## Model Yapılandırması

### XGBoost Hiperparametreleri (07_run_xgboost.py)

```python
xgb_params = {
    'n_estimators': 1200,      # Ağaç sayısı (daha fazla = daha hassas)
    'max_depth': 6,            # Ağaç derinliği
    'learning_rate': 0.015,    # Öğrenme hızı (düşük = daha yavaş ama hassas)
    'min_child_weight': 4,     # Minimum yaprak ağırlığı
    'subsample': 0.8,          # Satır örnekleme
    'colsample_bytree': 0.75,  # Sütun örnekleme
    'reg_alpha': 0.3,          # L1 regularizasyon
    'reg_lambda': 1.5,         # L2 regularizasyon
    'gamma': 0.1               # Minimum loss azaltma
}
```

### Outlier Handling

```python
Q1 = train['price'].quantile(0.01)  # Alt %1
Q3 = train['price'].quantile(0.99)  # Üst %1
train_clean = train[(train['price'] >= Q1) & (train['price'] <= Q3)]
```

**Neden:** Aşırı değerler (çok ucuz veya çok pahalı evler) modeli yanıltır. %1-%99 aralığı en stabil sonuçları verdi.

### Target Transformation

```python
y_log = np.log1p(y)  # Log transformation
```

**Neden:** Fiyat dağılımı sağa çarpık (skewed). Log transformasyonu dağılımı normale yaklaştırır ve RMSLE metriğiyle uyumludur.

---

## Denenen Ama Başarısız Olan Yöntemler

### 1. Target Encoding (10_ultimate_model.py - SİLİNDİ)
**Sorun:** K-Fold ile data leakage önlemeye çalıştık ama public score kötüleşti.
**Sonuç:** Overfitting yarattı, validation score'dan daha kötü performans.

### 2. Advanced Mining Features (run_models_mining.py - SİLİNDİ)
**Denenen:** K-Means clustering, hotspot distance
**Sorun:** CV'de iyi görünüyordu ama public score'da fayda sağlamadı.

### 3. Random Forest ile Ensemble
**Denenen:** XGBoost + RF stacking
**Sonuç:** RF performansı düşük olduğu için ensemble fayda sağlamadı.

---

## Skor Geçmişi

| Aşama | CV RMSE | Public Score | Notlar |
|-------|---------|--------------|--------|
| Başlangıç | ~0.56 | - | Temel model |
| Optimized XGBoost | 0.43 | 0.52 | Hiperparametre ayarı |
| + Calendar Features | 0.429 | - | Availability rate |
| + Text Features | **0.42477** | - | has_view, has_luxury |

---

## Çalıştırma Sırası

Projeyi sıfırdan çalıştırmak için:

```powershell
cd c:\Users\NURETTİN\OneDrive\Masaüstü\YZV311_2526_10

# 1. Calendar özelliklerini ekle
python notebooks/09_calendar_features.py

# 2. Text özelliklerini ekle
python notebooks/11_text_features.py

# 3. Final modeli çalıştır ve submission oluştur
python notebooks/07_run_xgboost.py
```

---

## Sonuç

Bu proje boyunca:
- **59 özellik** oluşturuldu
- **5 farklı veri kaynağı** kullanıldı (train, test, calendar, reviews, text)
- **10+ farklı yöntem** denendi
- En iyi sonuç **Text Mining + Fine-tuned XGBoost** kombinasyonuyla elde edildi

**Final CV RMSE:** 0.42477
