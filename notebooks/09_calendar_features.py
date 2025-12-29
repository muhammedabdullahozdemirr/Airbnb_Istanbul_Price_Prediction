"""
09 - Calendar Features
Extracts availability and booking features from calendar.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CALENDAR FEATURES EXTRACTION")
print("="*60)

# =============================================================================
# 1. Calendar Verisini Yukle
# =============================================================================
print("\nCalendar verisi yukleniyor (bu birkaç dakika surebilir)...")
calendar = pd.read_csv('data/raw/calendar.csv')

print(f"Calendar shape: {calendar.shape}")
print(f"Unique listings: {calendar['listing_id'].nunique()}")

# =============================================================================
# 2. Calendar Features Hesapla
# =============================================================================
print("\nCalendar ozellikleri hesaplaniyor...")

# Available kolonunu boolean'a cevir
calendar['is_available'] = (calendar['available'] == 't').astype(int)

# Her listing icin aggregation
calendar_features = calendar.groupby('listing_id').agg({
    'is_available': ['mean', 'sum'],           # availability rate & total available days
    'minimum_nights': ['mean', 'std'],         # avg & variation
    'maximum_nights': ['mean', 'std']
}).reset_index()

# Kolon isimlerini duzelt
calendar_features.columns = [
    'listing_id',
    'availability_rate',      # Musaitlik orani (0-1)
    'total_available_days',   # Toplam musait gun sayisi
    'avg_min_nights',         # Ortalama minimum gece
    'min_nights_std',         # Minimum gece varyasyonu
    'avg_max_nights',         # Ortalama maximum gece  
    'max_nights_std'          # Maximum gece varyasyonu
]

# Booking rate (doluluk orani)
calendar_features['booking_rate'] = 1 - calendar_features['availability_rate']

# Nights flexibility (esneklik)
calendar_features['nights_flexibility'] = calendar_features['avg_max_nights'] - calendar_features['avg_min_nights']

# NaN degerlerini doldur
calendar_features = calendar_features.fillna(0)

print(f"Calendar features shape: {calendar_features.shape}")
print("\nFeature istatistikleri:")
print(calendar_features[['availability_rate', 'booking_rate', 'nights_flexibility']].describe())

# =============================================================================
# 3. Train ve Test Verilerine Ekle
# =============================================================================
print("\nTrain ve Test verilerine ekleniyor...")

train = pd.read_csv('data/processed/train_processed.csv')
test = pd.read_csv('data/processed/test_processed.csv')

print(f"Train shape before: {train.shape}")
print(f"Test shape before: {test.shape}")

# Mevcut calendar kolonlarini sil (varsa)
cal_cols = ['availability_rate', 'total_available_days', 'avg_min_nights', 
            'min_nights_std', 'avg_max_nights', 'max_nights_std', 
            'booking_rate', 'nights_flexibility']

for col in cal_cols:
    if col in train.columns:
        train = train.drop(col, axis=1)
    if col in test.columns:
        test = test.drop(col, axis=1)

# Merge
train_merged = train.merge(calendar_features, left_on='id', right_on='listing_id', how='left')
train_merged = train_merged.drop('listing_id', axis=1, errors='ignore')

test_merged = test.merge(calendar_features, left_on='id', right_on='listing_id', how='left')
test_merged = test_merged.drop('listing_id', axis=1, errors='ignore')

# NaN doldur
for col in cal_cols:
    if col in train_merged.columns:
        train_merged[col] = train_merged[col].fillna(train_merged[col].median())
    if col in test_merged.columns:
        test_merged[col] = test_merged[col].fillna(test_merged[col].median())

print(f"\nTrain shape after: {train_merged.shape}")
print(f"Test shape after: {test_merged.shape}")

# Eslesen listing sayisi
train_matched = (train_merged['availability_rate'] > 0).sum()
test_matched = (test_merged['availability_rate'] > 0).sum()
print(f"\nTrain listings with calendar data: {train_matched:,} / {len(train_merged):,}")
print(f"Test listings with calendar data: {test_matched:,} / {len(test_merged):,}")

# =============================================================================
# 4. Kaydet
# =============================================================================
print("\nKaydediliyor...")
train_merged.to_csv('data/processed/train_processed.csv', index=False)
test_merged.to_csv('data/processed/test_processed.csv', index=False)

print("\nTamamlandi!")
print(f"Yeni kolonlar: {cal_cols}")
