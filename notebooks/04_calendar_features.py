# CALENDAR FEATURES
"""
python notebooks/04_calendar_features.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def load_calendar(path='data/raw/calendar.csv'):
    calendar= pd.read_csv(path)
    print(f"calendar shape : {calendar.shape}")
    print(f"unique listings: {calendar['listing_id'].nunique()}")
    return calendar


def create_calendar_features(calendar):
     # available ı booleana çevirelim
    calendar['is_available'] = (calendar['available'] == 't').astype(int)
    
    #her listing için aggregation
    calendar_features = calendar.groupby('listing_id').agg({
        'is_available': ['mean', 'sum'],
        'minimum_nights': ['mean', 'std'],
        'maximum_nights': ['mean', 'std']
    }).reset_index()
    
    calendar_features.columns = [
        'listing_id',
        'availability_rate',      #müsaitlik oranı (0-1)
        'total_available_days',   #toplam müsait gün
        'avg_min_nights',         #ortalama min gece
        'min_nights_std',         # min gece varyasyonu
        'avg_max_nights',         #ortalama max gece
        'max_nights_std'          # max gece varyasyonu
    ]
    
    calendar_features['booking_rate'] = 1 - calendar_features['availability_rate']
    calendar_features['nights_flexibility'] = calendar_features['avg_max_nights'] - calendar_features['avg_min_nights']
    
    calendar_features = calendar_features.fillna(0)
    
    print(f"calendar features shape : {calendar_features.shape}")
    return calendar_features


def add_calendar_to_data(train, test, calendar_features):
    cal_cols = ['availability_rate', 'total_available_days', 'avg_min_nights', 'min_nights_std', 'avg_max_nights', 'max_nights_std', 'booking_rate', 'nights_flexibility']
    
    # var olan calendar columnlarını sil
    for col in cal_cols:
        if col in train.columns:
            train = train.drop(col, axis=1)
        if col in test.columns:
            test = test.drop(col, axis=1)
    
    # merge
    train_merged= train.merge(calendar_features, left_on='id',right_on='listing_id', how='left')
    train_merged =train_merged.drop('listing_id', axis=1, errors='ignore')
    
    test_merged= test.merge(calendar_features, left_on='id',right_on='listing_id', how='left')
    test_merged =test_merged.drop('listing_id', axis=1, errors='ignore')
    
    #medianla dolduruyoz
    for col in cal_cols:
        if col in train_merged.columns:
            train_merged[col] = train_merged[col].fillna(train_merged[col].median())
        if col in test_merged.columns:
            test_merged[col] = test_merged[col].fillna(test_merged[col].median())
    
    print(f"train shape: {train.shape} --> {train_merged.shape}")
    print(f"test shape: {test.shape} --> {test_merged.shape}")
    
    return train_merged, test_merged


def save_data(train, test, output_dir='data/processed'):
    train.to_csv(f'{output_dir}/train_processed.csv', index=False)
    test.to_csv(f'{output_dir}/test_processed.csv', index=False)


if __name__ == "__main__":
    calendar = load_calendar()
    calendar_features = create_calendar_features(calendar)
  
    train = pd.read_csv('data/processed/train_processed.csv')
    test = pd.read_csv('data/processed/test_processed.csv')
    print(f"\ntrain shape before :{train.shape}")
    print(f"test shape before :{test.shape}")
    
    
    train, test = add_calendar_to_data(train, test, calendar_features)
    save_data(train, test)
