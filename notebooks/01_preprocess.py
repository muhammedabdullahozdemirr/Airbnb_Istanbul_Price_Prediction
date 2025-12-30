#PREPROCESSING
"""
python notebooks/01_preprocess.py

"""

import pandas as pd
import numpy as np

        # Silinecekler
COLS_TO_DROP=[
    # Çok eksik olanlar (edada tespit etmiştik 60 threshold)
    'neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated',
    # url, id filan
    'listing_url', 'scrape_id', 'last_scraped', 'source',
    # Text kolonları (ayrı işlenecek)
    'description', 'name', 'neighborhood_overview', 'picture_url',
    'host_url', 'host_thumbnail_url', 'host_picture_url', 'host_about',
    'host_neighbourhood', 'host_verifications', 'amenities',
    # diğer silincekelr
    'license', 'host_name', 'host_location', 'neighbourhood',
    'Unnamed: 0', 'Unnamed: 0.1'
]

NUMERIC_COLS = [
    'host_response_rate', 'host_acceptance_rate', 
    'host_listings_count', 'host_total_listings_count',
    'bedrooms', 'beds', 'reviews_per_month',
    'review_scores_rating', 'review_scores_accuracy', 
    'review_scores_cleanliness', 'review_scores_checkin', 
    'review_scores_communication', 'review_scores_location',
    'review_scores_value'
]

BOOL_COLS = [
    'host_is_superhost', 'host_has_profile_pic', 
    'host_identity_verified', 'instant_bookable'
]


def preprocess_data(train, test):       # ana preprocess functionu bu
    train = train.copy()
    test = test.copy()
    
    # Gereksiz kolonları dropla
    train = train.drop(columns=[c for c in COLS_TO_DROP if c in train.columns], errors='ignore')
    test = test.drop(columns=[c for c in COLS_TO_DROP if c in test.columns], errors='ignore')
    
    #price temizleme
    if 'price' in train.columns:
        train['price'] = train['price'].str.replace(',', '').astype(float)
    
     #yüzde işaretlerini kaldırma
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in train.columns:
            train[col] = train[col].str.rstrip('%').astype(float)
        if col in test.columns:
            test[col] = test[col].str.rstrip('%').astype(float)
    
    # t yi 1 f yi 0 yapma
    for col in BOOL_COLS:
        if col in train.columns:
            train[col] = train[col].map({'t': 1, 'f': 0})
        if col in test.columns:
            test[col] = test[col].map({'t': 1, 'f': 0})
    
    # price eksikse sil
    if 'price' in train.columns:
        train = train.dropna(subset=['price'])
    
    # numeric kolonları doldurdum (median)
    for col in NUMERIC_COLS:
        if col in train.columns:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            if col in test.columns:
                test[col] = test[col].fillna(median_val)
    
    # Boolean ları mode ile 
    for col in BOOL_COLS:
        if col in train.columns:
            mode_val = train[col].mode()[0] if len(train[col].mode()) > 0 else 0
            train[col] = train[col].fillna(mode_val)
            if col in test.columns:
                test[col] = test[col].fillna(mode_val)
    
    # kalan eksikleri de doldurdum
    for df in [train, test]:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna('Unknown')
    
    print(f"Train: {train.shape}")
    print(f"Test: {test.shape}")
    
    return train, test


def save_processed(train, test, output_dir='data/processed'):
    train.to_csv(f'{output_dir}/train_preprocessed.csv', index=False)
    test.to_csv(f'{output_dir}/test_preprocessed.csv', index=False)


if __name__ == "__main__":
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    print(f"Train: {train.shape}")
    print(f"Test: {test.shape}")
   
    
    train, test = preprocess_data(train, test)
    save_processed(train, test)