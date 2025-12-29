# FEATURE ENGINEERING
"""
Kullanim:
    from step_03_feature_engineering import create_features
    train, test = create_features(train, test, raw_train, raw_test)
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime


def extract_amenities_count(amenity_str):
    if pd.isna(amenity_str):
        return 0
    return len(re.findall(r'"([^"]*)"', str(amenity_str)))


def has_amenity(amenity_str, keyword):
    if pd.isna(amenity_str):
        return 0
    return 1 if keyword.lower() in str(amenity_str).lower() else 0


def extract_bathrooms(bathroom_text):
    if pd.isna(bathroom_text):
        return 0
    match = re.search(r'(\d+\.?\d*)', str(bathroom_text))
    return float(match.group(1)) if match else 0


def haversine_distance(lat1, lon1, lat2, lon2):
    """İki koordinat arası mesafe (km)"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def create_location_features(df):
    """İstanbul'un önemli noktalarına uzaklık"""
    
    SULTANAHMET = (41.0082, 28.9784)
    TAKSIM = (41.0370, 28.9850)
    BESIKTAS = (41.0422, 29.0067)
    
    df['dist_to_sultanahmet'] = haversine_distance(
        df['latitude'], df['longitude'], SULTANAHMET[0], SULTANAHMET[1])
    
    df['dist_to_taksim'] = haversine_distance(
        df['latitude'], df['longitude'], TAKSIM[0], TAKSIM[1])
    
    df['dist_to_bosphorus'] = haversine_distance(
        df['latitude'], df['longitude'], BESIKTAS[0], BESIKTAS[1])
    
    df['dist_to_center_min'] = df[['dist_to_sultanahmet', 'dist_to_taksim']].min(axis=1)
    
    df['is_european_side'] = (df['longitude'] < 29.0).astype(int)
    
    return df


def create_amenity_features(df, raw_df):
    """Amenities'den feature çıkar"""
    
    amenities = raw_df['amenities']
    
    df['amenities_count'] = amenities.apply(extract_amenities_count)
    df['has_wifi'] = amenities.apply(lambda x: has_amenity(x, 'wifi'))
    df['has_kitchen'] = amenities.apply(lambda x: has_amenity(x, 'kitchen'))
    df['has_ac'] = amenities.apply(lambda x: has_amenity(x, 'air conditioning'))
    df['has_pool'] = amenities.apply(lambda x: has_amenity(x, 'pool'))
    df['has_parking'] = amenities.apply(lambda x: has_amenity(x, 'parking'))
    df['has_balcony'] = amenities.apply(lambda x: has_amenity(x, 'balcony'))
    df['has_washer'] = amenities.apply(lambda x: has_amenity(x, 'washer'))
    df['has_tv'] = amenities.apply(lambda x: has_amenity(x, 'tv'))
    df['has_heating'] = amenities.apply(lambda x: has_amenity(x, 'heating'))
    df['has_elevator'] = amenities.apply(lambda x: has_amenity(x, 'elevator'))
    df['has_gym'] = amenities.apply(lambda x: has_amenity(x, 'gym'))
    
    df['premium_amenities'] = (
        df['has_pool'] * 3 +
        df['has_gym'] * 2 +
        df['has_parking'] * 2 +
        df['has_balcony'] * 1 +
        df['has_elevator'] * 1
    )
    
    df['basic_amenities'] = (
        df['has_wifi'] + df['has_kitchen'] + df['has_ac'] +
        df['has_heating'] + df['has_washer'] + df['has_tv']
    )
    
    return df


def create_features(train, test, raw_train=None, raw_test=None):
    train = train.copy()
    test = test.copy()
    
    print("Creating features...")
    
    # Host Experience
    reference_date = datetime(2025, 1, 1)
    for df in [train, test]:
        if 'host_since' in df.columns:
            df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
            df['host_experience_days'] = (reference_date - df['host_since']).dt.days
            df['host_experience_days'] = df['host_experience_days'].fillna(0)
    
    # Price per person
    if 'price' in train.columns and 'accommodates' in train.columns:
        train['price_per_person'] = train['price'] / train['accommodates'].clip(lower=1)
    
    # Location Features
    print("  - Location features...")
    train = create_location_features(train)
    test = create_location_features(test)
    
    # Raw data'dan feature çıkarma
    if raw_train is not None:
        print("  - Amenity features...")
        
        raw_features_train = pd.DataFrame({'id': raw_train['id']})
        raw_features_test = pd.DataFrame({'id': raw_test['id']}) if raw_test is not None else None
        
        if 'bathrooms_text' in raw_train.columns:
            raw_features_train['bathrooms_clean'] = raw_train['bathrooms_text'].apply(extract_bathrooms)
            if raw_test is not None and 'bathrooms_text' in raw_test.columns:
                raw_features_test['bathrooms_clean'] = raw_test['bathrooms_text'].apply(extract_bathrooms)
        
        if 'amenities' in raw_train.columns:
            temp_train = pd.DataFrame()
            temp_train = create_amenity_features(temp_train, raw_train)
            for col in temp_train.columns:
                raw_features_train[col] = temp_train[col].values
            
            if raw_test is not None and 'amenities' in raw_test.columns:
                temp_test = pd.DataFrame()
                temp_test = create_amenity_features(temp_test, raw_test)
                for col in temp_test.columns:
                    raw_features_test[col] = temp_test[col].values
        
        train = train.merge(raw_features_train, on='id', how='left')
        if raw_test is not None and raw_features_test is not None:
            test = test.merge(raw_features_test, on='id', how='left')
        
        amenity_cols = ['bathrooms_clean', 'amenities_count', 'has_wifi', 'has_kitchen', 'has_ac',
                        'has_pool', 'has_parking', 'has_balcony', 'has_washer', 'has_tv',
                        'has_heating', 'has_elevator', 'has_gym', 'premium_amenities', 'basic_amenities']
        for col in amenity_cols:
            if col in train.columns:
                train[col] = train[col].fillna(0)
            if col in test.columns:
                test[col] = test[col].fillna(0)
    
    # Categorical Encoding
    categorical_cols = ['neighbourhood_cleansed', 'property_type', 'room_type', 'host_response_time']
    for col in categorical_cols:
        if col in train.columns:
            unique_vals = train[col].unique()
            val_to_num = {val: i for i, val in enumerate(unique_vals)}
            train[col] = train[col].map(val_to_num)
            if col in test.columns:
                test[col] = test[col].map(val_to_num).fillna(-1).astype(int)
    
    # Drop unnecessary columns
    cols_to_drop = ['host_since', 'bathrooms_text', 'first_review', 'last_review']
    train = train.drop(columns=[c for c in cols_to_drop if c in train.columns], errors='ignore')
    test = test.drop(columns=[c for c in cols_to_drop if c in test.columns], errors='ignore')
    
    print(f"Train shape after FE: {train.shape}")
    print(f"Test shape after FE: {test.shape}")
    
    return train, test


def save_engineered_data(train, test, output_dir='data/processed'):
    train.to_csv(f'{output_dir}/train_processed.csv', index=False)
    test.to_csv(f'{output_dir}/test_processed.csv', index=False)
    print(f"Engineered data saved to {output_dir}/")


if __name__ == "__main__":
    train = pd.read_csv('data/processed/train_preprocessed.csv')
    test = pd.read_csv('data/processed/test_preprocessed.csv')
    
    raw_train = pd.read_csv('data/raw/train.csv')
    raw_test = pd.read_csv('data/raw/test.csv')
    
    train, test = create_features(train, test, raw_train, raw_test)
    save_engineered_data(train, test)