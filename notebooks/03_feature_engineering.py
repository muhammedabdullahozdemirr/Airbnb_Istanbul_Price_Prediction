"""
03 - Feature Engineering
Creates new features from existing columns.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def extract_amenities_count(amenity_str):
    """Count number of amenities from amenities string."""
    if pd.isna(amenity_str):
        return 0
    return len(re.findall(r'"([^"]*)"', str(amenity_str)))

def has_amenity(amenity_str, keyword):
    """Check if amenity string contains a specific keyword."""
    if pd.isna(amenity_str):
        return 0
    return 1 if keyword.lower() in str(amenity_str).lower() else 0

def extract_bathrooms(bathroom_text):
    """Extract bathroom count from bathroom_text."""
    if pd.isna(bathroom_text):
        return 0
    match = re.search(r'(\d+\.?\d*)', str(bathroom_text))
    return float(match.group(1)) if match else 0

def create_features(train, test, raw_train=None, raw_test=None):
    """
    Create new features for train and test datasets.
    """
    train = train.copy()
    test = test.copy()
    
    print("Creating features...")
    
    # =========================================================================
    # 1. Host Experience (days since host_since)
    # =========================================================================
    reference_date = datetime(2025, 1, 1)
    
    for df in [train, test]:
        if 'host_since' in df.columns:
            df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
            df['host_experience_days'] = (reference_date - df['host_since']).dt.days
            df['host_experience_days'] = df['host_experience_days'].fillna(0)
    
    # =========================================================================
    # 2. Price per person
    # =========================================================================
    if 'price' in train.columns and 'accommodates' in train.columns:
        train['price_per_person'] = train['price'] / train['accommodates'].clip(lower=1)
    
    # =========================================================================
    # 3. Bathroom cleaning
    # =========================================================================
    if raw_train is not None and 'bathrooms_text' in raw_train.columns:
        train['bathrooms_clean'] = raw_train['bathrooms_text'].apply(extract_bathrooms)
        if raw_test is not None and 'bathrooms_text' in raw_test.columns:
            test['bathrooms_clean'] = raw_test['bathrooms_text'].apply(extract_bathrooms)
    elif 'bathrooms_text' in train.columns:
        train['bathrooms_clean'] = train['bathrooms_text'].apply(extract_bathrooms)
        if 'bathrooms_text' in test.columns:
            test['bathrooms_clean'] = test['bathrooms_text'].apply(extract_bathrooms)
    
    # =========================================================================
    # 4. Amenities features
    # =========================================================================
    if raw_train is not None and 'amenities' in raw_train.columns:
        train['amenities_count'] = raw_train['amenities'].apply(extract_amenities_count)
        train['has_wifi'] = raw_train['amenities'].apply(lambda x: has_amenity(x, 'wifi'))
        train['has_kitchen'] = raw_train['amenities'].apply(lambda x: has_amenity(x, 'kitchen'))
        train['has_ac'] = raw_train['amenities'].apply(lambda x: has_amenity(x, 'air conditioning'))
        
        if raw_test is not None and 'amenities' in raw_test.columns:
            test['amenities_count'] = raw_test['amenities'].apply(extract_amenities_count)
            test['has_wifi'] = raw_test['amenities'].apply(lambda x: has_amenity(x, 'wifi'))
            test['has_kitchen'] = raw_test['amenities'].apply(lambda x: has_amenity(x, 'kitchen'))
            test['has_ac'] = raw_test['amenities'].apply(lambda x: has_amenity(x, 'air conditioning'))
    
    # =========================================================================
    # 5. Categorical Encoding
    # =========================================================================
    categorical_cols = ['neighbourhood_cleansed', 'property_type', 'room_type', 
                        'host_response_time']
    
    for col in categorical_cols:
        if col in train.columns:
            # Label encoding
            unique_vals = train[col].unique()
            val_to_num = {val: i for i, val in enumerate(unique_vals)}
            train[col] = train[col].map(val_to_num)
            
            if col in test.columns:
                test[col] = test[col].map(val_to_num).fillna(-1).astype(int)
    
    # =========================================================================
    # 6. Drop unnecessary columns after feature creation
    # =========================================================================
    cols_to_drop = ['host_since', 'bathrooms_text', 'first_review', 'last_review']
    train = train.drop(columns=[c for c in cols_to_drop if c in train.columns], errors='ignore')
    test = test.drop(columns=[c for c in cols_to_drop if c in test.columns], errors='ignore')
    
    print(f"Train shape after FE: {train.shape}")
    print(f"Test shape after FE: {test.shape}")
    
    return train, test


def save_engineered_data(train, test, output_dir='data/processed'):
    """Save feature-engineered datasets to CSV."""
    train.to_csv(f'{output_dir}/train_processed.csv', index=False)
    test.to_csv(f'{output_dir}/test_processed.csv', index=False)
    print(f"\nEngineered data saved to {output_dir}/")


if __name__ == "__main__":
    # Load preprocessed data
    train = pd.read_csv('data/processed/train_preprocessed.csv')
    test = pd.read_csv('data/processed/test_preprocessed.csv')
    
    # Load raw data for amenities
    raw_train = pd.read_csv('data/raw/train.csv')
    raw_test = pd.read_csv('data/raw/test.csv')
    
    train, test = create_features(train, test, raw_train, raw_test)
    save_engineered_data(train, test)
