"""
02 - Data Preprocessing
Handles data cleaning, type conversions, and missing value imputation.
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(train, test):
    """
    Preprocess train and test datasets.
    - Drop unnecessary columns
    - Fix data types
    - Handle missing values
    - Create basic features
    """
    train = train.copy()
    test = test.copy()
    
    # =========================================================================
    # 1. Drop unnecessary columns
    # =========================================================================
    print("Dropping unnecessary columns...")
    
    cols_to_drop = [
        # High missing values (>60%)
        'neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated',
        # Useless identifiers
        'listing_url', 'scrape_id', 'last_scraped', 'source',
        # Text columns
        'description', 'name', 'neighborhood_overview', 'picture_url',
        'host_url', 'host_thumbnail_url', 'host_picture_url', 'host_about',
        'host_neighbourhood', 'host_verifications', 'amenities',
        # Other
        'license', 'host_name', 'host_location'
    ]
    
    train = train.drop(columns=[c for c in cols_to_drop if c in train.columns], errors='ignore')
    test = test.drop(columns=[c for c in cols_to_drop if c in test.columns], errors='ignore')
    
    print(f"Train shape after drop: {train.shape}")
    print(f"Test shape after drop: {test.shape}")
    
    # =========================================================================
    # 2. Fix Data Types
    # =========================================================================
    print("\nFixing data types...")
    
    # Price (remove comma and convert to float)
    if 'price' in train.columns:
        train['price'] = train['price'].str.replace(',', '').astype(float)
    
    # Percentage columns
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in train.columns:
            train[col] = train[col].str.rstrip('%').astype(float)
        if col in test.columns:
            test[col] = test[col].str.rstrip('%').astype(float)
    
    # Boolean columns (t/f -> 1/0)
    bool_columns = ['host_is_superhost', 'host_has_profile_pic', 
                    'host_identity_verified', 'instant_bookable']
    
    for col in bool_columns:
        if col in train.columns:
            train[col] = train[col].map({'t': 1, 'f': 0})
        if col in test.columns:
            test[col] = test[col].map({'t': 1, 'f': 0})
    
    # =========================================================================
    # 3. Handle Missing Values
    # =========================================================================
    print("\nHandling missing values...")
    
    # Drop rows with missing price in train
    if 'price' in train.columns:
        train = train.dropna(subset=['price'])
    
    # Numeric columns - fill with median
    numeric_cols = ['host_response_rate', 'host_acceptance_rate', 'host_listings_count',
                    'host_total_listings_count', 'bedrooms', 'beds',
                    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                    'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                    'review_scores_value', 'reviews_per_month']
    
    for col in numeric_cols:
        if col in train.columns:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            if col in test.columns:
                test[col] = test[col].fillna(median_val)
    
    # Boolean columns - fill with mode
    for col in bool_columns:
        if col in train.columns:
            mode_val = train[col].mode()[0] if len(train[col].mode()) > 0 else 0
            train[col] = train[col].fillna(mode_val)
            if col in test.columns:
                test[col] = test[col].fillna(mode_val)
    
    # Remaining missing values - fill with 0 or 'Unknown'
    for col in train.columns:
        if train[col].isnull().sum() > 0:
            if train[col].dtype in ['float64', 'int64']:
                train[col] = train[col].fillna(0)
            else:
                train[col] = train[col].fillna('Unknown')
    
    for col in test.columns:
        if test[col].isnull().sum() > 0:
            if test[col].dtype in ['float64', 'int64']:
                test[col] = test[col].fillna(0)
            else:
                test[col] = test[col].fillna('Unknown')
    
    print(f"Train missing values: {train.isnull().sum().sum()}")
    print(f"Test missing values: {test.isnull().sum().sum()}")
    
    return train, test


def save_processed_data(train, test, output_dir='data/processed'):
    """Save processed datasets to CSV."""
    train.to_csv(f'{output_dir}/train_preprocessed.csv', index=False)
    test.to_csv(f'{output_dir}/test_preprocessed.csv', index=False)
    print(f"\nData saved to {output_dir}/")


if __name__ == "__main__":
    from _01_data_loading import load_data
    
    train, test = load_data()
    train, test = preprocess_data(train, test)
    save_processed_data(train, test)
