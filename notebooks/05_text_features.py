# TEXT FEATURES
"""
python notebooks/05_text_features.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def create_text_features(df_raw):
    df = df_raw[['id', 'name', 'description']].copy()
    df['name'] = df['name'].fillna('')
    df['description'] = df['description'].fillna('')
    desc_lower = df['description'].str.lower()
    name_lower = df['name'].str.lower()
    
    #uzunluk özellikleri
    df['name_len'] = df['name'].apply(len)
    df['desc_len'] = df['description'].apply(len)
    df['name_word_count'] = df['name'].apply(lambda x: len(str(x).split()))
    df['desc_word_count'] = df['description'].apply(lambda x: len(str(x).split()))
    
    # Premium keywordler
    df['has_view'] = desc_lower.str.contains('view').astype(int)
    df['has_luxury'] = desc_lower.str.contains('luxury|luxurious').astype(int)
    df['has_modern'] = desc_lower.str.contains('modern').astype(int)
    df['has_spacious'] = desc_lower.str.contains('spacious|large').astype(int)
    df['has_cozy'] = desc_lower.str.contains('cozy|cosy').astype(int)
    
    # istanbula özel olanlar
    df['has_bosphorus'] = desc_lower.str.contains('bosphorus|boğaz|bogaz').astype(int)
    df['has_sea'] = desc_lower.str.contains('sea view|sea-view|deniz').astype(int)
    df['has_historic'] = desc_lower.str.contains('historic|historical|tarihi').astype(int)
    df['has_terrace'] = desc_lower.str.contains('terrace|teras').astype(int)
    df['has_rooftop'] = desc_lower.str.contains('rooftop|roof top|çatı').astype(int)
    df['has_balcony_desc'] = desc_lower.str.contains('balcony|balkon').astype(int)
    
    # lokasyon
    df['has_center'] = desc_lower.str.contains('center|centre|merkez').astype(int)
    df['has_walking'] = desc_lower.str.contains('walking distance|yürüme').astype(int)
    df['has_metro'] = desc_lower.str.contains('metro|tram|tramvay').astype(int)
    
    #nameden özellikler
    df['name_has_luxury'] = name_lower.str.contains('luxury|deluxe|premium').astype(int)
    df['name_has_view'] = name_lower.str.contains('view|sea|bosphorus').astype(int)
    df['name_has_central'] = name_lower.str.contains('central|center|taksim|sultanahmet|besiktas|kadikoy').astype(int)
    
    # composite skorlar
    df['text_premium_score'] = (
        df['has_view'] + df['has_luxury'] + df['has_modern'] + 
        df['has_bosphorus'] + df['has_sea'] + df['has_terrace'] + 
        df['has_rooftop'] + df['name_has_luxury'] + df['name_has_view']
    )
    
    df['text_location_score'] = (
        df['has_center'] + df['has_walking'] + df['has_metro'] + 
        df['has_historic'] + df['name_has_central']
    )
    
    return df.drop(columns=['name', 'description'])


def add_text_to_data(train_proc, test_proc, train_text, test_text):
    text_cols = [col for col in train_proc.columns if col in train_text.columns and col != 'id']
    train_proc = train_proc.drop(columns=text_cols, errors='ignore')
    test_proc = test_proc.drop(columns=text_cols, errors='ignore')
    
    # merge
    train_merged = train_proc.merge(train_text, on='id', how='left')
    test_merged = test_proc.merge(test_text, on='id', how='left')
    
    for col in train_text.columns:
        if col != 'id':
            train_merged[col]= train_merged[col].fillna(0)
            test_merged[col] =test_merged[col].fillna(0)
    
    print(f"train shape: {train_proc.shape} --> {train_merged.shape}")
    print(f"test shape: {test_proc.shape} --> {test_merged.shape}")
    
    return train_merged, test_merged


def save_data(train, test, output_dir='data/processed'):
    train.to_csv(f'{output_dir}/train_processed.csv', index=False)
    test.to_csv(f'{output_dir}/test_processed.csv', index=False)


if __name__ == "__main__":
    train_raw = pd.read_csv('data/raw/train.csv')
    test_raw = pd.read_csv('data/raw/test.csv')
    
    train_proc = pd.read_csv('data/processed/train_processed.csv')
    test_proc = pd.read_csv('data/processed/test_processed.csv')

    train_text = create_text_features(train_raw)
    test_text = create_text_features(test_raw)
    train, test = add_text_to_data(train_proc, test_proc, train_text, test_text)

    save_data(train, test)
 
    print(f"\n Premium keyword dağılımı:")
    print(f"  has_bosphorus: {train['has_bosphorus'].sum():,}")
    print(f"  has_sea : {train['has_sea'].sum():,}")
    print(f"  has_terrace:{train['has_terrace'].sum():,}")
    print(f"  has_view: {train['has_view'].sum():,}")