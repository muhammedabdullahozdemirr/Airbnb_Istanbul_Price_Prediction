# TEXT FEATURES
"""
Name ve description'dan text ozellikleri cikarir
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TEXT FEATURES EXTRACTION")
print("="*60)

# 1. Verileri yukle
print("Veriler yukleniyor...")
train_raw = pd.read_csv('data/raw/train.csv')
test_raw = pd.read_csv('data/raw/test.csv')

train_proc = pd.read_csv('data/processed/train_processed.csv')
test_proc = pd.read_csv('data/processed/test_processed.csv')

# 2. Text ozelliklerini hesapla
def create_text_features(df_raw):
    df = df_raw[['id', 'name', 'description']].copy()
    
    df['name'] = df['name'].fillna('')
    df['description'] = df['description'].fillna('')
    
    # Lowercase for searching
    desc_lower = df['description'].str.lower()
    name_lower = df['name'].str.lower()
    
    # Uzunluk ozellikleri
    df['name_len'] = df['name'].apply(len)
    df['desc_len'] = df['description'].apply(len)
    df['name_word_count'] = df['name'].apply(lambda x: len(str(x).split()))
    df['desc_word_count'] = df['description'].apply(lambda x: len(str(x).split()))
    
    # Genel premium keyword'ler
    df['has_view'] = desc_lower.str.contains('view').astype(int)
    df['has_luxury'] = desc_lower.str.contains('luxury|luxurious').astype(int)
    df['has_modern'] = desc_lower.str.contains('modern').astype(int)
    df['has_spacious'] = desc_lower.str.contains('spacious|large').astype(int)
    df['has_cozy'] = desc_lower.str.contains('cozy|cosy').astype(int)
    
    # Istanbul'a ozel keyword'ler
    df['has_bosphorus'] = desc_lower.str.contains('bosphorus|boğaz|bogaz').astype(int)
    df['has_sea'] = desc_lower.str.contains('sea view|sea-view|deniz').astype(int)
    df['has_historic'] = desc_lower.str.contains('historic|historical|tarihi').astype(int)
    df['has_terrace'] = desc_lower.str.contains('terrace|teras').astype(int)
    df['has_rooftop'] = desc_lower.str.contains('rooftop|roof top|çatı').astype(int)
    df['has_balcony_desc'] = desc_lower.str.contains('balcony|balkon').astype(int)
    
    # Lokasyon keyword'leri
    df['has_center'] = desc_lower.str.contains('center|centre|merkez').astype(int)
    df['has_walking'] = desc_lower.str.contains('walking distance|yürüme').astype(int)
    df['has_metro'] = desc_lower.str.contains('metro|tram|tramvay').astype(int)
    
    # Name'den ozellikler
    df['name_has_luxury'] = name_lower.str.contains('luxury|deluxe|premium').astype(int)
    df['name_has_view'] = name_lower.str.contains('view|sea|bosphorus').astype(int)
    df['name_has_central'] = name_lower.str.contains('central|center|taksim|sultanahmet|besiktas|kadikoy').astype(int)
    
    # Premium skor (tum premium keyword'lerin toplami)
    df['text_premium_score'] = (
        df['has_view'] + df['has_luxury'] + df['has_modern'] + 
        df['has_bosphorus'] + df['has_sea'] + df['has_terrace'] + 
        df['has_rooftop'] + df['name_has_luxury'] + df['name_has_view']
    )
    
    # Location skor
    df['text_location_score'] = (
        df['has_center'] + df['has_walking'] + df['has_metro'] + 
        df['has_historic'] + df['name_has_central']
    )
    
    return df.drop(columns=['name', 'description'])

print("Train text features...")
train_text = create_text_features(train_raw)
print(f"  {len(train_text.columns)-1} feature olusturuldu")

print("Test text features...")
test_text = create_text_features(test_raw)

# 3. Processed veriye birlestir
print("\nBirlestiriliyor...")

# Eski text kolonlarini sil
text_cols = [col for col in train_proc.columns if col in train_text.columns and col != 'id']
train_proc = train_proc.drop(columns=text_cols, errors='ignore')
test_proc = test_proc.drop(columns=text_cols, errors='ignore')

train_merged = train_proc.merge(train_text, on='id', how='left')
test_merged = test_proc.merge(test_text, on='id', how='left')

# NaN doldur
for col in train_text.columns:
    if col != 'id':
        train_merged[col] = train_merged[col].fillna(0)
        test_merged[col] = test_merged[col].fillna(0)

print(f"Train shape: {train_proc.shape} -> {train_merged.shape}")
print(f"Test shape: {test_proc.shape} -> {test_merged.shape}")

# 4. Kaydet
print("\nKaydediliyor...")
train_merged.to_csv('data/processed/train_processed.csv', index=False)
test_merged.to_csv('data/processed/test_processed.csv', index=False)

# Ozetlemeler
print("\n" + "="*60)
print("FEATURE ISTATISTIKLERI")
print("="*60)
print(f"\nPremium keyword dagilimi (train):")
print(f"  has_bosphorus: {train_merged['has_bosphorus'].sum():,}")
print(f"  has_sea: {train_merged['has_sea'].sum():,}")
print(f"  has_terrace: {train_merged['has_terrace'].sum():,}")
print(f"  has_rooftop: {train_merged['has_rooftop'].sum():,}")
print(f"  has_view: {train_merged['has_view'].sum():,}")

print("\nTAMAMLANDI!")