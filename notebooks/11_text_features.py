"""
11 - Text Features
Extracts simple text stats from name and description.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TEXT FEATURES EXTRACTION")
print("="*60)

# 1. Ham veriyi ve processed veriyi yukle
print("Veriler yukleniyor...")
train_raw = pd.read_csv('data/raw/train.csv')
test_raw = pd.read_csv('data/raw/test.csv')

train_proc = pd.read_csv('data/processed/train_processed.csv')
test_proc = pd.read_csv('data/processed/test_processed.csv')

# 2. Text ozelliklerini hesapla
def create_text_features(df_raw):
    df = df_raw[['id', 'name', 'description']].copy()
    
    # Fill NA
    df['name'] = df['name'].fillna('')
    df['description'] = df['description'].fillna('')
    
    # Uzunluk ozellikleri
    df['name_len'] = df['name'].apply(len)
    df['desc_len'] = df['description'].apply(len)
    
    # Kelime sayisi
    df['name_word_count'] = df['name'].apply(lambda x: len(str(x).split()))
    df['desc_word_count'] = df['description'].apply(lambda x: len(str(x).split()))
    
    # Ozel anahtar kelimeler (basit sentiment/lux gostergeleri)
    df['has_view'] = df['description'].str.lower().str.contains('view').astype(int)
    df['has_luxury'] = df['description'].str.lower().str.contains('luxury').astype(int)
    df['has_heart'] = df['description'].str.lower().str.contains('heart').astype(int) # 'heart of istanbul'
    
    return df.drop(columns=['name', 'description'])

print("Train text features...")
train_text = create_text_features(train_raw)
print("Test text features...")
test_text = create_text_features(test_raw)

# 3. Processed veriye birlestir
print("\nBirlestiriliyor...")

# Varsa eski kolonlari sil
text_cols = ['name_len', 'desc_len', 'name_word_count', 'desc_word_count', 
             'has_view', 'has_luxury', 'has_heart']

train_proc = train_proc.drop(columns=[c for c in text_cols if c in train_proc.columns], errors='ignore')
test_proc = test_proc.drop(columns=[c for c in text_cols if c in test_proc.columns], errors='ignore')

train_merged = train_proc.merge(train_text, on='id', how='left')
test_merged = test_proc.merge(test_text, on='id', how='left')

# NaN doldur (olasi uyumsuzluklar icin)
for col in text_cols:
    train_merged[col] = train_merged[col].fillna(0)
    test_merged[col] = test_merged[col].fillna(0)

print(f"Train shape: {train_proc.shape} -> {train_merged.shape}")
print(f"Test shape: {test_proc.shape} -> {test_merged.shape}")

print("\nKaydediliyor...")
train_merged.to_csv('data/processed/train_processed.csv', index=False)
test_merged.to_csv('data/processed/test_processed.csv', index=False)

print("\nTAMAMLANDI!")
