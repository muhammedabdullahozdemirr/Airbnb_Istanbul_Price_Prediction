"""
01 - Data Loading
Loads raw train and test data from CSV files.
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load raw train and test datasets."""
    print("Loading data...")
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    return train, test

if __name__ == "__main__":
    train, test = load_data()
    print("\nTrain columns:", list(train.columns))
    print("\nTest columns:", list(test.columns))
    
    # Columns in train but not in test
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    print("\nColumns in train but not in test:")
    print(train_cols - test_cols)
