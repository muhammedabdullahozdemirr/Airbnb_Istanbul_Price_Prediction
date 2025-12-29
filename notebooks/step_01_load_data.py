# DATA LOADING
"""
Kullanim:
    from step01_load_data import load_data
    train,test = load_data()
"""

import pandas as pd

def load_data(data_dir="data/raw"):

    train_path = f"{data_dir}/train.csv"
    test_path = f"{data_dir}/test.csv"
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"Train: {train.shape}")
    print(f"Test:  {test.shape}")
    
    return train, test


def show_data_info(train, test):
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    
    only_in_train = train_cols - test_cols
    only_in_test = test_cols - train_cols
    
    print(f"\nOrtak kolon sayisi: {len(train_cols & test_cols)}")
    
    if only_in_train:
        print(f"just train:{only_in_train}")
    
    if only_in_test:
        print(f"just test:{only_in_test}")


def load_calendar(data_dir="data/raw"):
    path = f"{data_dir}/calendar.csv"
    print("dosya büyük ya biraz uzun yükleniyor az bekle")
    calendar = pd.read_csv(path)
    print(f"Calendar: {calendar.shape}")
    return calendar


def load_reviews(data_dir="data/raw"):
    path = f"{data_dir}/reviews.csv"
    reviews = pd.read_csv(path)
    print(f" Reviews: {reviews.shape}")
    return reviews

if __name__ == "__main__":
    train, test = load_data()
    show_data_info(train, test)