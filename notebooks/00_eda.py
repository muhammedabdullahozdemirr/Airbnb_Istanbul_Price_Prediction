 #EDA
"""
python notebooks/00_eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def analyze_data(train, test):

    print(f"train: {train.shape}")
    print(f"test: {test.shape}")
    print("\n isnull olanlar(train)")
    missing= train.isnull().sum()
    missing_pct= (missing / len(train) * 100).round(2)
    missing_df= pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
    missing_df =missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    if len(missing_df) > 0:
        print(missing_df.head())
    else:
        print("isnull yok")
  
    if 'price' in train.columns:
        print("\n targetımız (price)")
        print(train['price'].describe())
        print(f"\n skewness: {train['price'].skew():.2f}")
        print(f" Kurtosis : {train['price'].kurtosis():.2f}")
  
    print("\n numeric özellikler:")
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    print(len(numeric_cols))
    print(numeric_cols[:10])
    
    print("\ncategoric özellikler :")
    cat_cols= train.select_dtypes(include=['object']).columns.tolist()
    print(len(cat_cols))
    if len(cat_cols) >0:
        print(cat_cols)
    
    print("\n önemli featurelarımız:")
    key_features = ['accommodates', 'bedrooms', 'beds', 'bathrooms_clean', 'review_scores_rating', 'price']
    for col in key_features:
        if col in train.columns:
            print(f"\n{col} :")
            print(f"      Mean: {train[col].mean():.2f}")
            print(f"      Median: {train[col].median():.2f}")
            print(f"      Std: {train[col].std():.2f}")
            print(f"      Min: {train[col].min():.2f}, Max: {train[col].max():.2f}")
    
    return numeric_cols, cat_cols


def plot_distributions(train, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    plot_cols = ['price', 'accommodates', 'bedrooms','review_scores_rating', 'minimum_nights', 'amenities_count']
    
    for idx, col in enumerate(plot_cols):
        ax = axes[idx // 3, idx % 3]
        if col in train.columns:
            train[col].hist(bins=50, ax=ax, edgecolor='black')
            ax.set_title(f'{col} Dağılımı')
            ax.set_xlabel(col)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_correlation_heatmap(train, save_path=None):
    numeric_data= train.select_dtypes(include=[np.number])
    corr_matrix= numeric_data.corr()
    
    plt.figure(figsize=(16, 12))
    mask= np.triu(np.ones_like(corr_matrix,dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
    plt.title('Özellik Korelasyon Matrisi')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    train = pd.read_csv('data/processed/train_processed.csv')
    test = pd.read_csv('data/processed/test_processed.csv')
    
    numeric_cols, cat_cols = analyze_data(train, test)
    plot_distributions(train)
    plot_correlation_heatmap(train)