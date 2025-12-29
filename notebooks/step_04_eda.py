"""
04 - Exploratory Data Analysis (EDA)
Analyzes data distributions, missing values, and key statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def analyze_data(train, test):
    """
    Perform exploratory data analysis on train and test datasets.
    """
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # =========================================================================
    # 1. Basic Info
    # =========================================================================
    print("\n1. DATASET SHAPES")
    print(f"   Train: {train.shape}")
    print(f"   Test: {test.shape}")
    
    # =========================================================================
    # 2. Missing Values
    # =========================================================================
    print("\n2. MISSING VALUES (Train)")
    missing = train.isnull().sum()
    missing_pct = (missing / len(train) * 100).round(2)
    missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    if len(missing_df) > 0:
        print(missing_df.head(15))
    else:
        print("   No missing values!")
    
    # =========================================================================
    # 3. Target Variable (Price)
    # =========================================================================
    if 'price' in train.columns:
        print("\n3. TARGET VARIABLE (Price)")
        print(train['price'].describe())
        print(f"\n   Skewness: {train['price'].skew():.2f}")
        print(f"   Kurtosis: {train['price'].kurtosis():.2f}")
    
    # =========================================================================
    # 4. Numerical Features
    # =========================================================================
    print("\n4. NUMERICAL FEATURES")
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    print(f"   Count: {len(numeric_cols)}")
    print(f"   Columns: {numeric_cols[:10]}...")
    
    # =========================================================================
    # 5. Categorical Features
    # =========================================================================
    print("\n5. CATEGORICAL FEATURES")
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    print(f"   Count: {len(cat_cols)}")
    if len(cat_cols) > 0:
        print(f"   Columns: {cat_cols}")
    
    # =========================================================================
    # 6. Key Features Statistics
    # =========================================================================
    print("\n6. KEY FEATURES STATISTICS")
    key_features = ['accommodates', 'bedrooms', 'beds', 'bathrooms_clean', 
                    'review_scores_rating', 'price']
    for col in key_features:
        if col in train.columns:
            print(f"\n   {col}:")
            print(f"      Mean: {train[col].mean():.2f}")
            print(f"      Median: {train[col].median():.2f}")
            print(f"      Std: {train[col].std():.2f}")
            print(f"      Min: {train[col].min():.2f}, Max: {train[col].max():.2f}")
    
    return numeric_cols, cat_cols


def plot_distributions(train, save_path=None):
    """Plot key feature distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    plot_cols = ['price', 'accommodates', 'bedrooms', 
                 'review_scores_rating', 'minimum_nights', 'amenities_count']
    
    for idx, col in enumerate(plot_cols):
        ax = axes[idx // 3, idx % 3]
        if col in train.columns:
            train[col].hist(bins=50, ax=ax, edgecolor='black')
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"\nPlot saved to {save_path}")
    plt.show()


def plot_correlation_heatmap(train, save_path=None):
    """Plot correlation heatmap for numeric features."""
    numeric_data = train.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"\nHeatmap saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    # Load processed data
    train = pd.read_csv('data/processed/train_processed.csv')
    test = pd.read_csv('data/processed/test_processed.csv')
    
    # Analyze
    numeric_cols, cat_cols = analyze_data(train, test)
    
    # Optional: Plot distributions
    # plot_distributions(train)
    # plot_correlation_heatmap(train)
