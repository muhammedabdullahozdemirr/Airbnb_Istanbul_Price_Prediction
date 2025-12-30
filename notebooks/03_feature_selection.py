# FEATURE SELECTİON and POPULARİTY
"""
python notebooks/03_feature_selection.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


####Correlation Analysis


def find_high_correlations(train, threshold=0.85):
    numeric_data = train.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    stacked_corr = upper_tri.stack()
    high_corr = stacked_corr[abs(stacked_corr) > threshold].sort_values(ascending=False)
    
    print(f"\nHigh correlation pairs (>{threshold}):")
    for (col1, col2), val in high_corr.items():
        print(f"   {col1} - {col2}: {val:.4f}")
    
    return high_corr, corr_matrix

def get_columns_to_drop(corr_matrix, threshold=0.85):
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) > threshold)]
    return to_drop

def remove_correlated_features(train, test, threshold=0.85):
    train = train.copy()
    test = test.copy()
    
    print(f"train shape before: {train.shape}")
    print(f"test shape before: {test.shape}")
    
    numeric_data = train.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    to_drop = get_columns_to_drop(corr_matrix, threshold)
    
    # önemli olanları koruma altına al
    protected_cols = ['price', 'id', 'accommodates', 'bedrooms', 'bathrooms_clean']
    to_drop = [c for c in to_drop if c not in protected_cols]
    
    print(f"\ncolumns to drop ({len(to_drop)}): {to_drop}")
    
    train_clean= train.drop(columns=to_drop, errors='ignore')
    test_clean= test.drop(columns=[c for c in to_drop if c in test.columns], errors='ignore')
    
    print(f"train shape after: {train_clean.shape}")
    print(f"test shape after: {test_clean.shape}")
    
    return train_clean, test_clean, to_drop






#### Popularity Features


def create_popularity_features(reviews_path='data/raw/reviews.csv'):
    reviews=pd.read_csv(reviews_path)
    reviews['date']=pd.to_datetime(reviews['date'])
    max_date =reviews['date'].max()
    
    print(f"Reviews: {len(reviews):,}")
    print(f"max review date: {max_date}")
    
    total_reviews = reviews.groupby('listing_id').size().reset_index(name='total_reviews')
    
    #Son 6 ay
    six_months_ago = max_date - pd.DateOffset(months=6)
    recent = reviews[reviews['date'] >= six_months_ago]
    recent_reviews = recent.groupby('listing_id').size().reset_index(name='recent_reviews')
    
    #review sıklığı
    review_dates= reviews.groupby('listing_id')['date'].agg(['min','max','count'])
    review_dates['months_active'] =((review_dates['max'] - review_dates['min']).dt.days/30).clip(lower=1)
    review_dates['review_frequency'] = review_dates['count'] / review_dates['months_active']
    review_frequency = review_dates[['review_frequency']].reset_index()
    review_frequency.columns = ['listing_id', 'review_frequency']
    
    # son reviewdan bu yana ne kadar geçti
    last_review = reviews.groupby('listing_id')['date'].max().reset_index()
    last_review['days_since_last_review'] = (max_date - last_review['date']).dt.days
    days_since = last_review[['listing_id', 'days_since_last_review']]
    days_since['is_active'] = (days_since['days_since_last_review'] < 90).astype(int)
    
    # merge
    popularity_features = total_reviews.merge(recent_reviews, on='listing_id', how='left')
    popularity_features = popularity_features.merge(review_frequency, on='listing_id', how='left')
    popularity_features = popularity_features.merge(days_since, on='listing_id', how='left')
    popularity_features['recent_reviews'] = popularity_features['recent_reviews'].fillna(0)
    
    #log dönüşümü
    popularity_features['total_reviews_log'] = np.log1p(popularity_features['total_reviews'])
    
    # min-max normalization
    scaler=MinMaxScaler()
    popularity_features['popularity_score'] = scaler.fit_transform(popularity_features[['total_reviews_log']])
    
  
    popularity_features['popularity_category'] = pd.qcut(popularity_features['total_reviews'], q=3,labels=['low', 'medium', 'high'], duplicates='drop')
    popularity_features['popularity_level'] = popularity_features['popularity_category'].cat.codes
    
    final_features = popularity_features[['listing_id','total_reviews','total_reviews_log','popularity_score', 'popularity_level']]
    
    print(f"popularity features created for {len(final_features):,} listings")
    print(f"score distrubution: mean={final_features['popularity_score'].mean():.3f}, std={final_features['popularity_score'].std():.3f}")
    
    return final_features


def add_popularity_to_data(train, test, popularity_features):
    old_cols = ['total_reviews', 'total_reviews_log', 'recent_reviews', 'review_frequency', 
                'days_since_last_review', 'is_active', 'popularity_score', 'popularity_level']
    
    for col in old_cols:
        if col in train.columns:
            train = train.drop(col, axis=1)
        if col in test.columns:
            test = test.drop(col, axis=1)
    
    train_merged = train.merge(popularity_features, left_on='id', right_on='listing_id', how='left')
    train_merged = train_merged.drop('listing_id', axis=1)
    
    test_merged = test.merge(popularity_features, left_on='id', right_on='listing_id', how='left')
    test_merged = test_merged.drop('listing_id', axis=1)
    
    popularity_cols = ['total_reviews', 'total_reviews_log', 'popularity_score', 'popularity_level']
    for col in popularity_cols:
        train_merged[col] = train_merged[col].fillna(0)
        test_merged[col] = test_merged[col].fillna(0)
    
    print(f"train shape: {train.shape} -->{train_merged.shape}")
    print(f"test shape: {test.shape}  --> {test_merged.shape}")
    
    return train_merged, test_merged


def save_data(train, test, output_dir='data/processed'):
    train.to_csv(f'{output_dir}/train_processed.csv', index=False)
    test.to_csv(f'{output_dir}/test_processed.csv', index=False)


if __name__ == "__main__":
    train = pd.read_csv('data/processed/train_processed.csv')
    test = pd.read_csv('data/processed/test_processed.csv')

    high_corr, corr_matrix = find_high_correlations(train, threshold=0.70)
    train_clean, test_clean, dropped = remove_correlated_features(train, test, threshold=0.85)
    
    try:
        popularity_features = create_popularity_features()
        train_clean, test_clean = add_popularity_to_data(train_clean, test_clean, popularity_features)
    except FileNotFoundError:
        print("reviews.csv yok, popularity featuresı atlıcak")
    
    save_data(train_clean, test_clean)