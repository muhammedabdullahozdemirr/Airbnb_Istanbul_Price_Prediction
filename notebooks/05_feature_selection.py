"""
05 - Feature Selection & Popularity Features
Identifies correlated features, creates popularity features from reviews data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PART 1: Correlation Analysis
# =============================================================================

def find_high_correlations(train, threshold=0.85):
    """Find pairs of features with correlation above threshold."""
    print(f"Finding correlations above {threshold}...")
    
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
    """Get list of columns to drop based on high correlation."""
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) > threshold)]
    return to_drop


def remove_correlated_features(train, test, threshold=0.85):
    """Remove highly correlated features from train and test datasets."""
    train = train.copy()
    test = test.copy()
    
    print(f"Train shape before: {train.shape}")
    print(f"Test shape before: {test.shape}")
    
    numeric_data = train.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    to_drop = get_columns_to_drop(corr_matrix, threshold)
    
    # Don't drop important columns
    protected_cols = ['price', 'id', 'accommodates', 'bedrooms', 'bathrooms_clean']
    to_drop = [c for c in to_drop if c not in protected_cols]
    
    print(f"\nColumns to drop ({len(to_drop)}): {to_drop}")
    
    train_clean = train.drop(columns=to_drop, errors='ignore')
    test_clean = test.drop(columns=[c for c in to_drop if c in test.columns], errors='ignore')
    
    print(f"Train shape after: {train_clean.shape}")
    print(f"Test shape after: {test_clean.shape}")
    
    return train_clean, test_clean, to_drop


# =============================================================================
# PART 2: Popularity Features from Reviews
# =============================================================================

def create_popularity_features(reviews_path='data/raw/reviews.csv'):
    """Create popularity features from reviews data."""
    print("\n" + "="*50)
    print("POPULARITY FEATURES")
    print("="*50)
    
    reviews = pd.read_csv(reviews_path)
    reviews['date'] = pd.to_datetime(reviews['date'])
    max_date = reviews['date'].max()
    
    print(f"Reviews: {len(reviews):,} rows")
    print(f"Max review date: {max_date}")
    
    # Total reviews
    total_reviews = reviews.groupby('listing_id').size().reset_index(name='total_reviews')
    
    # Recent reviews (last 6 months)
    six_months_ago = max_date - pd.DateOffset(months=6)
    recent = reviews[reviews['date'] >= six_months_ago]
    recent_reviews = recent.groupby('listing_id').size().reset_index(name='recent_reviews')
    
    # Review frequency
    review_dates = reviews.groupby('listing_id')['date'].agg(['min', 'max', 'count'])
    review_dates['months_active'] = ((review_dates['max'] - review_dates['min']).dt.days / 30).clip(lower=1)
    review_dates['review_frequency'] = review_dates['count'] / review_dates['months_active']
    review_frequency = review_dates[['review_frequency']].reset_index()
    review_frequency.columns = ['listing_id', 'review_frequency']
    
    # Days since last review
    last_review = reviews.groupby('listing_id')['date'].max().reset_index()
    last_review['days_since_last_review'] = (max_date - last_review['date']).dt.days
    days_since = last_review[['listing_id', 'days_since_last_review']]
    days_since['is_active'] = (days_since['days_since_last_review'] < 90).astype(int)
    
    # Merge features
    popularity_features = total_reviews.merge(recent_reviews, on='listing_id', how='left')
    popularity_features = popularity_features.merge(review_frequency, on='listing_id', how='left')
    popularity_features = popularity_features.merge(days_since, on='listing_id', how='left')
    popularity_features['recent_reviews'] = popularity_features['recent_reviews'].fillna(0)
    
    # Log transformation
    popularity_features['total_reviews_log'] = np.log1p(popularity_features['total_reviews'])
    
    # Min-Max normalization
    scaler = MinMaxScaler()
    popularity_features['popularity_score'] = scaler.fit_transform(popularity_features[['total_reviews_log']])
    
    # Categorical bins
    popularity_features['popularity_category'] = pd.qcut(
        popularity_features['total_reviews'], q=3, 
        labels=['low', 'medium', 'high'], duplicates='drop'
    )
    popularity_features['popularity_level'] = popularity_features['popularity_category'].cat.codes
    
    final_features = popularity_features[['listing_id', 'total_reviews', 'total_reviews_log', 
                                          'popularity_score', 'popularity_level']]
    
    print(f"Popularity features created for {len(final_features):,} listings")
    print(f"Score distribution: mean={final_features['popularity_score'].mean():.3f}, std={final_features['popularity_score'].std():.3f}")
    
    return final_features


def add_popularity_to_data(train, test, popularity_features):
    """Add popularity features to train and test datasets."""
    # Remove existing popularity columns
    old_cols = ['total_reviews', 'total_reviews_log', 'recent_reviews', 'review_frequency', 
                'days_since_last_review', 'is_active', 'popularity_score', 'popularity_level']
    
    for col in old_cols:
        if col in train.columns:
            train = train.drop(col, axis=1)
        if col in test.columns:
            test = test.drop(col, axis=1)
    
    # Merge
    train_merged = train.merge(popularity_features, left_on='id', right_on='listing_id', how='left')
    train_merged = train_merged.drop('listing_id', axis=1)
    
    test_merged = test.merge(popularity_features, left_on='id', right_on='listing_id', how='left')
    test_merged = test_merged.drop('listing_id', axis=1)
    
    # Fill NaN
    popularity_cols = ['total_reviews', 'total_reviews_log', 'popularity_score', 'popularity_level']
    for col in popularity_cols:
        train_merged[col] = train_merged[col].fillna(0)
        test_merged[col] = test_merged[col].fillna(0)
    
    print(f"\nTrain shape: {train.shape} -> {train_merged.shape}")
    print(f"Test shape: {test.shape} -> {test_merged.shape}")
    
    return train_merged, test_merged


# =============================================================================
# MAIN
# =============================================================================

def save_final_data(train, test, output_dir='data/processed'):
    """Save final datasets."""
    train.to_csv(f'{output_dir}/train_final.csv', index=False)
    test.to_csv(f'{output_dir}/test_final.csv', index=False)
    print(f"\nFinal data saved to {output_dir}/")


if __name__ == "__main__":
    # Load processed data
    train = pd.read_csv('data/processed/train_processed.csv')
    test = pd.read_csv('data/processed/test_processed.csv')
    
    # 1. Correlation Analysis
    high_corr, corr_matrix = find_high_correlations(train, threshold=0.70)
    
    # 2. Remove correlated features
    train_clean, test_clean, dropped = remove_correlated_features(train, test, threshold=0.85)
    
    # 3. Add popularity features (optional - requires reviews.csv)
    try:
        popularity_features = create_popularity_features()
        train_clean, test_clean = add_popularity_to_data(train_clean, test_clean, popularity_features)
    except FileNotFoundError:
        print("\nreviews.csv not found, skipping popularity features")
    
    # 4. Save
    save_final_data(train_clean, test_clean)
