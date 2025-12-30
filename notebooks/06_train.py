# MODEL TRAİNİNG
"""
python notebooks/06_train.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


def create_features(df):
    df = df.copy()
    # Kapasite oranları
    df['accommodates_per_bedroom'] = df['accommodates'] / (df['bedrooms'] + 1)
    df['accommodates_per_bathroom'] = df['accommodates'] / (df['bathrooms_clean'] + 1)
    df['bedrooms_per_bathroom'] = df['bedrooms'] / (df['bathrooms_clean'] + 1)
    df['beds_per_bedroom'] = df['beds'] / (df['bedrooms'] + 1)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms_clean']
    df['space_efficiency'] = df['accommodates'] / (df['bedrooms'] + df['beds'] + 1)
    
    # Review composite score
    review_cols = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness','review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']
    existing_review_cols = [col for col in review_cols if col in df.columns]
    if existing_review_cols:
        for col in existing_review_cols:
            df[col] = df[col].fillna(df[col].median())
        df['review_composite']= df[existing_review_cols].mean(axis=1)
        df['review_variance']= df[existing_review_cols].std(axis=1)
    
    # Host kalite skoru
    df['host_quality'] =(
        df['host_is_superhost'].fillna(0)* 3+
        df['host_identity_verified'].fillna(0)* 2+
        df['host_has_profile_pic'].fillna(0)* 1+
        (df['host_response_rate'].fillna(0)/100)+
        (df['host_acceptance_rate'].fillna(0) /100)
    )
    
    #rezervasyon esnekliği
    df['booking_flexibility'] = 1 /(df['minimum_nights'] +1)
    df['long_term_friendly'] =(df['minimum_nights'] >=30).astype(int)
    
    if 'amenities_count' in df.columns:
        df['amenity_per_person'] = df['amenities_count'] / (df['accommodates'] +1)
    
    if all(col in df.columns for col in ['has_wifi', 'has_kitchen', 'has_ac']):
        df['luxury_amenities'] = df['has_wifi'] + df['has_kitchen'] + df['has_ac']
    
    # multilister
    if 'host_listings_count' in df.columns:
        df['is_multi_lister'] = (df['host_listings_count'] > 1).astype(int)
    
    return df


def prepare_data(train, test):
    exclude_cols = ['id', 'host_id', 'price', 'price_per_person']
    
     #yüksek corr olanlar
    high_corr_cols =[
        'host_listings_count', 'host_total_listings_count',
        'calculated_host_listings_count_entire_homes',
        'calculated_host_listings_count_private_rooms',
        'calculated_host_listings_count_shared_rooms',
        'review_scores_rating', 
        'review_scores_accuracy', 
        'review_scores_cleanliness',
        'review_scores_checkin', 
        'review_scores_communication', 
        'review_scores_value'
    ]
    
    exclude_cols=exclude_cols + high_corr_cols
    
    X= train.drop(columns=exclude_cols, errors='ignore')
    y= train['price']
    y_log= np.log1p(y)
    
    X_test =test.drop(columns=['id', 'host_id', 'price_per_person'], errors='ignore')
    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X.columns]
    
    return X, y_log, X_test


def train_xgboost(X, y_log, X_test):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    xgb_params = {
        'n_estimators':1200,
        'max_depth': 6,
        'learning_rate': 0.015,
        'min_child_weight': 4,
        'subsample':0.8,
        'colsample_bytree': 0.75,
        'colsample_bylevel': 0.75,
        'reg_alpha':0.3,
        'reg_lambda':1.5,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    oof_preds=np.zeros(len(X))
    test_preds=np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"Fold {fold}/5...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
        
        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        oof_preds[val_idx] =model.predict(X_val)
        test_preds += model.predict(X_test) /5
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"  Log RMSE :{fold_rmse:.5f}")
    
    cv_rmse =np.sqrt(mean_squared_error(y_log, oof_preds))
    
    return test_preds, cv_rmse, model


if __name__ == "__main__":
    train = pd.read_csv('data/processed/train_processed.csv')
    test = pd.read_csv('data/processed/test_processed.csv')
    print(f"train shape: {train.shape}")
    print(f"test shape: {test.shape}")
   
    train = create_features(train)
    test = create_features(test)
    print(f"train shape after FE: {train.shape}")
    print(f"test shape after FE: {test.shape}")
    
    # Outlier handling
    Q1= train['price'].quantile(0.01)
    Q3= train['price'].quantile(0.99)
    train_clean = train[(train['price'] >= Q1) & (train['price'] <= Q3)].copy()
    print(f"Ttrain shape after outlier removal : {train_clean.shape}")
   
    X, y_log, X_test = prepare_data(train_clean, test)
    print(f"features shape: {X.shape}")
    
    # traim
    test_preds, cv_rmse, model = train_xgboost(X, y_log, X_test)
    
    print(f"CV Log RMSE: {cv_rmse:.5f}")
    
    #preds
    y_pred= np.expm1(test_preds)
    y_pred= np.maximum(y_pred,0)
    
    print(f"\n predictions:")
    print(f"  Min: {y_pred.min():.2f}")
    print(f"  Max: {y_pred.max():.2f}")
    print(f"  Mean: {y_pred.mean():.2f}")
    print(f"  Median: {np.median(y_pred):.2f}")
    
    submission = pd.DataFrame({'id': test['id'], 'TARGET': y_pred})
    submission.to_csv('data/processed/submission.csv', index=False)
   
    fi= pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})  #feature önemi
    fi= fi.sort_values('importance', ascending=False)
    print(fi.head(20).to_string(index=False))
