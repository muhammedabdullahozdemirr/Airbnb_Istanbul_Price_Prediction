# MODEL COMPARİSON
"""
Farklı modeller --> Ridge, Random forest, XGBoost.

python notebooks/07_model_comparison.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


def prepare_data(train, test):
    exclude_cols = ['id','host_id','price', 'price_per_person']
    
    X= train.drop(columns=exclude_cols, errors='ignore')
    y =train['price']
    y_log =np.log1p(y)
    
    X_test=test.drop(columns=[c for c in exclude_cols if c in test.columns], errors='ignore')
    
    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X.columns]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, y_log, X_test


def evaluate_model(y_true, y_pred, model_name="Model"):
    y_true_orig =np.expm1(y_true)
    y_pred_orig =np.expm1(y_pred)
    
    rmse =np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae =mean_absolute_error(y_true_orig, y_pred_orig)
    r2 =r2_score(y_true_orig, y_pred_orig)
    log_rmse= np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {'model': model_name, 'log_rmse': log_rmse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def cross_validate_ridge(X, y_log, n_splits=5): #ridge regression 
    print("\n --ridge--")   
    kfold =KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler= StandardScaler()
    oof_preds= np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train,X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train,y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_scaled, y_train)
        oof_preds[val_idx] = model.predict(X_val_scaled)
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"  Fold {fold}: Log RMSE = {fold_rmse:.5f}")
    
    cv_rmse = np.sqrt(mean_squared_error(y_log, oof_preds))
    print(f"  CV Log RMSE: {cv_rmse:.5f}")
    
    return cv_rmse


def cross_validate_rf(X, y_log, n_splits=5):
    print("\n --Random Forest--")
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    rf_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'random_state': 42,
        'n_jobs': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
        
        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict(X_val)
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"  Fold {fold}: Log RMSE = {fold_rmse:.5f}")
    
    cv_rmse = np.sqrt(mean_squared_error(y_log, oof_preds))
    print(f"  CV Log RMSE: {cv_rmse:.5f}")
    
    return cv_rmse


def cross_validate_xgboost(X, y_log, n_splits=5):
    print("\n ---XGBoost---")
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    xgb_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.03,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.3,
        'reg_lambda': 1.5,
        'random_state': 42,
        'n_jobs': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
        
        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds[val_idx] = model.predict(X_val)
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"  Fold {fold}: Log RMSE = {fold_rmse:.5f}")
    
    cv_rmse = np.sqrt(mean_squared_error(y_log, oof_preds))
    print(f"  CV Log RMSE: {cv_rmse:.5f}")
    
    return cv_rmse, model


if __name__ == "__main__":
    train = pd.read_csv('data/processed/train_processed.csv')
    test = pd.read_csv('data/processed/test_processed.csv')
    
    Q1, Q3 = train['price'].quantile(0.01), train['price'].quantile(0.99)
    train = train[(train['price'] >= Q1) & (train['price'] <= Q3)]
    
    X, y, y_log, X_test = prepare_data(train, test)

    print("\n karşılaştırma")   
    results ={}
    
    
    results['Ridge'] = cross_validate_ridge(X, y_log)                 #Ridge
    results['Random Forest'] = cross_validate_rf(X, y_log)            #Random Forest
    results['XGBoost'], model = cross_validate_xgboost(X, y_log)      #XGBoost
    
   
    print("\nModel comparsion (Log RMSE):")
    for model_name, score in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {model_name}: {score:.5f}")
    
    best_model = min(results, key=results.get)
    print(f"\n bestmodel: {best_model} ({results[best_model]:.5f})")