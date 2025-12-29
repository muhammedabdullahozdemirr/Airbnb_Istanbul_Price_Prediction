"""
06 - Model Training
Trains and evaluates machine learning models for price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def prepare_data(train, test, exclude_cols=None):
    """
    Prepare features and target for modeling.
    """
    if exclude_cols is None:
        exclude_cols = ['id', 'host_id', 'price', 'price_per_person']
    
    # Features and target
    X = train.drop(columns=exclude_cols, errors='ignore')
    y = train['price']
    y_log = np.log1p(y)
    
    # Test features
    X_test = test.drop(columns=[c for c in exclude_cols if c in test.columns], errors='ignore')
    
    # Align columns
    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X.columns]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    return X, y, y_log, X_test


def train_ridge(X_train, X_val, y_train, y_val):
    """Train Ridge Regression model."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_val_scaled)
    return model, scaler, y_pred


def train_random_forest(X_train, y_train, params=None):
    """Train Random Forest model."""
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'random_state': 42,
            'n_jobs': -1
        }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, X_val=None, y_val=None, params=None):
    """Train XGBoost model."""
    if params is None:
        params = {
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
    
    model = XGBRegressor(**params)
    
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    
    return model


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Evaluate model performance."""
    # Convert back from log if needed
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    r2 = r2_score(y_true_orig, y_pred_orig)
    log_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"\n{model_name} Results:")
    print(f"   Log RMSE: {log_rmse:.5f}")
    print(f"   RMSE: {rmse:,.2f}")
    print(f"   MAE: {mae:,.2f}")
    print(f"   R² Score: {r2:.4f}")
    
    return {'model': model_name, 'log_rmse': log_rmse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def cross_validate_xgboost(X, y_log, n_splits=5):
    """Perform K-Fold cross-validation with XGBoost."""
    print(f"\nRunning {n_splits}-Fold Cross-Validation...")
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    
    fold = 0
    for train_idx, val_idx in kfold.split(X):
        fold += 1
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
        
        model = train_xgboost(X_train, y_train, X_val, y_val)
        oof_preds[val_idx] = model.predict(X_val)
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"   Fold {fold}: Log RMSE = {fold_rmse:.5f}")
    
    cv_rmse = np.sqrt(mean_squared_error(y_log, oof_preds))
    print(f"\nCV Log RMSE: {cv_rmse:.5f}")
    
    return cv_rmse, model


def get_feature_importance(model, feature_names, top_n=20):
    """Get and display feature importance."""
    fi = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Features:")
    print(fi.head(top_n).to_string(index=False))
    
    return fi


def create_submission(test, predictions, output_path='data/processed/submission.csv'):
    """Create submission file."""
    submission = pd.DataFrame({
        'id': test['id'],
        'price': predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Submission shape: {submission.shape}")
    return submission


if __name__ == "__main__":
    # Load data
    train = pd.read_csv('data/processed/train_processed.csv')
    test = pd.read_csv('data/processed/test_processed.csv')
    
    # Remove outliers
    Q1, Q3 = train['price'].quantile(0.01), train['price'].quantile(0.99)
    train = train[(train['price'] >= Q1) & (train['price'] <= Q3)]
    
    # Prepare data
    X, y, y_log, X_test = prepare_data(train, test)
    
    # Cross-validate
    cv_rmse, model = cross_validate_xgboost(X, y_log)
    
    # Feature importance
    get_feature_importance(model, X.columns)
    
    # Final predictions
    final_model = train_xgboost(X, y_log)
    test_preds = np.expm1(final_model.predict(X_test))
    test_preds = np.maximum(test_preds, 0)
    
    # Create submission
    create_submission(test, test_preds)
