# Airbnb Istanbul Price Prediction

**Team EnAi** - Data Mining Term Project

Kaggle Competition: Predicting nightly prices for Airbnb listings in Istanbul

## 🏆 Results

| Model | CV Log RMSE | Kaggle Private | Kaggle Public |
|-------|-------------|----------------|---------------|
| Ridge Regression | 0.523 | - | - |
| Random Forest | 0.446 | - | - |
| **XGBoost** | **0.422** | **0.495** | **0.507** |


## 📁 Project Structure

```
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── calendar.csv
│   │   └── reviews.csv
│   └── processed/              # Processed datasets
│       ├── train_preprocessed.csv
│       ├── test_preprocessed.csv
│       ├── train_processed.csv
│       ├── test_processed.csv
│       └── submission.csv
├── notebooks/
│   ├── 00_eda.py               # Exploratory Data Analysis
│   ├── 01_preprocess.py        # Data cleaning
│   ├── 02_feature_engineering.py
│   ├── 03_feature_selection.py
│   ├── 04_calendar_features.py
│   ├── 05_text_features.py
│   ├── 06_train.py             # Main model training
│   ├── 07_model_comparison.py  # Model comparison
│   └── 08_shap_analysis.ipynb  # SHAP interpretability
├── reports/                    # Visualizations
├── outputs/                    # Model outputs
├── requirements.txt
└── README.md
```

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/muhammedabdullahozdemirr/YZV311_2526_10.git 
cd YZV311_2526_10

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📊 Pipeline Execution

Run the scripts in order:

```bash
# 1. Data Preprocessing
python notebooks/01_preprocess.py

# 2. Feature Engineering
python notebooks/02_feature_engineering.py

# 3. Feature Selection & Popularity Features
python notebooks/03_feature_selection.py

# 4. Calendar Features
python notebooks/04_calendar_features.py

# 5. Text Features
python notebooks/05_text_features.py

# 6. Model Training
python notebooks/06_train.py
```

Optional scripts:
```bash
# EDA (Exploratory Data Analysis)
python notebooks/00_eda.py

# Model Comparison (Ridge, Random Forest, XGBoost)
python notebooks/07_model_comparison.py

# SHAP Analysis (Model Interpretability)
jupyter notebook notebooks/08_shap_analysis.ipynb
```

## 🔧 Feature Engineering

### Location Features
- `dist_to_sultanahmet` - Distance to Sultanahmet (km)
- `dist_to_taksim` - Distance to Taksim (km)
- `is_european_side` - European/Asian side flag

### Amenity Features
- `has_pool`, `has_parking`, `has_elevator`, `has_gym`, etc.
- `premium_amenities` - Weighted luxury amenity score
- `basic_amenities` - Basic amenity count

### Text Features (Istanbul-specific)
- `has_bosphorus` - Bosphorus view mention
- `has_sea` - Sea view mention
- `has_terrace`, `has_rooftop` - Outdoor space
- `text_premium_score` - Combined premium keyword score

### Calendar Features
- `availability_rate` - Listing availability ratio
- `booking_rate` - Occupancy rate
- `nights_flexibility` - Booking flexibility score

### Popularity Features
- `total_reviews` - Review count
- `popularity_score` - Normalized popularity metric

## 📈 Model Details

**Final Model:** XGBoost Regressor

```python
xgb_params = {
    'n_estimators': 1200,
    'max_depth': 6,
    'learning_rate': 0.015,
    'min_child_weight': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.3,
    'reg_lambda': 1.5
}
```

**Top 10 Important Features (SHAP):**
1. accommodates (18.0%)
2. room_type (9.1%)
3. longitude (6.2%)
4. latitude (5.2%)
5. text_premium_score (5.2%)
6. dist_to_sultanahmet (5.1%)
7. avg_min_nights (5.1%)
8. total_rooms (4.7%)
9. has_ac (4.4%)
10. bathrooms_clean (3.9%)

## 🔍 Model Interpretability (SHAP)

We used SHAP (SHapley Additive exPlanations) to interpret model decisions:

- **Summary Plot**: Impact of each feature on price prediction
- **Bar Plot**: Feature importance ranking
- **Dependence Plot**: Relationship between feature value and SHAP value

Visualizations are available in the `reports/` folder.

## 📋 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- shap

## 👥 Team EnAi

- Muhammed Abdullah Ozdemir
- Nurettin Macit
- Muhammed Hasan Bilal Cebeci

## 🤖 Generative AI Usage Disclosure

**Tool Used:** Claude, Chatgpt

**Purposes:**
- Code refactoring and cleanup
- Debugging assistance
- README generation
---