# YZV311E Data Mining Term Project - Team EnAi

## 🏠 Explaining Airbnb Prices: A Transparent and Interpretable Data Mining Approach

Predicting Airbnb prices for Istanbul listings using interpretable data mining techniques.

### 👥 Team Members
| Name | Email | Role |
|------|-------|------|
| Muhammed Abdullah Özdemir | ozdemirmuh22@itu.edu.tr | Data Preprocessing, Feature Engineering, Evaluation |
| Muhammed Hasan Bilal Cebeci | cebecim22@itu.edu.tr | Model Training, Hyperparameter Tuning |
| Nurettin Macit | macit22@itu.edu.tr | EDA, SHAP Analysis, Visualization |

### 📁 Project Structure
```
├── data/
│   ├── raw/              # Original Kaggle data
│   └── processed/        # Preprocessed data
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── src/                  # Python modules
├── outputs/              # Kaggle submissions
└── reports/              # Figures and reports
```

### 🔗 Links
- **Kaggle Competition:** [Airbnb Istanbul Price Prediction](https://www.kaggle.com/competitions/yzv311-2526-airbnb-price-prediction)
- **Kaggle Team Name:** Team EnAi

### 🛠️ Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 📊 Dataset
- `train.csv`: 24,153 listings with 58 features
- `test.csv`: 4,750 listings for prediction
- `calendar.csv`: 10.8M rows of availability data
- `reviews.csv`: 516K review records

### 📈 Methodology
1. Exploratory Data Analysis
2. Data Preprocessing (missing values, outliers)
3. Feature Engineering
4. Model Training (Ridge, Lasso, Random Forest, LightGBM)
5. Interpretability Analysis (SHAP, Feature Importance)

### 📝 Timeline
- [x] Proposal Submission (Oct 31)
- [ ] Intermediate Meeting (Dec 2-6)
- [ ] Competition Deadline (Dec 29)
- [ ] Demo & Presentation (Dec 30)
- [ ] Final Report (Jan 6)
