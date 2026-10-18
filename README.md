# Student Exam Score Prediction - Deployment Ready

End-to-end ML regression project to predict student exam scores from study habits, attendance, and socio-economic factors. Built with a **leak-free scikit-learn Pipeline** and ready for production deployment via Flask API.

[Python](https://img.shields.io/badge/Python-3.9+-blue)
[Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
[Status](https://img.shields.io/badge/Status-Deployment%20Ready-green)

**Best Model: Ridge Regression | MAE: 0.4456 | RMSE: 1.8001 | R²: 0.7707**

> Notebook converted to PDF: `Student_Exam_Score_Prediction.pdf` (29 pages, 31 cells) - Full code with detailed BAB markdown from Data Loading to Monitoring.

---

## 📊 Dataset Overview
- **File:** `Student_Performance_Factors.csv`
- **Size:** 6,607 rows × 20 columns (19 features + 1 target `Exam_Score`)
- **Numeric (7):** `Hours_Studied` (1-44), `Attendance` (60-100), `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`, `Exam_Score` (Target 55-101, note value 101 exists)
- **Categorical (13):**
  - Ordinal (7): `Parental_Involvement`, `Access_to_Resources`, `Motivation_Level`, `Family_Income`, `Teacher_Quality` [Low, Medium, High], `Parental_Education_Level` [High School, College, Postgraduate], `Distance_from_Home` [Near, Moderate, Far]
  - Nominal (6): `Extracurricular_Activities`, `Internet_Access`, `School_Type`, `Learning_Disabilities`, `Gender`, `Peer_Influence`
- **Missing:** Only 78 rows (1.18%) in 3 ordinal columns - handled inside pipeline to avoid leakage
- **Duplicates:** 0

---

## 🔬 Methodology (Sesuai Jupyter 7 BAB)

### BAB 1: Data Loading & Understanding [Cell 1-4]
- Import: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn` (ColumnTransformer, Pipeline, SimpleImputer, StandardScaler, OneHotEncoder, OrdinalEncoder, Ridge, RandomForest, GradientBoosting)
- `pd.read_csv`, `info()`, `describe(include="all")`, `select_dtypes`
- `numeric_features_scaled = [c for c in numeric_features if c != 'Exam_Score']` - target excluded from scaling
- `df = data_raw.copy()` to keep raw backup

### BAB 2: Data Cleaning [Cell 5,6,10]
- `isnull().sum()` and `duplicated()` check
- Missing handled with `SimpleImputer(most_frequent)` inside Pipeline (not dropped) to prevent data leakage. Numeric uses median (robust to outliers)

### BAB 3: EDA - Outlier & Distribution [Cell 7,8,9,11,12,13]
- **Outlier:** Boxplot per feature + IQR method `Q1-1.5*IQR` / `Q3+1.5*IQR` (Tukey standard)
- **Frequency:** `value_counts().sort_index()` to keep natural order, not frequency order
- **Distribution:** `sns.histplot(kde=True)`, `bins=20`
- **Key Detail - Why index [0]:** `numeric_features[0] = Hours_Studied` (skew 0.01, mean 19.9, std 5.9) - very normal, so after `z=(x-mean)/std` it forms a bell curve centered at 0 like the module example. If using `[3]=Previous_Scores` (uniform distribution) the shape stays flat. Empty gaps after standardization are artifacts because original data is discrete (integers) -> use `bins=20` to fix.

### BAB 4: ExDA - Correlation [Cell 14,15]
- `corr()`, `heatmap(annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')`
- Correlation to `Exam_Score`: `Attendance` 0.57, `Hours_Studied` 0.44, `Previous_Scores` 0.17. Weak multicollinearity (<0.1) among features. Linear model expected to win.

### BAB 5: Data Preparation - Splitting & Preprocessing [Cell 16-20]
- `X = drop(Exam_Score)`, `y = Exam_Score`
- `train_test_split(test_size=0.2, random_state=42)` - 80:20 Pareto, 42 for reproducibility
- Manual ordinal categories: `[Low, Medium, High]` etc. to avoid alphabetical misordering (High, Low, Medium)
- **Two preprocessors:**
  - `preprocessor_linear` (Ridge): `SimpleImputer(median)` + `StandardScaler` + `OrdinalEncoder` + `OneHotEncoder(drop='first', handle_unknown='ignore')` - Scaler needed because Ridge is scale-sensitive
  - `preprocessor_tree` (RF, GB): No scaler - trees only need split thresholds
- `OneHotEncoder` for nominal to avoid **false ordinal relationship**: e.g., Gender Male=1 Female=0 would make model think Male > Female. OneHot creates separate binary columns.
- All inside `Pipeline` to fit only on train data -> no leakage

### BAB 6: Modeling [Cell 21,22]
```python
pipe_ridge = Pipeline([('prep', preprocessor_linear), ('reg', Ridge(alpha=1.0, random_state=42))])
pipe_rf = Pipeline([('prep', preprocessor_tree), ('reg', RandomForestRegressor(n_estimators=300, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1))])
pipe_gb = Pipeline([('prep', preprocessor_tree), ('reg', GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, ...))])
```
- Ridge alpha=1.0 L2 regularization
- RF 300 trees for stability, min_samples_split=5 to prevent overfitting, n_jobs=-1 for all cores
- GB learning_rate=0.05 small for slow but accurate boosting, max_depth=3 weak learners

### BAB 7: Validation & Evaluation [Cell 23-29]
- Validation: `pipe_ridge.named_steps['prep'].named_transformers_['num'].transform()` - proves scaler fitted only on train (mean ~0 std ~1, test mean !=0)
- Metrics: MAE, RMSE = sqrt(MSE), R2. Sorted by RMSE (more sensitive to large errors)
- Visuals:
  - Barplot MAE/RMSE/R2 (steelblue, edgecolor black)
  - Scatter Actual vs Predicted with red diagonal `y=x`, alpha=0.4 - Ridge tightest
  - Residual plot `residual = y_test - y_pred`, `axhline(0)` - Ridge most random (no heteroscedasticity)
  - Top 15 Feature Importance: `get_feature_names_out()` + `feature_importances_` + `argsort()[-15:]` + `barh` forestgreen. Result: Attendance 41.5%, Hours_Studied 25.5%, Previous_Scores 8.3% = 75% dominance

| Model | MAE | RMSE | R2 |
| :--- | :--- | :--- | :--- |
| Ridge | 0.4456 | 1.8001 | 0.7707 |
| GradientBoosting | 0.6997 | 1.9189 | 0.7394 |
| RandomForest | 1.0446 | 2.1054 | 0.6863 |

### BAB 8: Saving Model [Cell 30-31 - from previous version]
```python
import joblib, pickle
joblib.dump(pipe_ridge, 'ridge_student_exam_prediction.joblib')
with open('ridge_student_exam_prediction.pkl','wb') as f: pickle.dump(pipe_ridge,f)
```
Pipeline saved (not just regressor) so preprocessing params are preserved.

### BAB 9: Deployment [Added per Deployment_dan_Monitoring_Notion_OCR.pdf]
- Load model: `joblib.load()` and `pickle.load()` - ensure same sklearn/joblib version (1.6.1 / 1.5.1) to avoid conflict
- Flask REST API (`app.py`):
```python
from flask import Flask, request, jsonify
import joblib
app = Flask(__name__)
model = joblib.load('ridge_student_exam_prediction.joblib')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    prediction = model.predict(pd.DataFrame([data]))
    return jsonify({'prediction': prediction.tolist()})
```
- `data.json` example for one student with all 19 raw features (categorical as strings)
- Test: `curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @data.json`
- Output: `{"prediction": [68.5]}`

### BAB 10: Monitoring
- **Metrics:** MSE/R2 for regression, Data Drift (input distribution vs train), Model Latency
- **Tools:** Prometheus/Grafana for dashboard, ELK Stack for logs, Sentry for error reporting, MLflow for experiment tracking
- **Current Baseline:** Mean Attendance Train 79.87, MSE 3.24 (RMSE²). Degradation if MSE > +30% or R2 < 0.65. No degradation detected yet (MSE 3.2405 == baseline)
- Flow: MODEL DEVELOPMENT -> SERVING (Offline Training, Prediction Service, End User Application) -> MONITORING (Feedback, Validation) -> RE-TRAINING

---

## 📁 Repository Structure
```
├── Student_Performance_Factors.csv
├── Student_Exam_Score_Prediction.pdf      # Latest: 29 pages, exported from ipynb (source of truth, since ipynb not supported as input)
├── Student_Exam_Score_Prediction.html     # Previous version
├── Deployment_dan_Monitoring_Notion_OCR.pdf # Reference for BAB 9-10
├── ridge_student_exam_prediction.joblib   # Best pipeline (joblib)
├── ridge_student_exam_prediction.pkl      # Best pipeline (pickle)
├── best_model_ridge.pkl / .joblib         # Alternative saves
├── app.py                                 # Flask API (BAB 9)
├── data.json                              # Sample request for API
└── README.md
```

## 🚀 Quick Start

### 1. Install
```bash
pip install pandas numpy matplotlib seaborn scikit-learn==1.6.1 joblib==1.5.1 flask
```

### 2. Train / View
- Open `Student_Exam_Score_Prediction.pdf` (full notebook) or run the ipynb if you have it locally

### 3. Load & Predict
```python
import joblib, pandas as pd
model = joblib.load('ridge_student_exam_prediction.joblib')
# Example raw input (no manual scaling needed)
sample = pd.DataFrame([{
  "Hours_Studied": 25, "Attendance": 85, "Parental_Involvement": "Medium",
  "Access_to_Resources": "High", "Extracurricular_Activities": "Yes",
  "Sleep_Hours": 7, "Previous_Scores": 75, "Motivation_Level": "High",
  "Internet_Access": "Yes", "Tutoring_Sessions": 2, "Family_Income": "Medium",
  "Teacher_Quality": "Medium", "School_Type": "Public", "Peer_Influence": "Positive",
  "Physical_Activity": 3, "Learning_Disabilities": "No",
  "Parental_Education_Level": "College", "Distance_from_Home": "Near", "Gender": "Male"
}])
print(model.predict(sample))  # -> [~68.x]
```

### 4. Deploy API
```bash
python app.py
# in another terminal
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @data.json
```

---

## 🔍 Key Insights
- Attendance and study hours explain 67% of exam score variance
- Linear Ridge beats tree ensembles because relationship is mostly linear
- Leak-free Pipeline is critical: imputer/scaler fitted only on train
- Model ready for production with Flask + monitoring for drift

## 📄 License & Credits
- Dataset: Student Performance Factors
- Deployment reference: Notion OCR - Deployment dan Monitoring (Dicoding)
- Notebook: 29-page PDF converted from ipynb (latest progress)
