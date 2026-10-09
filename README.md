# Student Exam Score Prediction

End-to-end regression project to predict student exam scores based on study habits, attendance, and socio-economic factors. Built with a leak-free scikit-learn Pipeline approach.

**Best Model: Ridge Regression | MAE: 0.44 | RMSE: 1.80 | R²: 0.77**

## Dataset Overview
- **Source:** `Student_Performance_Factors.csv`
- **Size:** 6,607 rows, 20 columns (19 features + 1 target)
- **Numeric (7):** `Hours_Studied` (1-44), `Attendance` (60-100), `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`, `Exam_Score` (Target: 55-101)
- **Categorical (13):** `Parental_Involvement`, `Access_to_Resources`, `Motivation_Level`, `Family_Income`, `Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home` (Ordinal) and `Extracurricular_Activities`, `Internet_Access`, `School_Type`, `Learning_Disabilities`, `Gender`, `Peer_Influence` (Nominal)
- **Missing Values:** Only 3 categorical columns - handled inside pipeline.

## Methodology

### 1. Data Understanding & Cleaning
- Separated numeric vs categorical features programmatically with `select_dtypes`
- Used `df.copy()` to keep raw data intact
- No duplicate rows found

### 2. EDA & Outlier Analysis
- **Outlier Detection:** Boxplot + IQR method (Q1 - 1.5*IQR)
- **Distribution:** `Hours_Studied` is the most normal (skew 0.01) - chosen as demo for standardization. The empty gaps after standardization are expected because the original data is discrete (integer values).
- **Correlation (ExDA):** `Attendance` (0.57) and `Hours_Studied` (0.44) have the highest correlation with `Exam_Score`

### 3. Data Preparation (Leak-Free Pipeline)
To prevent data leakage, all preprocessing is fitted ONLY on train data:
- **Split:** 80% Train / 20% Test, `random_state=42` for reproducibility
- **Numeric:** `SimpleImputer(median)` + `StandardScaler()` - median is robust to outliers
- **Ordinal:** `SimpleImputer(most_frequent)` + `OrdinalEncoder` with manually defined order `[Low, Medium, High]` to preserve hierarchy
- **Nominal:** `SimpleImputer(most_frequent)` + `OneHotEncoder(drop='first', handle_unknown='ignore')` - OneHot is used instead of Label Encoding to avoid **false ordinal relationship** (e.g., Male=1, Female=0 would make the model think Male > Female). OneHot creates separate binary columns.

Two preprocessors were built:
- `preprocessor_linear` (with Scaler) for Ridge
- `preprocessor_tree` (without Scaler) for tree-based models

### 4. Modeling
Three algorithms compared inside `Pipeline`:

| Model | Key Hyperparameters | Reason |
| :--- | :--- | :--- |
| **Ridge** | `alpha=1.0` | L2 regularization to prevent overfitting |
| **Random Forest** | `n_estimators=300, min_samples_split=5, min_samples_leaf=2, n_jobs=-1` | 300 trees for stability, min samples to avoid deep overfitting |
| **Gradient Boosting** | `n_estimators=300, learning_rate=0.05, max_depth=3` | Low learning rate + shallow trees (weak learners) for better generalization |

### 5. Evaluation
Sorted by RMSE (more sensitive to large errors than MAE):

| Model | MAE | RMSE | R2 |
| :--- | :--- | :--- | :--- |
| **Ridge** | **0.4456** | **1.8001** | **0.7707** |
| GradientBoosting | 0.6997 | 1.9189 | 0.7394 |
| RandomForest | 1.0446 | 2.1054 | 0.6863 |

**Visuals:**
- Actual vs Predicted scatter with ideal `y=x` red line
- Residual plot with `axhline(0)` - Ridge residuals are most random (no heteroscedasticity)
- Top 15 Feature Importance (Random Forest)

### 6. Key Findings
Top 3 most influential features (75% total importance):
1.  **Attendance (41.5%)**
2.  **Hours_Studied (25.5%)**
3.  **Previous_Scores (8.3%)**

Linear relationship dominates this dataset, which is why Ridge outperforms complex tree models.

## Project Structure
```txt
├── Student_Performance_Factors.csv
├── Student_Exam_Score_Prediction.ipynb
├── ridge_student_exam_prediction.pkl
├── README.md
```

## Installation & Usage
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```
# Train
```bash
jupyter notebook Student_Exam_Score_Prediction.ipynb
```

# Load saved model for inference
```python
import joblib
model = joblib.load("ridge_student_exam_prediction.joblib")
prediction = model.predict(X_new) # X_new = raw dataframe with same columns
```
