# Credit Default Prediction

A machine learning project for predicting whether a borrower is likely to experience serious delinquency within two years.

The project started with a simple classification problem, but most of the work ended up being around data quality, class imbalance, feature engineering, model evaluation, and understanding where the model makes mistakes.

## Problem

The goal is to predict whether a borrower will experience serious delinquency within two years.

The target variable is:

- `SeriousDlqin2yrs`
  - `0` → No serious delinquency
  - `1` → Serious delinquency

This is a binary classification problem with a highly imbalanced target.

## Dataset

The project uses the **Give Me Some Credit** dataset from Kaggle.

**Dataset:** Give Me Some Credit  
**Source:** [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)

The dataset contains borrower-level financial and credit history information used to predict serious delinquency within two years.

Main files:

- `cs-training.csv`
- `cs-test.csv`
- `Data Dictionary.xls`
- `sampleEntry.csv`

The raw files are stored in `data/raw/`, while the processed dataset is stored in `data/cleaned/`.

## What I Did

### 1. Exploratory Data Analysis

I first explored the distributions, missing values, outliers, and relationships between the features and target.

Some issues I found:

- Around 20% of `MonthlyIncome` values were missing
- `NumberOfDependents` also contained missing values
- Placeholder-like values such as `96` and `98` appeared in delinquency-related columns
- An impossible age value of `0`
- Extreme values in some financial features
- Duplicate rows

Instead of removing everything automatically, I checked the features and decided how each issue should be handled.

### 2. Data Cleaning and Feature Engineering

After cleaning the data, I created several features to capture additional credit-risk information.

Some of the engineered features include:

- `TotalDelinquencies`
- `HasAnyDelinquency`
- `Has90DaysLate`
- `HighUtilization`
- `TotalCreditLines`
- `IncomePerCreditLine`
- `IncomeMissing`
- `AgeGroup`

The cleaned dataset contains **149,016 rows**.

### 3. Baseline Models

I started with simpler models to establish a baseline.

- **Logistic Regression** — high accuracy, but recall for defaulters was only around 15%
- **Decision Tree** — showed signs of overfitting, with ROC-AUC around 0.61
- **Random Forest** — recall remained below 19%

This showed why accuracy alone was not a useful metric for this problem.

I focused more on:

- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

### 4. Tree-Based Models

I compared several tree-based models:

- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- Extra Trees

For model tuning, I used **5-fold GridSearchCV** with F1 Score as the optimization metric.

XGBoost gave the best overall results and was selected as the final model.

## Final Model

The trained XGBoost model is stored in:

`models/best_xgboost.joblib`

Using a probability threshold of **0.45** on the evaluated test set:

| Metric | Score |
|---|---:|
| Accuracy | 91.68% |
| Precision | 39.45% |
| Recall | 48.02% |
| F1 Score | 43.32% |
| ROC-AUC | 86.22% |

The threshold was lowered from the default 0.5 to identify more potentially risky borrowers rather than focusing only on overall accuracy.

## SHAP Analysis

I used SHAP to understand which features were influencing the model's predictions.

The strongest features included:

1. `RevolvingUtilizationOfUnsecuredLines`
2. `HasAnyDelinquency`
3. `TotalDelinquencies`
4. `age`
5. `IncomePerCreditLine`

Some key observations:

- Higher credit utilization generally increased predicted risk
- Delinquency-related features were strong indicators of risk
- Age showed a non-linear relationship with predicted risk

## Error Analysis

I also looked at where the model was making mistakes, especially:

- False positives
- False negatives

This helped show that improving the model is not only about increasing accuracy. In a credit-risk problem, the cost of missing a real defaulter can be different from the cost of incorrectly flagging a borrower.

A cost-based threshold analysis was also explored as part of the project.

## Streamlit App

The project includes a Streamlit application that allows users to enter borrower information and receive a predicted default risk.

The application is located at:

`app/streamlit_app.py`

Run it using:

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```text
credit-default-prediction/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── cleaned/
│
├── models/
│   ├── agegroup_encoder.joblib
│   ├── best_xgboost.joblib
│   └── model_config.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_cleaning_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_tree_models.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   ├── 06_threshold_evaluation.ipynb
│   ├── 07_shap_analysis.ipynb
│   └── 08_Error_Analysis.ipynb
│
├── reports/
│   ├── baseline_results.csv
│   └── figures/
│
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

Then start the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

## What I Learned

This project helped me understand that building a machine learning model is not just about choosing an algorithm.

The main things I learned were:

- Understanding the data before modeling
- Handling missing values and unusual data correctly
- Working with imbalanced classification problems
- Feature engineering for tree-based models
- Choosing evaluation metrics based on the actual problem
- Hyperparameter tuning with GridSearchCV
- Using SHAP for model interpretation
- Analyzing false positives and false negatives
- Understanding how prediction thresholds affect model performance
- Deploying a trained model using Streamlit

The current model still misses a significant number of actual defaulters, so there is room to improve the threshold selection, cost-sensitive evaluation, and overall risk prediction process.

## Future Improvements

Some areas I would like to explore further:

- Select the final threshold using a separate validation set
- More detailed cost-sensitive evaluation
- Probability calibration
- Additional feature engineering
- Model monitoring
- Improving the Streamlit interface
- Testing the model on new data
