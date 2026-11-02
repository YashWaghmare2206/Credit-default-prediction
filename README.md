# Credit Default Prediction with Tree-Based Models

## 1. Project Overview

This project builds and compares machine learning models to predict whether a borrower will experience serious delinquency (90+ days past due or worse) within two years.

The project focuses on understanding and comparing tree-based machine learning algorithms, while following a leakage-free validation strategy and using metrics appropriate for an imbalanced classification problem.

## 2. Problem Statement

The goal is to predict the probability that a borrower will become seriously delinquent within two years based on their available financial and credit-history information.

The target variable is `SeriousDlqin2yrs`.

- `0` → No serious delinquency within two years
- `1` → Serious delinquency within two years

The model is intended as a learning and research project for understanding credit-risk prediction and tree-based machine learning.

## 3. Business Context

The model is considered from the perspective of a lender evaluating borrower risk.

A false negative means that a borrower who eventually becomes seriously delinquent was classified as lower risk. A false positive means that a borrower who would not become seriously delinquent was incorrectly classified as higher risk.

For this project, we initially assume that a false negative has 10 times the cost of a false positive.

This is an explicit modelling assumption and will be tested under different cost ratios later.

## 4. Dataset

Dataset: Give Me Some Credit

Source: Kaggle

The dataset contains borrower-level financial and credit-history information.

The primary training dataset is `cs-training.csv`.

Additional downloaded files include:

- `cs-test.csv`
- `Data Dictionary.xls`
- `sampleEntry.csv`

The raw data is stored locally under `data/raw/` and is not modified.

## 5. Features

The dataset contains borrower information including:

- Revolving credit utilization
- Age
- Previous 30–59 day delinquency
- Debt ratio
- Monthly income
- Number of open credit lines and loans
- Previous 90+ day delinquency
- Number of real-estate loans or lines
- Previous 60–89 day delinquency
- Number of dependents

## 6. Target

`SeriousDlqin2yrs`

Binary classification target representing serious delinquency within two years.

## 7. Models

The project will compare:

1. Dummy Classifier
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. XGBoost
6. LightGBM
7. CatBoost

## 8. Evaluation

Primary metrics:

- PR-AUC
- ROC-AUC

Additional metrics:

- Precision
- Recall
- F1 Score
- Confusion Matrix
- Brier Score
- Cost-based evaluation

## 9. Project Status

Currently completing project setup and data understanding.