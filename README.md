# Credit Risk Modelling & Loan Default Prediction

## 📌 Project Overview

Credit risk modelling is used by financial institutions to estimate the likelihood that a borrower will default on a loan.

In this project, I developed a machine learning classification system to predict **loan default status**, where:

- `0` = Non-Default
- `1` = Default

The project focuses on building and comparing multiple machine learning models, handling class imbalance using SMOTE, optimizing the best-performing model using Optuna, and tuning the classification threshold to improve the detection of default borrowers.

---

## 🎯 Objective

The main objective of this project is to:

- Predict whether a borrower is likely to default.
- Handle class imbalance in the loan dataset.
- Compare different classification algorithms.
- Optimize the best-performing model.
- Evaluate models using appropriate classification metrics.
- Improve the detection of default borrowers through probability-threshold tuning.

---

## 🛠️ Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Optuna

---

## 🔄 Project Workflow

```text
Data Collection
      ↓
Data Understanding
      ↓
Exploratory Data Analysis
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Train-Test Split
      ↓
SMOTE for Class Imbalance
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Random Forest vs XGBoost
      ↓
Optuna Hyperparameter Tuning
      ↓
ROC-AUC Comparison
      ↓
Threshold Tuning
      ↓
Final XGBoost Model
