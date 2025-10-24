# Credit-Card-Fraud-Detection-using-XGBoost

This repository contains a complete machine learning pipeline for detecting credit card fraud using transaction data. The project demonstrates end-to-end data preprocessing, model training, evaluation, and visualization to handle highly imbalanced datasets.

Features

Data Exploration & Visualization: Understand transaction distributions and fraud patterns through plots and histograms.

Data Preprocessing:

Scaling of transaction amounts using RobustScaler

Removal of non-predictive features (Time)

Manual oversampling to balance classes

Machine Learning Models:

Logistic Regression

Random Forest Classifier

XGBoost Classifier

Model Evaluation:

ROC-AUC, F1-Score, Precision, Recall

Confusion Matrix

Precision-Recall and ROC Curves

Threshold Tuning: Optimize detection vs. false positive trade-off.

Feature Importance Analysis: Identify which features contribute most to fraud detection.

Dataset

Source: Kaggle Credit Card Fraud Detection Dataset

284,807 transactions, 492 fraud cases (0.17% of total)

Results

Best Model: XGBoost

ROC-AUC: 0.9734

Recall: 86% (detects most fraudulent transactions)

F1-Score: 0.3414 (balanced precision-recall)

Flexible threshold tuning allows prioritizing either fraud detection or minimizing false alarms.

Technical Stack

Python 3.x

Libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

Environment: Google Colab

Usage

Clone this repository.

Install required libraries:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn


Load the dataset into the notebook.

Run cells sequentially to execute preprocessing, training, evaluation, and visualization.
