# Credit Card Fraud Detection

An end-to-end machine learning project to detect fraudulent credit card transactions. This project demonstrates a complete workflow for handling highly imbalanced datasets, from data preprocessing and oversampling to comparative model training and evaluation.



---

## Table of Contents
* [About the Project](#-about-the-project)
* [Dataset](#-dataset)
* [Project Workflow](#-project-workflow)
* [Model Performance](#-model-performance)
* [Feature Importance](#-feature-importance)
* [How to Run](#-how-to-run)
* [Technology Stack](#-technology-stack)

---

## About the Project

This repository contains a complete Python-based machine learning pipeline for detecting credit card fraud. The primary challenge of this problem is the **highly imbalanced dataset**, where fraudulent transactions represent only **0.17%** of the total.

This project demonstrates:
* Exploratory Data Analysis (EDA) to visualize the imbalance.
* Data preprocessing using `RobustScaler` to handle outliers in transaction amounts.
* Handling class imbalance using **Random Oversampling** on the training set.
* A comparative analysis of three supervised learning models:
    * Logistic Regression
    * Random Forest
    * XGBoost

---

## Dataset

The project uses the "Credit Card Fraud Detection" dataset, which is publicly available on Kaggle.

* **Total Transactions:** 284,807
* **Legitimate (Class 0):** 284,315 (99.83%)
* **Fraudulent (Class 1):** 492 (0.17%)
* **Features:** Contains 30 features, including `Time`, `Amount`, and 28 anonymized PCA components (`V1` to `V28`) to protect user confidentiality.

---

## Project Workflow

1.  **Data Loading & EDA:** The `creditcard.csv` file is loaded, and the extreme class imbalance is visualized.
2.  **Preprocessing:**
    * The `Amount` feature is scaled using `RobustScaler`, which is less sensitive to the large outliers present in the data.
    * The `Time` feature is dropped as it is non-predictive.
3.  **Train-Test Split:**
    * The data is split into a 70% training set and a 30% test set.
    * A `stratified` split is used to ensure the tiny percentage of fraud cases (0.17%) is represented equally in both the train and test sets.
4.  **Handling Class Imbalance:**
    * **Random Oversampling** is applied to the **training set only**.
    * This technique duplicates minority class (fraud) samples until they match the number of majority class (legitimate) samples.
    * The resulting balanced training set contains 199,020 legitimate and 199,020 fraud samples.
5.  **Model Training:** Three models (Logistic Regression, Random Forest, XGBoost) are trained on the new *balanced* training data.
6.  **Evaluation:** The trained models are then evaluated on the original, *unbalanced* test set to simulate real-world performance.

---

## Model Performance

Models were evaluated on the unseen (and unbalanced) test set. The results are as follows:

| Model | ROC-AUC | F1-Score | Precision (Fraud) | Recall (Fraud) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9699** | 0.1180 | 0.06 | 0.87 |
| **Random Forest** | 0.9441 | **0.8276** | 0.96 | 0.73 |
| **XGBoost** | 0.9624 | 0.7879 | 0.79 | 0.79 |



### Performance Analysis

* **Logistic Regression** achieved the highest **ROC-AUC (0.97)**, indicating excellent class separation. However, its **F1-Score (0.12)** is extremely low. This is a classic trap in imbalanced problems: it catches many frauds (87% Recall) but produces a massive number of false positives (only 6% Precision).
* **Random Forest** had the highest **F1-Score (0.83)**, showing the best balance between precision and recall. Its high precision (96%) means that when it flags a transaction as fraud, it is almost certainly correct.
* **XGBoost** provides a strong, balanced performance with a high ROC-AUC (0.96) and a very good F1-Score (0.79).

**Conclusion:** For a fraud detection system, minimizing false positives (Precision) and catching fraud (Recall) are more important than just the AUC. Therefore, the **Random Forest** and **XGBoost** models are far superior for this practical task.

---

## Feature Importance

Feature importance from the **Random Forest** model shows which (anonymized) features were most predictive of fraud. The PCA-transformed features `V10`, `V4`, and `V14` are the strongest indicators.



| Rank | Feature | Importance Score |
| :---: | :---: | :---: |
| 1 | `V10` | 0.1568 |
| 2 | `V4` | 0.1395 |
| 3 | `V14` | 0.1376 |
| 4 | `V12` | 0.1034 |
| 5 | `V11` | 0.0876 |

---

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection.git](https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection.git)
    cd Credit-Card-Fraud-Detection
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```
    *(Note: The `imbalanced-learn` library was imported in the notebook but not used, as a manual oversampler was implemented).*

3.  **Get the data:**
    * Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
    * Place the `creditcard.csv` file in the root directory of the project.

4.  **Run the notebook:**
    * Open and run the `credit_card_fraud_detection.ipynb` notebook using Jupyter Lab or Google Colab.

---

## Technology Stack

* **Python 3**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For preprocessing (`RobustScaler`), splitting (`train_test_split`), and metrics.
* **Models:** `LogisticRegression`, `RandomForestClassifier` (from Scikit-learn), and `XGBClassifier` (from XGBoost).
* **Matplotlib & Seaborn:** For data visualization.
