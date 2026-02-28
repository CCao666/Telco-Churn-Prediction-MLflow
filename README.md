## 🎯 Project Goal

The objective of this project is to build a production-style churn prediction system that:

- Identifies customers at high risk of cancellation  
- Handles class imbalance in a principled way  
- Compares multiple model families  
- Tracks experiments using MLflow  
- Selects and registers the best-performing model  
- Optimizes decision thresholds for business tradeoffs  

Unlike a simple notebook-based modeling exercise, this project demonstrates a structured ML lifecycle workflow:

**EDA → Preprocessing → Multi-model comparison → Hyperparameter tuning → Model selection → Registry**

The final production candidate is an optimized, class-weighted **XGBoost** model, tuned to balance churn detection and intervention cost.

## 📂 Repository Structure

The repository is organized to reflect a clean, production-oriented machine learning workflow:

```text
.
├── data/
│   └── raw_data.csv              # Original Telco Churn dataset
│
├── notebooks/
│   └── EDA.ipynb                 # Exploratory data analysis and business insights
│
├── src/
│   ├── preprocess.py             # Data cleaning, encoding, scaling, train/test split
│   ├── train.py                  # Unified training script with MLflow logging
│   ├── hpo.py                    # Multi-model hyperparameter sweep
│   ├── xgb_finetune.py           # Fine-grained XGBoost hyperparameter tuning
│   ├── register_model.py         # Automated best-model registration to MLflow Registry
│   └── save_best_model.py        # Export finalized production model after business validation
│
├── models/                       # Stored production-ready model artifacts (.pkl)
│
├── MLProject                     # MLflow project definition
├── conda.yaml                    # Reproducible environment specification
├── README.md                     # Project documentation
└── .gitignore                    # Ignore artifacts and cache files

```
This structure separates experimentation, model training, tracking, and production artifacts — mirroring how ML systems are organized in real-world environments.

## 📊 Dataset & Problem Framing

This project uses the **Telco Customer Churn dataset**, where each row represents a telecom customer and the target variable indicates whether the customer churned.

### Dataset Overview

- ~7,000 customers
- Binary classification problem
- Churn rate: **~26.5%**
- Mix of numerical and categorical features
- Behavioral and billing-related attributes

Because churners represent a minority class, this is an **imbalanced classification problem**.

If not handled properly, models tend to over-predict the majority class (non-churn), leading to high accuracy but poor churn detection.


## ⚖️ Handling Class Imbalance

To address imbalance, this project applies:

- **Class weighting during training**
  - `scale_pos_weight` for XGBoost
  - `class_weight='balanced'` for other models
- **Threshold tuning on predicted probabilities**
  - Final optimized threshold: **0.51**

This ensures the model prioritizes identifying churners (high recall) while controlling unnecessary intervention costs (precision tradeoff).


## 🔍 Exploratory Data Analysis (EDA) – Key Findings

Based on detailed EDA (see `notebooks/EDA.ipynb`), several strong behavioral patterns emerge:

### 1️⃣ Contract Type

- **Month-to-Month contracts show significantly higher churn**
- Long-term contracts (1–2 years) strongly reduce churn probability

Interpretation:
Customers with flexible contracts are less committed and more price-sensitive.


### 2️⃣ Tenure

- Short-tenure customers are at much higher risk
- Tenure shows strong negative correlation with churn

Interpretation:
Customer loyalty increases over time.


### 3️⃣ Monthly Charges

- Higher monthly charges increase churn likelihood
- Particularly strong effect within month-to-month contracts

Interpretation:
Pricing pressure drives churn in flexible subscription plans.


### 4️⃣ Service Adoption

- Customers subscribing to more services churn less
- Clear negative relationship between total services and churn probability

Interpretation:
Higher product engagement increases switching cost.


### 5️⃣ Payment Method

- Electronic check users churn at significantly higher rates

Interpretation:
May reflect lower engagement or more price-sensitive segments.


### 6️⃣ Internet Service Type

- Fiber optic users show elevated churn compared to DSL users

Interpretation:
Could indicate higher expectations or competitive alternatives.


### 7️⃣ Paperless Billing

- Associated with slightly higher churn probability

Interpretation:
Digitally active users may be more responsive to pricing or competitor offers.


## 🎯 Summary of Behavioral Drivers

Churn in this dataset is primarily driven by:

- Contract flexibility
- Short customer tenure
- High pricing exposure
- Low service engagement
- Certain payment behaviors

These insights guide model design and threshold selection decisions.

## 🔁 Model Lifecycle

This project follows a structured ML workflow beyond notebook experimentation.

### 1️⃣ Model Comparison  
Multiple model families (LR, RF, SVM, XGBoost) were evaluated under a unified preprocessing pipeline.  
All experiments were tracked using MLflow, logging hyperparameters and key metrics (AUC, F1, Precision, Recall).  
**XGBoost base model delivered best performance.**



![Model Comparison Experiment](assets/ModelComparison_Experiment.jpg)



### 2️⃣ Imbalance-Aware Training  
Given the 26.5% churn rate, models were trained with class weighting:  
- `scale_pos_weight` (XGBoost)  
- `class_weight='balanced'` (others)  

This improves churn detection by penalizing false negatives more heavily.

### 3️⃣ XGBoost Optimization  
A dedicated hyperparameter search refined:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`

All runs were tracked in MLflow for reproducible comparison.

![XGBoost Best Model](assets/XGBoost_Optimization_Experiment.jpg)

Model selection was primarily based on **maximum F1 score**, rather than AUC alone.

While AUC measures the model’s ranking ability across all possible thresholds, it does not reflect performance at a specific operating point.

In churn prediction, decisions are made using a fixed classification threshold.  
Therefore, business impact depends on:

- **Recall** → capturing actual churners (revenue protection)  
- **Precision** → limiting unnecessary retention cost  

F1 score balances these two factors and directly evaluates performance at the chosen decision boundary.

In contrast:

- A model can have high AUC but poor precision–recall balance at the deployment threshold.
- AUC evaluates ranking quality, not intervention cost tradeoffs.

For this reason, AUC was used to assess overall separability,  
while **maximum F1 under class-weighted training was used as the primary model selection criterion.**

### 4️⃣ Threshold Tuning  
The final decision threshold was optimized to **0.51**, balancing recall (revenue protection) and precision (intervention cost control).

### 5️⃣ Model Registry & Production Export  
The best run was registered in MLflow for version tracking.  
After business validation, the finalized production model can be exported via:

`src/save_best_model.py`

Stored under:

`models/`


### 🔄 Workflow Summary

EDA → Preprocessing → Model Comparison → Optimization → Threshold Tuning → Registry → Production Export
