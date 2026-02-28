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
│   └── raw_data.csv              # Raw Telco churn dataset
│
├── notebooks/
│   └── EDA.ipynb                 # Exploratory data analysis and insights
│
├── src/
│   ├── preprocess.py             # Data cleaning, encoding, scaling, train/test split
│   ├── train.py                  # Unified training script with MLflow logging
│   ├── hpo.py                    # Multi-model hyperparameter sweep
│   ├── xgb_finetune.py           # Fine-grained XGBoost hyperparameter tuning
│   ├── register_model.py         # Automated best-model registration to MLflow Registry
│   └── save_best_model.py        # Business-driven export of selected production model    
│
├── mlruns/                       # MLflow experiment artifacts (ignored by Git)
├── mlflow.db                     # MLflow tracking database (ignored by Git)
│
├── conda.yaml                    # Reproducible conda environment
├── MLProject                     # MLflow project definition
├── README.md                     # Project documentation
└── .gitignore                    # Excludes artifacts, cache, and local files
