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
