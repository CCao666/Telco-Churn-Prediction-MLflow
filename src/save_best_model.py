import joblib
import json
import os
import pandas as pd 
from xgboost import XGBClassifier
from preprocess import load_and_preprocess

def save_model():

    X_train, X_test, y_train, y_test = load_and_preprocess("data/raw_data.csv")


    raw_df = pd.read_csv("data/raw_data.csv")
    

    feature_names = [f"f{i}" for i in range(X_train.shape[1])] 

    weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        scale_pos_weight=weight,
        random_state=42
    )
    model.fit(X_train, y_train)

    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(model, "models/xgb_production.pkl")

    metadata = {
        "model_name": "XGBoost Churn Predictor",
        "best_threshold": 0.51,
        "feature_count": X_train.shape[1],
        "feature_names": feature_names 
    }

    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    print("Model and Metadata saved successfully!")

if __name__ == "__main__":
    save_model()