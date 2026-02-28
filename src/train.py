import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
from preprocess import load_and_preprocess

def eval_metrics(actual, pred, pred_proba):
    acc = accuracy_score(actual, pred)
    prec = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    auc = roc_auc_score(actual, pred_proba)
    return acc, prec, recall, f1, auc

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="xgboost", choices=["xgboost", "rf", "lr", "svm"])
    parser.add_argument("--param1", type=float, default=100) 
    parser.add_argument("--param2", type=float, default=3)   
    parser.add_argument("--experiment_name", type=str, default="Model_Comparison")
    parser.add_argument("--learning_rate", type=float, default=0.1) 
    parser.add_argument("--subsample", type=float, default=1.0)     
    
    args = parser.parse_args()
    mlflow.set_experiment(args.experiment_name)

    X_train, X_test, y_train, y_test = load_and_preprocess('data/raw_data.csv')

    with mlflow.start_run(run_name=f"Model_{args.model_type}"):
        # 2. Model Dispatching Logic
        if args.model_type == "xgboost":
            weight = (y_train == 0).sum() / (y_train == 1).sum()
            model = XGBClassifier(
                n_estimators=int(args.param1),    
                max_depth=int(args.param2), 
                scale_pos_weight=weight,      
                learning_rate=args.learning_rate, 
                subsample=args.subsample,         
                random_state=42
            )
        elif args.model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=int(args.param1), 
                max_depth=int(args.param2),
                class_weight='balanced', 
                random_state=42
            )
        elif args.model_type == "lr":
            model = LogisticRegression(
                C=args.param1, 
                max_iter=int(args.param2),
                class_weight='balanced', 
                random_state=42
            )
        elif args.model_type == "svm":
            model = SVC(
                C=args.param1, 
                kernel='rbf',
                probability=True,
                class_weight='balanced', 
                random_state=42
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        (acc, prec, rec, f1, auc) = eval_metrics(y_test, y_pred, y_pred_proba)

        # Log parameters (records all hyperparameters)
        mlflow.log_param("model_family", args.model_type)
        mlflow.log_param("param1_val", args.param1)
        mlflow.log_param("param2_val", args.param2)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("subsample", args.subsample)
        
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(model, "model")
        print(f"{args.model_type} in {args.experiment_name} - AUC: {auc:.4f}")

if __name__ == "__main__":
    train()