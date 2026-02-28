import mlflow
from mlflow.tracking import MlflowClient

def register_best_model():
    # 1. Initialize the MLflow Client for advanced metadata management
    client = MlflowClient()
    
    # 2. Define the experiments to search through
    # We include both the initial screening and the fine-tuning experiments
    target_experiments = ["Model_Comparison", "XGBoost_Optimization"]
    
    # Fetch experiment IDs based on names, ignoring those that don't exist yet
    experiment_ids = []
    for name in target_experiments:
        exp = client.get_experiment_by_name(name)
        if exp:
            experiment_ids.append(exp.experiment_id)
    
    if not experiment_ids:
        print("Error: No experiments found. Please run hpo.py or xgb_fine_tune.py first.")
        return

    # 3. Search for the 'Global Champion' run across all identified experiments
    # We order by auc_roc in descending order and pick the top result
    print(f"🔍 Searching for the best model in experiments: {target_experiments}...")
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        max_results=1,
        order_by=["metrics.auc_roc DESC"]
    )
    
    if not runs:
        print("No successful runs found in the specified experiments.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_auc = best_run.data.metrics.get('auc_roc', 0)
    origin_exp_id = best_run.info.experiment_id
    
    # Display the best result found
    print(f"🏆 Best Run Found!")
    print(f"   - Run ID: {best_run_id}")
    print(f"   - Experiment ID: {origin_exp_id}")
    print(f"   - AUC: {best_auc:.4f}")

    # 4. Register the best model to the Model Registry
    # Using a unified name ensures different experiments contribute to the same model lineage
    model_name = "Telco_Churn_Production_Model"
    model_uri = f"runs:/{best_run_id}/model"
    
    print(f"Registering model as '{model_name}'...")
    result = mlflow.register_model(model_uri, model_name)
    
    # 5. Set an alias for the new version
    # 'candidate' signals that this version is ready for evaluation or shadow deployment
    client.set_registered_model_alias(
        name=model_name,
        alias="candidate",
        version=result.version
    )
    
    print(f"Success!")
    print(f"   - Model Name: {model_name}")
    print(f"   - Version: {result.version}")
    print(f"   - Alias: 'candidate'")

if __name__ == "__main__":
    register_best_model()