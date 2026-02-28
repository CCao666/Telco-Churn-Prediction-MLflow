import subprocess

def run_comprehensive_hpo():
    # define grid search for each model
    # model_type: [ (param1, param2), ... ]
    search_grids = {
        "lr": [(0.1, 100), (1.0, 500), (10.0, 1000)],           # C, max_iter
        "rf": [(50, 5), (100, 10), (200, 15)],                  # n_estimators, max_depth
        "xgboost": [(100, 3), (200, 5), (300, 7)],              # n_estimators, max_depth
        "svm": [(0.1, 1), (1.0, 1), (10.0, 1)]                  # C, unused_p2
    }

    print("Starting Multi-Model Hyperparameter Sweep...")

    for model_type, params in search_grids.items():
        for p1, p2 in params:
            print(f"Running {model_type} with p1={p1}, p2={p2}...")
            cmd = [
                "python", "src/train.py",
                "--model_type", model_type,
                "--param1", str(p1),
                "--param2", str(p2)
            ]
            subprocess.run(cmd)

    print("Comprehensive Sweep Finished! Open MLflow UI to compare.")

if __name__ == "__main__":
    run_comprehensive_hpo()