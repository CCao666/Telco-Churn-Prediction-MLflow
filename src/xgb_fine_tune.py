import subprocess

def fine_tune():

    
    n_estimators_list = [50, 100, 150, 200]      
    max_depth_list = [3, 5, 7, 9]          
    learning_rates = [0.05, 0.1]       
    subsamples = [0.8, 1.0]             

    print("Starting Comprehensive XGBoost Fine-Tuning...")
    print("Experiment Name: XGBoost_Optimization")

    count = 0
    for n_est in n_estimators_list:
        for d in max_depth_list:
            for lr in learning_rates:
                for s in subsamples:
                    count += 1
                    print(f"--- Running Job {count}: n_est={n_est}, depth={d}, lr={lr}, sub={s} ---")
                    
                    cmd = [
                        "python", "src/train.py",
                        "--model_type", "xgboost",
                        "--experiment_name", "XGBoost_Optimization",
                        "--param1", str(n_est),   
                        "--param2", str(d),       
                        "--learning_rate", str(lr),
                        "--subsample", str(s)
                    ]
                    subprocess.run(cmd)

    print(f"✅ Fine-tuning complete! Total {count} runs logged to 'XGBoost_Optimization'.")

if __name__ == "__main__":
    fine_tune()