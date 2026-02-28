# 🎯 Project Goal

def build_production_churn_pipeline():

    """
    Objective:
        Develop a production-style churn prediction system
        with structured ML lifecycle and experiment tracking.
    """

    goals = [
        "Identify high-risk customers likely to churn",
        "Handle class imbalance via class weighting",
        "Compare multiple model families (LR, RF, SVM, XGBoost)",
        "Track experiments using MLflow",
        "Select and register the best-performing model",
        "Optimize decision threshold for business tradeoffs"
    ]

    workflow = [
        "EDA",
        "Preprocessing",
        "Multi-model comparison",
        "Hyperparameter tuning",
        "Model selection",
        "Model Registry"
    ]

    final_model = {
        "model_type": "XGBoost",
        "strategy": "class-weighted training (scale_pos_weight)",
        "threshold": 0.51,
        "objective": "Balance churn detection and intervention cost"
    }

    return goals, workflow, final_model
