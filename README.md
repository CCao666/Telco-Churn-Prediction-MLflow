# Telco Customer Churn Prediction 🚀

This project implements an end-to-end Machine Learning pipeline to predict customer churn for a telecommunications company. By addressing severe class imbalance and optimizing decision thresholds, the final model achieves high recall, making it a valuable tool for proactive customer retention.

## 📊 Performance Summary
- **Model**: XGBoost (Weighted)
- **AUC-ROC**: 0.845
- **F1-Score**: 0.637
- **Optimal Threshold**: 0.51

## 🔍 Key Insights from EDA
* **Contract Risk**: Customers on "Month-to-month" contracts are at the highest risk of churning.
* **Newbie Crisis**: High churn rates are observed within the first 12 months of tenure.
* **Service Stickiness**: Users with "Tech Support" and "Online Security" show significantly higher loyalty.
* **Imbalance Strategy**: Since only 26.5% of the data represents churn, we utilized `scale_pos_weight=2.77` in XGBoost to penalize missing churners.



## 🛠️ Tech Stack
- **Languages**: Python 3.10
- **Libraries**: XGBoost, Scikit-Learn, Pandas, Plotly, Seaborn
- **Experiment Tracking**: MLflow (for hyperparameter tuning and model versioning)

## 📈 Model Diagnostics
We prioritized **Recall (81%)** over pure Accuracy to ensure that the business captures the majority of customers at risk. The decision threshold was tuned to 0.51 to balance marketing costs (False Positives) against the cost of churn (False Negatives).



## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run EDA: Open `notebooks/EDA.ipynb`
3. Train model: `python src/train.py`