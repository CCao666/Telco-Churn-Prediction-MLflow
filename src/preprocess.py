import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(file_path):
    """
    Load the Telco Churn dataset and perform basic feature engineering.
    Returns scaled training and testing sets.
    """
    # 1. Load dataset
    df = pd.read_csv(file_path)
    
    # 2. Data Cleaning: Fix 'TotalCharges' column
    # Some values are empty strings " ", causing the column to be read as object.
    # Convert to numeric, coerce errors to NaN, then fill with 0.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # 3. Feature Selection: Drop irrelevant columns
    # customerID is a unique identifier and provides no predictive power.
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    # 4. Label Encoding: Handle binary categorical features
    # Transform 'Yes'/'No', 'Gender' etc. into 0 and 1.
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])
            
    # 5. One-Hot Encoding: Handle multi-class categorical features
    # Create dummy variables for features like 'PaymentMethod'.
    # drop_first=True helps avoid multi-collinearity (Dummy Variable Trap).
    df = pd.get_dummies(df, drop_first=True)
    
    # 6. Split Features and Target
    # 'Churn' is our target label (binary classification).
    if 'Churn' not in df.columns:
        # After get_dummies, if Churn was object, it stays 'Churn' if binary encoded early
        # or becomes 'Churn_Yes' if handled by get_dummies. 
        # But since we LabelEncoded it in step 4, it remains 'Churn'.
        pass
        
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 7. Dataset Splitting
    # 80% for training, 20% for evaluation. 
    # stratify=y ensures the same churn/non-churn ratio in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 8. Feature Scaling
    # Standardize features by removing the mean and scaling to unit variance.
    # Crucial for models sensitive to feature magnitudes.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Local unit test
    try:
        # Using relative path from project root
        X_train, X_test, y_train, y_test = load_and_preprocess('data/raw_data.csv')
        print(f"Preprocessing successful!")
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
    except FileNotFoundError:
        print("Error: 'data/raw_data.csv' not found. Please ensure the dataset is in the data folder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")