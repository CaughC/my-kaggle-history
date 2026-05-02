import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import LabelEncoder, StandardScaler
import detailed_analysis as da
from kaggle_utils import run_sklearn_models

def main():
    print("Evaluating Classical Models for House Prices...")
    train_path = "n02_houseprice/house-prices-advanced-regression-techniques/train.csv"
    train_df = pd.read_csv(train_path)
    
    # Use all features engineered by detailed_analysis
    X, y = da.load_and_preprocess(train_df, raw_features=None, features=None, target="SalePrice", fill_na=True)
    
    # Label encode all categorical columns for classical models
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Scaling is important for Linear Regression and GPR
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # For Gaussian Process, it can be very slow with many rows.
    # We use a subset or a simple kernel for demonstration if it's too slow.
    kernel = DotProduct() + WhiteKernel()
    
    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(C=1.0, epsilon=0.1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Gaussian Process": GaussianProcessRegressor(kernel=kernel, random_state=42)
    }
    
    # Note: GPR might still be slow but let's try.
    results = run_sklearn_models(models, X_train, y_train, X_val, y_val, task_type="regression")
    
    print("\nSummary of Results (RMSE on log scale):")
    # Sort by RMSE (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])
    for name, res in sorted_results:
        print(f"{name}: RMSE = {res['rmse']:.4f}")

if __name__ == "__main__":
    main()
