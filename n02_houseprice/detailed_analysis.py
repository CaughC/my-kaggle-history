import json
import os
import numpy as np
import argparse
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load the House Price dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def any_analysis(df: pd.DataFrame) -> dict:
    """Do any analysis you want and see the results. This is just a placeholder function."""
    print(df.describe())
    return {}


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values for numerical and standard categorical columns."""
    df = df.copy()
    
    # Categorical features where NA means "None"
    none_cols = [
        'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType', 
        'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
    ]
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
            
    # Numerical features
    if 'LotFrontage' in df.columns:
        # Fill LotFrontage with median of its neighborhood
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
        
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
        
    # Other numerical columns with small number of missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    # Other categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
            
    return df

def apply_feature_engineering(df: pd.DataFrame, fill_na: bool = True) -> pd.DataFrame:
    """Apply the full feature engineering pipeline."""
    df = df.copy()
    if fill_na:
        df = fill_missing_values(df)
    
    # Create new features
    if '1stFlrSF' in df.columns and '2ndFlrSF' in df.columns and 'TotalBsmtSF' in df.columns:
        df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
        
    if 'FullBath' in df.columns and 'HalfBath' in df.columns and 'BsmtFullBath' in df.columns and 'BsmtHalfBath' in df.columns:
        df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
        
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        
    if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        
    if 'YrSold' in df.columns and 'GarageYrBlt' in df.columns:
        df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']

    # Total Porch SF
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']
    available_porch = [c for c in porch_cols if c in df.columns]
    if available_porch:
        df['TotalPorchSF'] = df[available_porch].sum(axis=1)

    return df

def load_and_preprocess(df: pd.DataFrame, raw_features: list, features: list, target: str = None, fill_na: bool = True):
    """General preprocessing including feature engineering and feature selection."""
    # Start with raw features needed for engineering
    # If raw_features is empty or None, use all available columns except target
    if not raw_features:
        cols_to_use = [c for c in df.columns if c != target]
    else:
        cols_to_use = list(raw_features)
    
    X = df[cols_to_use].copy()
    
    y = None
    if target and target in df.columns:
        # Use log transformation for SalePrice as it's typically right-skewed
        y = np.log1p(df[target])
    
    # Apply feature engineering
    X = apply_feature_engineering(X, fill_na=fill_na)
    
    # Filter for final features used by the model
    # If features list is empty, use all columns from X
    if features:
        available_features = [f for f in features if f in X.columns]
        X = X[available_features].copy()
            
    return X, y




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=os.path.join('n02_houseprice', 'house-prices-advanced-regression-techniques', 'train.csv'))
    parser.add_argument(
        "--output_path", 
        default=os.path.join('n02_houseprice', 'output', 'results.json'),
        help="Path to save the analysis results as JSON"
    )    
    args = parser.parse_args()

    df = load_data(args.data_path)
    results = any_analysis(df)
