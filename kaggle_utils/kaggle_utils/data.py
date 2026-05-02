import pandas as pd
import json
import os
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def basic_analysis(df: pd.DataFrame) -> dict:
    """Perform basic statistical analysis on a DataFrame."""
    results = {}
    for col in df.columns:
        dtype = df[col].dtype
        col_info = {"dtype": str(dtype)}
        
        if pd.api.types.is_numeric_dtype(dtype):
            col_info.update(
                _check_numeric_column(df[col])
            )
        elif pd.api.types.is_string_dtype(dtype):
            col_info.update(
                _check_string_column(df[col])
            )
        
        col_info.update(
            _check_empty_column(df[col])
        )
        results[col] = col_info
    return results

def _check_numeric_column(series: pd.Series):
    stats = series.describe()
    return {
        "Max": stats["max"],
        "Min": stats["min"],
        "Mean": stats["mean"],
        "Std": stats["std"],
    }

def _check_string_column(series: pd.Series):
    return {
        "top5": series.value_counts().head(5).to_dict(),
    }

def _check_empty_column(series: pd.Series):
    return {
        "nunique": series.nunique(),
        "row_count": len(series),
        "Null": int(series.isnull().sum())
    }

def save_analysis(results: dict, output_path: str):
    """Save analysis results to a JSON file."""
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
