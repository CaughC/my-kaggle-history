import pandas as pd
import json
import os
import numpy as np
import argparse


def load_data(file_path: str) -> pd.DataFrame:
    """Load the Titanic dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def basic_analysis(df: pd.DataFrame) -> dict:
    results = {}
    for col in df.columns:
        dtype = df[col].dtype
        col_info = {"dtype": str(dtype)}
        
        if pd.api.types.is_numeric_dtype(dtype):
            col_info.update(
                check_numeric_column(df[col])
            )
        elif pd.api.types.is_string_dtype(dtype):
            col_info.update(
                check_string_column(df[col])
            )
        
        col_info.update(
            check_empty_column(df[col])
        )
        results[col] = col_info
    return results


def check_numeric_column(series: pd.Series):
    # 必要最小限の統計量だけ抽出して辞書化
    stats = series.describe()
    return {
        "Max": stats["max"],
        "Min": stats["min"],
        "Mean": stats["mean"],
        "Std": stats["std"],
    }

def check_string_column(series: pd.Series):
    return {
        "top5": series.value_counts().head(5).to_dict(),
    }

def check_empty_column(series: pd.Series):
    return {
        "nunique": series.nunique(),
        "row_count": len(series),
        "Null": int(series.isnull().sum())
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=os.path.join('titanic', 'train.csv'))
    parser.add_argument(
        "--output_path", 
        default=os.path.join('output', 'results.json'),
        help="Path to save the analysis results as JSON"
    )    
    args = parser.parse_args()


    df = load_data(args.data_path)
    analysis_results = basic_analysis(df)
    if os.path.dirname(args.output_path):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(analysis_results, f, indent=4)

    print("Analysis completed and saved to:", args.output_path)

