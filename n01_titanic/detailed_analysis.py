import json
import os
import numpy as np
import argparse
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load the Titanic dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def any_analysis(df: pd.DataFrame) -> dict:
    """Do any analysis you want and see the results. This is just a placeholder function."""
    df["ID"] = df["Cabin"].fillna("NA") + "_" + df["Ticket"].fillna("NA")
    print(df["Cabin"].value_counts().head(5))
    print(len(df["Cabin"].unique()))
    print(df["Ticket"].value_counts().head(5))
    print(len(df["Ticket"].unique()))
    print(df["ID"].value_counts().head(5))
    print(len(df["ID"].unique()))

    return {}


def ticket_analysis(df: pd.DataFrame) -> pd.DataFrame:
    # Extracting ticket prefixes and numeric parts
    df["Ticketab"] = df["Ticket"].str.extract('([A-Za-z]+)', expand=False)
    df["Ticketnum"] = df["Ticket"].str.extract('(\d+)', expand=False)
    df["Ticketab"] = df["Ticketab"].fillna("NA")
    df["Ticketnum"] = df["Ticketnum"].fillna(0).astype(int)
    df["Ticket"] = df["Ticket"].fillna("NA")
    return df

def cabin_analysis(df: pd.DataFrame) -> pd.DataFrame:
    # Suppose grouped passengers reserve the cabins next to each other.
    df["Cabin"] = df["Cabin"].map(lambda x: x.split()[0] if pd.notnull(x) and x != "NA" else x)  
    # Keep only the first cabin if multiple are listed
    df["Cabinab"] = df["Cabin"].str.extract('([A-Za-z]+)', expand=False)
    df["Cabinnum"] = df["Cabin"].str.extract('(\d+)', expand=False)
    df["Cabinab"] = df["Cabinab"].fillna("Z")
    df["Cabinnum"] = df["Cabinnum"].fillna(0).astype(int)
    df["Cabin"] = df["Cabin"].fillna("NA")
    return df

def name_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Perform detailed analysis on the Name column and return the modified DataFrame."""
    # Extract Title from Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].fillna("Unknown")
    
    # Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['FamilySize'] = df['FamilySize'].astype(int)

    # Is Alone
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    df["Family_name"] = df["Name"].str.split(",").str[0]
    df["GroupId"] = df["Family_name"].astype(str) + "_" + df["FamilySize"].astype(str) + "_" + df["Ticket"].fillna("NA").astype(str)
    
    return df

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values for numerical and standard categorical columns."""
    df = df.copy()
    # Fill Age with median
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    # Fill Fare with median
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    # Fill Embarked with mode (S)
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna('S')
    return df

def apply_feature_engineering(df: pd.DataFrame, fill_na: bool = True) -> pd.DataFrame:
    """Apply the full feature engineering pipeline."""
    df = df.copy()
    if fill_na:
        df = fill_missing_values(df)
    
    df = name_analysis(df)
    df = ticket_analysis(df)
    df = cabin_analysis(df)
    
    return df

def load_and_preprocess(df: pd.DataFrame, raw_features: list, features: list, target: str = None, fill_na: bool = True):
    """General preprocessing including feature engineering and feature selection."""
    # Start with raw features needed for engineering
    X = df[list(raw_features)].copy()
    
    y = None
    if target and target in df.columns:
        y = df[target]
    
    # Apply feature engineering
    X = apply_feature_engineering(X, fill_na=fill_na)
    
    # Filter for final features used by the model
    # Note: Only include features that actually exist in X after engineering
    available_features = [f for f in features if f in X.columns]
    X = X[available_features].copy()
            
    return X, y




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=os.path.join('n01_titanic', 'titanic', 'train.csv'))
    parser.add_argument(
        "--output_path", 
        default=os.path.join('output', 'results.json'),
        help="Path to save the analysis results as JSON"
    )    
    args = parser.parse_args()

    df = load_data(args.data_path)
    results = any_analysis(df)

