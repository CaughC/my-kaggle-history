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


def detailed_analysis(df: pd.DataFrame) -> dict:
    # df["Ticketspace"] = df["Ticket"].str.split().str.len()
    # print(df["Ticketspace"].value_counts())

    # all_tickets = df["Ticket"].values
    # all_elements = [
    #     element for ticket in all_tickets for element in ticket.split()
    # ]
    # series = pd.Series(all_elements)
    # print(series.value_counts().head(20))
    
    # # Extracting ticket prefixes and numeric parts
    # df["Ticketab"] = df["Ticket"].str.extract('([A-Za-z]+)', expand=False)
    # df["Ticketnum"] = df["Ticket"].str.extract('(\d+)', expand=False)
    # print(df["Ticketab"].value_counts())
    # print(df["Ticketnum"].value_counts().head(20))

    # Suppose grouped passengers reserve the cabins next to each other.
    df["Cabin"] = df["Cabin"].map(lambda x: x.split()[0] if pd.notnull(x) else x)  
    # Keep only the first cabin if multiple are listed
    df["Cabinab"] = df["Cabin"].str.extract('([A-Za-z]+)', expand=False)
    df["Cabinnum"] = df["Cabin"].str.extract('(\d+)', expand=False)
    print(df["Cabinab"].value_counts())
    print(df["Cabinnum"].value_counts().head(20))
    df["Cabinab"] = df["Cabinab"].fillna("Z")  # Fill NaN with a placeholder
    df["Cabinnum"] = df["Cabinnum"].fillna(0).astype(int)  # Fill NaN with 0 and convert to int

    
    return {}





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
    results = detailed_analysis(df)

