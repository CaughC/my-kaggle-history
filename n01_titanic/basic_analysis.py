import os
import argparse
from kaggle_utils import load_data, basic_analysis, save_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=os.path.join('n01_titanic', 'titanic', 'train.csv'))
    parser.add_argument(
        "--output_path", 
        default=os.path.join('n01_titanic', 'output', 'results.json'),
        help="Path to save the analysis results as JSON"
    )    
    args = parser.parse_args()

    df = load_data(args.data_path)
    analysis_results = basic_analysis(df)
    save_analysis(analysis_results, args.output_path)

    print("Analysis completed and saved to:", args.output_path)
