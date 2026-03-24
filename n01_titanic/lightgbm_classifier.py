import os
import sys

# Ensure the script's directory is in the path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import detailed_analysis as da

def train_model(cfg: DictConfig, X, y):
    """Train model using parameters from Hydra configuration."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=cfg.test_size, 
        random_state=cfg.random_state
    )
    
    # Initialize classifier with parameters from config
    params = OmegaConf.to_container(cfg.model.params, resolve=True)
    clf = lgb.LGBMClassifier(**params)
    
    print(f"Training LightGBM with params: {params}")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    return clf

def predict_and_save(cfg: DictConfig, model):
    """Predict on test set and save submission."""
    print(f"Loading test data from {cfg.dataset.test_path}...")
    test_df = pd.read_csv(cfg.dataset.test_path)
    
    # Preprocess test data
    X_test, _ = da.load_and_preprocess(
        test_df, 
        cfg.dataset.raw_features, 
        cfg.dataset.features, 
        fill_na=False
    )
    
    # Handle categorical features
    for col in cfg.dataset.categorical_features:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')
    
    # Make predictions
    print("Generating predictions...")
    test_preds = model.predict(X_test)
    
    # Create submission dataframe
    submission = test_df[list(cfg.dataset.output_features)].copy()
    submission[cfg.dataset.output_target] = test_preds
    
    # Ensure output directory exists
    output_dir = os.path.dirname(cfg.dataset.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    submission.to_csv(cfg.dataset.output_path, index=False)
    print(f"Submission saved to {cfg.dataset.output_path}")

@hydra.main(version_base=None, config_path="config", config_name="lgbm_config")
def main(cfg: DictConfig):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    try:
        # Load training data
        train_df = pd.read_csv(cfg.dataset.train_path)
        X, y = da.load_and_preprocess(
            train_df, 
            cfg.dataset.raw_features, 
            cfg.dataset.features, 
            cfg.dataset.target, 
            fill_na=False
        )
        
        # Handle categorical features
        for col in cfg.dataset.categorical_features:
            if col in X.columns:
                X[col] = X[col].astype('category')
        
        # Train
        model = train_model(cfg, X, y)
        
        # Predict
        predict_and_save(cfg, model)
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
