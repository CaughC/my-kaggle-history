import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import hydra
from omegaconf import DictConfig, OmegaConf

def load_and_preprocess(df, cfg: DictConfig, is_train=True):
    """General preprocessing based on Hydra configuration."""
    X = df[cfg.dataset.features].copy()
    
    y = None
    if is_train:
        y = df[cfg.dataset.target]
    
    # Handle categorical features
    for col in cfg.dataset.categorical_features:
        if col in X.columns:
            X[col] = X[col].astype('category')
            
    return X, y

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
    X_test, _ = load_and_preprocess(test_df, cfg, is_train=False)
    
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
        X, y = load_and_preprocess(train_df, cfg, is_train=True)
        
        # Train
        model = train_model(cfg, X, y)
        
        # Predict
        predict_and_save(cfg, model)
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
