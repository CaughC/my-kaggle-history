import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import hydra
from omegaconf import DictConfig, OmegaConf
import os

def feature_engineering(df):
    """Apply feature engineering to the Titanic dataset."""
    df = df.copy()
    
    # Extracting ticket prefixes and numeric parts
    df["Ticketab"] = df["Ticket"].str.extract('([A-Za-z]+)', expand=False)
    df["Ticketnum"] = df["Ticket"].str.extract('(\d+)', expand=False)
    df["Ticketnum"] = df["Ticketnum"].fillna(0).astype(int)
    df["Ticketab"] = df["Ticketab"].fillna("NA")

    # Suppose grouped passengers reserve the cabins next to each other.
    df["Cabin"] = df["Cabin"].map(lambda x: x.split()[0] if pd.notnull(x) else x)  
    # Keep only the first cabin if multiple are listed
    df["Cabinab"] = df["Cabin"].str.extract('([A-Za-z]+)', expand=False)
    df["Cabinnum"] = df["Cabin"].str.extract('(\d+)', expand=False)
    df["Cabinab"] = df["Cabinab"].fillna("Z")  # Fill NaN with a placeholder
    df["Cabinnum"] = df["Cabinnum"].fillna(0).astype(int)  # Fill NaN with 0 and convert to int
    
    return df

def load_and_preprocess(df, cfg: DictConfig, is_train=True):
    """General preprocessing including feature engineering and categorical casting."""
    # Start with raw features needed for engineering
    X = df[list(cfg.dataset.raw_features)].copy()
    
    y = None
    if is_train:
        y = df[cfg.dataset.target]
    
    # Apply feature engineering
    X = feature_engineering(X)
    
    # Filter for final features used by the model
    X = X[list(cfg.dataset.features)].copy()
    
    # Handle categorical features explicitly for LightGBM
    for col in cfg.dataset.categorical_features:
        if col in X.columns:
            X[col] = X[col].astype('category')
            
    return X, y

def objective(trial, cfg: DictConfig, X, y):
    """Optuna objective function for hyperparameter optimization using CV."""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000)
    }
    
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
    cv_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        cv_scores.append(accuracy_score(y_val, y_pred))
        
    return np.mean(cv_scores)

def predict_and_save(cfg: DictConfig, model):
    """Predict on test set and save submission."""
    print(f"Loading test data from {cfg.dataset.test_path}...")
    test_df = pd.read_csv(cfg.dataset.test_path)
    
    X_test, _ = load_and_preprocess(test_df, cfg, is_train=False)
    
    print("Generating predictions...")
    test_preds = model.predict(X_test)
    
    submission = test_df[list(cfg.dataset.output_features)].copy()
    submission[cfg.dataset.output_target] = test_preds
    
    output_dir = os.path.dirname(cfg.dataset.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    submission.to_csv(cfg.dataset.output_path, index=False)
    print(f"Submission saved to {cfg.dataset.output_path}")

@hydra.main(version_base=None, config_path="config", config_name="lgbm_opti_config")
def main(cfg: DictConfig):
    print("Starting Optuna optimization with Stratified K-Fold...")
    train_df = pd.read_csv(cfg.dataset.train_path)
    
    X_df, y_df = load_and_preprocess(train_df, cfg, is_train=True)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, cfg, X_df, y_df), n_trials=50)
    
    print(f"\nBest cross-validation accuracy: {study.best_trial.value:.4f}")
    best_params = study.best_trial.params
    best_params['objective'] = 'binary'
    best_params['verbosity'] = -1

    print("\nTraining final model on full dataset with best parameters...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_df, y_df)
    
    predict_and_save(cfg, final_model)

if __name__ == "__main__":
    main()
