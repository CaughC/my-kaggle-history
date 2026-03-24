import os
import sys

# Ensure the script's directory is in the path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
import hydra
from omegaconf import DictConfig, OmegaConf
import detailed_analysis as da

def objective(trial, cfg: DictConfig, X, y):
    """Optuna objective function for hyperparameter optimization using CV."""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': cfg.random_state,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 64),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15)
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

def predict_and_save(cfg: DictConfig, model, encoder):
    """Predict on test set and save submission."""
    print(f"Loading test data from {cfg.dataset.test_path}...")
    test_df = pd.read_csv(cfg.dataset.test_path)
    
    X_test, _ = da.load_and_preprocess(
        test_df, 
        cfg.dataset.raw_features, 
        cfg.dataset.features, 
        fill_na=False
    )
    
    # Consistent categorical encoding
    cat_cols = list(cfg.dataset.categorical_features)
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])
    for col in cat_cols:
        X_test[col] = X_test[col].astype('category')
    
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
    
    X_df, y_df = da.load_and_preprocess(
        train_df, 
        cfg.dataset.raw_features, 
        cfg.dataset.features, 
        cfg.dataset.target, 
        fill_na=False
    )
    
    # Consistent categorical encoding
    cat_cols = list(cfg.dataset.categorical_features)
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_df[cat_cols] = encoder.fit_transform(X_df[cat_cols])
    
    # Cast to category for LightGBM efficiency and to ensure it treats them as categorical
    for col in cat_cols:
        X_df[col] = X_df[col].astype('category')
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, cfg, X_df, y_df), n_trials=50)
    
    print(f"\nBest cross-validation accuracy: {study.best_trial.value:.4f}")
    best_params = study.best_trial.params
    best_params['objective'] = 'binary'
    best_params['verbosity'] = -1
    best_params['random_state'] = cfg.random_state

    print("\nTraining final model on full dataset with best parameters...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_df, y_df)

    predict_and_save(cfg, final_model, encoder)

if __name__ == "__main__":
    main()
