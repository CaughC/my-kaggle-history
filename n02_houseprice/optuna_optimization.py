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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import hydra
from omegaconf import DictConfig, OmegaConf
import detailed_analysis as da

def objective(trial, cfg: DictConfig, X, y):
    """Optuna objective function for hyperparameter optimization using CV."""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': cfg.random_state,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 15)
    }
    
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
    cv_scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        y_pred = model.predict(X_val)
        cv_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        
    return np.mean(cv_scores)

def predict_and_save(cfg: DictConfig, model, encoder):
    """Predict on test set and save submission."""
    print(f"Loading test data from {cfg.dataset.test_path}...")
    test_df = pd.read_csv(cfg.dataset.test_path)
    
    X_test, _ = da.load_and_preprocess(
        test_df, 
        cfg.dataset.raw_features, 
        cfg.dataset.features, 
        fill_na=True
    )
    
    # Consistent categorical encoding
    cat_cols = list(cfg.dataset.categorical_features)
    # Filter only available columns
    cat_cols = [c for c in cat_cols if c in X_test.columns]
    
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])
    for col in cat_cols:
        X_test[col] = X_test[col].astype('category')
    
    print("Generating predictions...")
    test_preds_log = model.predict(X_test)
    test_preds = np.expm1(test_preds_log)
    
    submission = test_df[list(cfg.dataset.output_features)].copy()
    submission[cfg.dataset.output_target] = test_preds
    
    output_dir = os.path.dirname(cfg.dataset.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    submission.to_csv(cfg.dataset.output_path, index=False)
    print(f"Submission saved to {cfg.dataset.output_path}")

@hydra.main(version_base=None, config_path="config", config_name="lgbm_opti_config")
def main(cfg: DictConfig):
    print("Starting Optuna optimization with K-Fold...")
    train_df = pd.read_csv(cfg.dataset.train_path)
    
    X_df, y_df = da.load_and_preprocess(
        train_df, 
        cfg.dataset.raw_features, 
        cfg.dataset.features, 
        cfg.dataset.target, 
        fill_na=True
    )
    
    # Consistent categorical encoding
    cat_cols = list(cfg.dataset.categorical_features)
    cat_cols = [c for c in cat_cols if c in X_df.columns]
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_df[cat_cols] = encoder.fit_transform(X_df[cat_cols])
    
    # Cast to category for LightGBM
    for col in cat_cols:
        X_df[col] = X_df[col].astype('category')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, cfg, X_df, y_df), n_trials=30)
    
    print(f"\nBest cross-validation RMSE: {study.best_trial.value:.4f}")
    best_params = study.best_trial.params
    best_params['objective'] = 'regression'
    best_params['metric'] = 'rmse'
    best_params['verbosity'] = -1
    best_params['random_state'] = cfg.random_state

    print("\nTraining final model on full dataset with best parameters...")
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_df, y_df)

    predict_and_save(cfg, final_model, encoder)

if __name__ == "__main__":
    main()
