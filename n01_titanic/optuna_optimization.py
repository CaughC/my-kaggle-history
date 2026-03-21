import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import hydra
from omegaconf import DictConfig, OmegaConf
import os

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

def objective(trial, cfg: DictConfig, X, y):
    """Optuna objective function for hyperparameter optimization."""
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
        'n_estimators': cfg.model.params.n_estimators
    }
    
    # Resampling in every trial
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=cfg.test_size, 
        random_state=trial.number
    )
    
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_val)
    return accuracy_score(y_val, y_pred)

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

@hydra.main(version_base=None, config_path="config", config_name="lgbm_config")
def main(cfg: DictConfig):
    print("Starting Optuna optimization...")
    train_df = pd.read_csv(cfg.dataset.train_path)
    X, y = load_and_preprocess(train_df, cfg, is_train=True)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, cfg, X, y), n_trials=50)
    
    print(f"\nBest accuracy: {study.best_trial.value:.4f}")
    best_params = study.best_trial.params
    # Add constant params from config if needed
    best_params['n_estimators'] = cfg.model.params.n_estimators
    best_params['objective'] = 'binary'
    best_params['verbosity'] = -1

    print("\nTraining final model on full dataset with best parameters...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X, y)
    
    predict_and_save(cfg, final_model)

if __name__ == "__main__":
    main()
