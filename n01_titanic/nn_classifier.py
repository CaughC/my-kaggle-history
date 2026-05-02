import os
import sys

# Ensure the script's directory is in the path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf
import detailed_analysis as da
from kaggle_utils import TabularMLP, train_nn_epoch, evaluate_nn

@hydra.main(version_base=None, config_path="config", config_name="nn_config")
def main(cfg: DictConfig):
    print("Starting Neural Network training with Stratified K-Fold and full features...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_df = pd.read_csv(cfg.dataset.train_path)
    test_df = pd.read_csv(cfg.dataset.test_path)
    
    # Feature Engineering and selection using outsourced logic
    X, y = da.load_and_preprocess(
        train_df, 
        cfg.dataset.raw_features, 
        cfg.dataset.features, 
        cfg.dataset.target
    )
    X_test, _ = da.load_and_preprocess(
        test_df, 
        cfg.dataset.raw_features, 
        cfg.dataset.features
    )
    
    # Identify numeric and categorical features from the final X columns
    features = list(X.columns)
    categorical_features = [f for f in cfg.dataset.categorical_features if f in features]
    numeric_features = [f for f in features if f not in categorical_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # CV Training
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
    cv_scores = []
    
    # Pre-fit on full train for feature space consistency
    preprocessor.fit(X)
    input_dim = preprocessor.transform(X).shape[1]
    
    final_test_preds = np.zeros(len(test_df))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Transform data - ensure dense arrays
        X_train_t = torch.FloatTensor(np.asarray(preprocessor.transform(X_train)))
        X_val_t = torch.FloatTensor(np.asarray(preprocessor.transform(X_val)))
        y_train_t = torch.FloatTensor(y_train.values)
        y_val_t = torch.FloatTensor(y_val.values)
        
        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_ds, batch_size=cfg.model.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.model.batch_size)
        
        # Initialize model
        model = TabularMLP(input_dim, cfg.model.layers, cfg.model.dropout, activation="sigmoid").to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.model.learning_rate, weight_decay=cfg.model.weight_decay)
        criterion = nn.BCELoss()
        
        total_steps = len(train_loader) * cfg.model.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        for epoch in range(cfg.model.epochs):
            train_loss = train_nn_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        
        acc = evaluate_nn(model, val_loader, device, task_type="classification")
        print(f"Fold {fold}: Accuracy = {acc:.4f}")
        cv_scores.append(acc)
        
        # Predict on test set for ensemble average
        X_test_t = torch.FloatTensor(np.asarray(preprocessor.transform(X_test))).to(device)
        with torch.no_grad():
            final_test_preds += (model(X_test_t).cpu().numpy().flatten() / cfg.n_folds)
            
    print(f"\nMean CV Accuracy: {np.mean(cv_scores):.4f}")
