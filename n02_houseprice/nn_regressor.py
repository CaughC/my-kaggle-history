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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import detailed_analysis as da

class HousePriceMLP(nn.Module):
    def __init__(self, input_dim, layer_sizes, dropout=0.2):
        super(HousePriceMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = size
        layers.append(nn.Linear(prev_dim, 1))
        # No sigmoid for regression
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.cpu().numpy())
            targets.extend(y.numpy())
    return np.sqrt(mean_squared_error(targets, preds))

@hydra.main(version_base=None, config_path="config", config_name="nn_config")
def main(cfg: DictConfig):
    print("Starting Neural Network training with K-Fold and full features...")
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    print(f"Using device: {device}")
    
    train_df = pd.read_csv(cfg.dataset.train_path)
    test_df = pd.read_csv(cfg.dataset.test_path)
    
    # Feature Engineering and selection using outsourced logic
    X, y = da.load_and_preprocess(
        train_df, 
        cfg.dataset.raw_features, 
        cfg.dataset.features, 
        cfg.dataset.target,
        fill_na=True
    )
    X_test, _ = da.load_and_preprocess(
        test_df, 
        cfg.dataset.raw_features, 
        cfg.dataset.features,
        fill_na=True
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
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
    cv_scores = []
    
    # Pre-fit on full train for feature space consistency
    preprocessor.fit(X)
    input_dim = preprocessor.transform(X).shape[1]
    
    final_test_preds = np.zeros(len(test_df))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
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
        model = HousePriceMLP(input_dim, cfg.model.layers, cfg.model.dropout).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.model.learning_rate, weight_decay=cfg.model.weight_decay)
        criterion = nn.MSELoss()
        
        total_steps = len(train_loader) * cfg.model.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
        
        best_val_rmse = float('inf')
        for epoch in range(cfg.model.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            if (epoch + 1) % 50 == 0:
                val_rmse = evaluate(model, val_loader, device)
                print(f"Fold {fold}, Epoch {epoch+1}: Val RMSE = {val_rmse:.4f}")
        
        val_rmse = evaluate(model, val_loader, device)
        print(f"Fold {fold}: Final RMSE = {val_rmse:.4f}")
        cv_scores.append(val_rmse)
        
        # Predict on test set for ensemble average
        X_test_t = torch.FloatTensor(np.asarray(preprocessor.transform(X_test))).to(device)
        model.eval()
        with torch.no_grad():
            final_test_preds += (model(X_test_t).cpu().numpy().flatten() / cfg.n_folds)
            
    print(f"\nMean CV RMSE: {np.mean(cv_scores):.4f}")
    
    # Generate final submission
    test_preds = np.expm1(final_test_preds)
    submission = test_df[list(cfg.dataset.output_features)].copy()
    submission[cfg.dataset.output_target] = test_preds
    
    output_dir = os.path.dirname(cfg.dataset.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    submission.to_csv(cfg.dataset.output_path, index=False)
    print(f"Submission saved to {cfg.dataset.output_path}")

if __name__ == "__main__":
    main()
