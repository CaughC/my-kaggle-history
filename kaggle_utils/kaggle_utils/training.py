import torch
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

def train_nn_epoch(model, loader, criterion, optimizer, scheduler, device):
    """Train a neural network for one epoch."""
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        # Handle both classification (BCELoss expects (N, 1)) and regression
        if isinstance(criterion, torch.nn.BCELoss) or isinstance(criterion, torch.nn.MSELoss):
             loss = criterion(out, y.unsqueeze(1))
        else:
             loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_nn(model, loader, device, task_type="classification"):
    """Evaluate a neural network."""
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.cpu().numpy())
            targets.extend(y.numpy())
    
    preds = np.array(preds)
    targets = np.array(targets)
    
    if task_type == "classification":
        preds_binary = (preds > 0.5).astype(int)
        return accuracy_score(targets, preds_binary)
    else:
        return np.sqrt(mean_squared_error(targets, preds))

def run_sklearn_models(models, X_train, y_train, X_val, y_val, task_type="classification"):
    """Train and evaluate multiple sklearn models."""
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        if task_type == "classification":
            score = accuracy_score(y_val, y_pred)
            results[name] = {"accuracy": score, "model": model}
        else:
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            results[name] = {"rmse": score, "model": model}
    return results

def get_lgbm_params(cfg_model_params, task_type="classification"):
    """Convert Hydra model params to LightGBM compatible dict."""
    from omegaconf import OmegaConf
    params = OmegaConf.to_container(cfg_model_params, resolve=True)
    if task_type == "classification" and "objective" not in params:
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"
    elif task_type == "regression" and "objective" not in params:
        params["objective"] = "regression"
        params["metric"] = "rmse"
    return params
