# Kaggle Utils

Common utilities for Kaggle competitions, providing standardized data analysis, model architectures, and training loops.

## Structure

- `data.py`: Data loading and basic statistical analysis.
- `models.py`: Reusable model architectures (e.g., `TabularMLP`).
- `training.py`: Standardized training and evaluation loops for both Neural Networks and Gradient Boosting models.

## Usage

### Installation

This package is designed to be installed in editable mode within the workspace:

```bash
uv add --editable ./kaggle_utils
```

### Examples

#### Basic Analysis
```python
from kaggle_utils import load_data, basic_analysis, save_analysis

df = load_data("train.csv")
results = basic_analysis(df)
save_analysis(results, "results.json")
```

#### Neural Network Training
```python
from kaggle_utils import TabularMLP, train_nn_epoch, evaluate_nn

model = TabularMLP(input_dim=10, layer_sizes=[64, 32], activation="sigmoid")
# ... setup optimizer, criterion, loaders ...
loss = train_nn_epoch(model, train_loader, criterion, optimizer, scheduler, device)
accuracy = evaluate_nn(model, val_loader, device, task_type="classification")
```
