# My Kaggle History

This repository documents my journey through various Kaggle competitions, featuring standardized utilities and multiple modeling approaches.

## Classical Solver Explanations

Here is a brief, easy-to-understand explanation of the classical machine learning algorithms used in this repository:

### 1. Multiple Linear Regression
- **What it is:** A basic regression model that tries to find the best-fitting straight line through the data.
- **How it works:** It assumes that the target value is a weighted sum of the input features.

### 2. Ridge & ElasticNet (Regularized Regression)
- **What they are:** Improved versions of Linear Regression that prevent "overfitting" (getting too distracted by noise in the training data).
- **How they work:** They add a "penalty" to the model for having weights that are too large. Ridge uses L2 penalty, while ElasticNet combines L1 and L2.

### 3. Random Forest
- **What it is:** An "ensemble" of many Decision Trees.
- **How it works:** It builds multiple trees on different subsets of the data and averages their results. It's like asking a group of experts for their opinion instead of just one.

### 4. Gradient Boosting
- **What it is:** Another ensemble method that builds trees one by one.
- **How it works:** Each new tree focuses on fixing the mistakes made by the previous ones. This often leads to very high accuracy.

### 5. SVM (Support Vector Machine)
- **What it is:** A model that tries to find the best boundary (gap) between different classes.
- **How it works:** It looks for the "hyperplane" that maximizes the distance to the nearest points of any class.

### 6. Naive Bayes
- **What it is:** A simple probabilistic classifier based on Bayes' Theorem.
- **How it works:** It calculates the probability of a class based on the features, assuming that all features are independent of each other (which is "naive" but often works well!).

### 7. Gaussian Process Regression (GPR)
- **What it is:** A powerful Bayesian regression method.
- **How it works:** Instead of finding one "best" line, it considers all possible functions that fit the data and provides a prediction along with a measure of uncertainty (how "sure" it is).

---

## Getting Started

### Installation
```bash
# Install the common utilities in editable mode
uv add --editable ./kaggle_utils
```

### Competitions
- [n01_titanic](./n01_titanic): Classification (Survived or not)
- [n02_houseprice](./n02_houseprice): Regression (Predicting Sale Price)
