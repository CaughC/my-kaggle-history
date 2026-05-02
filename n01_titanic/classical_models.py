import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import detailed_analysis as da
from kaggle_utils import run_sklearn_models

def main():
    print("Evaluating Classical Models for Titanic...")
    train_path = "n01_titanic/titanic/train.csv"
    train_df = pd.read_csv(train_path)
    
    # Must include all raw columns needed for engineering
    raw_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Name", "Ticket", "Cabin"]
    # Final features to use
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", "FamilySize", "IsAlone", "Ticketab", "Cabinab"]
    target = "Survived"
    
    X, y = da.load_and_preprocess(train_df, raw_features, features, target)
    
    # Classical models usually need numeric input
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Naive Bayes": GaussianNB()
    }
    
    results = run_sklearn_models(models, X_train, y_train, X_val, y_val, task_type="classification")
    
    print("\nSummary of Results:")
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for name, res in sorted_results:
        print(f"{name}: Accuracy = {res['accuracy']:.4f}")

if __name__ == "__main__":
    main()
