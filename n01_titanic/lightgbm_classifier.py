import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def load_and_preprocess(file_path):
    """Load and perform minimal preprocessing for LightGBM."""
    df = pd.read_csv(file_path)
    
    # Simple feature selection
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features].copy()
    y = df['Survived']
    
    # LightGBM can handle categorical features if they are set to 'category' dtype
    X['Sex'] = X['Sex'].astype('category')
    X['Embarked'] = X['Embarked'].astype('category')
    
    return X, y

def train_model(X, y):
    """Split data and train a LightGBM classifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the LightGBM dataset
    # Note: LightGBM handles missing values (NaN) automatically
    clf = lgb.LGBMClassifier(verbosity=-1)
    
    print("Training LightGBM model...")
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    return clf

if __name__ == "__main__":
    data_path = os.path.join('n01_titanic', 'titanic', 'train.csv')
    
    try:
        X, y = load_and_preprocess(data_path)
        model = train_model(X, y)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Ensure 'scikit-learn' is installed for train_test_split and accuracy_score.")
