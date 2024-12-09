import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

def load_data(input_path):
    data = pd.read_csv(input_path)
    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def tune_xgboost(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9]
    }
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    search = RandomizedSearchCV(xgb, param_dist, scoring='accuracy', cv=3, n_iter=10, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_

def tune_catboost(X_train, y_train):
    param_dist = {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5]
    }
    catboost = CatBoostClassifier(verbose=0, random_state=42)
    search = RandomizedSearchCV(catboost, param_dist, scoring='accuracy', cv=3, n_iter=10, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_

def tune_random_forest(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(rf, param_dist, scoring='accuracy', cv=3, n_iter=10, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_

if __name__ == "__main__":
    input_path = "data/processed/processed_data.csv"
    X_train, X_test, y_train, y_test = load_data(input_path)

    print("Tuning XGBoost...")
    best_xgb = tune_xgboost(X_train, y_train)
    print("Best XGBoost model:", best_xgb)

    print("Tuning CatBoost...")
    best_catboost = tune_catboost(X_train, y_train)
    print("Best CatBoost model:", best_catboost)

    print("Tuning Random Forest...")
    best_rf = tune_random_forest(X_train, y_train)
    print("Best Random Forest model:", best_rf)

    # Evaluate the best models
    for model_name, model in zip(['XGBoost', 'CatBoost', 'Random Forest'], [best_xgb, best_catboost, best_rf]):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {acc * 100:.2f}%")