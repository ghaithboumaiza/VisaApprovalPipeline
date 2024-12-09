import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(input_path, model_path):
    # Load the processed dataset
    data = pd.read_csv(input_path)

    # Split into features and target
    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # Save the model
    import joblib
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")