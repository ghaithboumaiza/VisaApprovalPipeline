import os
import json
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from google.cloud import storage


def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
    """Upload a local file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"Uploaded {local_file_path} to gs://{bucket_name}/{destination_blob_name}")


def evaluate_and_save(models, X_test, y_test, model_dir="models/", gcs_bucket_name=None, gcs_model_path=None, gcs_metrics_path=None):
    """
    Evaluate multiple models and save the best one locally and to GCS.

    Parameters:
    - models (dict): A dictionary of models to evaluate.
    - X_test (pd.DataFrame): Test features.
    - y_test (pd.Series): Test labels.
    - model_dir (str): Local directory to save models.
    - gcs_bucket_name (str): GCS bucket name for storage.
    - gcs_model_path (str): Path in GCS to save the best model.
    - gcs_metrics_path (str): Path in GCS to save model metrics.

    Returns:
    - best_model_name (str): Name of the best model.
    - best_model (object): Best model object.
    """
    best_model_name = None
    best_model = None
    best_score = -float("inf")
    best_metrics = {}

    print("Evaluating models...")
    for model_name, model in models.items():
        # Predict on the test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

        print(f"\nModel: {model_name}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc}")

        # Use accuracy as the primary metric for comparison
        if acc > best_score:
            best_score = acc
            best_model_name = model_name
            best_model = model
            best_metrics = {
                "model_name": model_name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "auc": auc,
            }

    # Save the best model locally
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_model_path = os.path.join(model_dir, f"{best_model_name}.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"\nBest Model: {best_model_name} saved locally to {best_model_path}")

    # Save the best model and metrics to GCS
    if gcs_bucket_name:
        if gcs_model_path:
            upload_to_gcs(best_model_path, gcs_bucket_name, gcs_model_path)

        if gcs_metrics_path:
            metrics_file_path = os.path.join(model_dir, "best_model_metrics.json")
            with open(metrics_file_path, "w") as f:
                json.dump(best_metrics, f, indent=4)
            upload_to_gcs(metrics_file_path, gcs_bucket_name, gcs_metrics_path)

    return best_model_name, best_model