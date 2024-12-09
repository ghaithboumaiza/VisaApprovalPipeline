from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import time
import os
from src.data.preprocess import preprocess_data
from src.models.train import train_model
from src.models.tune_hyperparams import tune_xgboost, tune_catboost, tune_random_forest
from src.models.evaluate import evaluate_and_save
from google.cloud import storage

def upload_directory_to_gcs(local_dir, bucket_name, gcs_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            gcs_path = os.path.join(gcs_dir, relative_path)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")


# Dummy HTTP server to satisfy Cloud Run's port requirement
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


def start_http_server():
    """Start a simple HTTP server for health checks."""
    port = int(os.getenv("PORT", 8080))  # Default Cloud Run port is 8080
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    print(f"Starting HTTP server on port {port}...")
    server.serve_forever()


def run_pipeline():
    """Main logic for running the ML pipeline."""
    # Paths
    raw_data_path = "gs://loanapprovalprediction/data/raw/LoanApprovalPrediction.csv"
    processed_data_path = "/tmp/processed_data.csv"  # Temp location in the container
    model_dir = "models/"  # Local directory for saving models
    gcs_bucket_name = "loanapprovalprediction"  # Your GCS bucket
    gcs_model_path = "models/best_model.pkl"  # Path in GCS for the best model
    gcs_metrics_path = "metrics/best_model_metrics.json"  # Path in GCS for model metrics

    # Step 1: Preprocessing
    print("Step 1: Preprocessing data...")
    encoder_dir = "encoders/"
    scaler_dir = "scaler/"
    preprocess_data(raw_data_path, processed_data_path, encoder_dir, scaler_dir)

    upload_directory_to_gcs(encoder_dir, gcs_bucket_name, 'encoders/')
    upload_directory_to_gcs(scaler_dir, gcs_bucket_name, 'scaler/')
    
    # Load data
    from src.models.tune_hyperparams import load_data
    X_train, X_test, y_train, y_test = load_data(processed_data_path)

    # Step 2: Hyperparameter tuning
    print("Step 2: Hyperparameter tuning for XGBoost...")
    best_xgb = tune_xgboost(X_train, y_train)

    print("Step 3: Hyperparameter tuning for CatBoost...")
    best_catboost = tune_catboost(X_train, y_train)

    print("Step 4: Hyperparameter tuning for Random Forest...")
    best_rf = tune_random_forest(X_train, y_train)

    # Step 5: Evaluate and save the best model
    models = {
        "XGBoost": best_xgb,
        "CatBoost": best_catboost,
        "Random Forest": best_rf,
    }
    print("Step 5: Evaluating models and saving the best one...")
    best_model_name, best_model = evaluate_and_save(
        models, 
        X_test, 
        y_test, 
        model_dir, 
        gcs_bucket_name, 
        gcs_model_path, 
        gcs_metrics_path
    )

    print(f"Pipeline completed successfully! Best Model: {best_model_name}")


if __name__ == "__main__":
    # Start the HTTP server in a separate thread
    server_thread = threading.Thread(target=start_http_server)
    server_thread.daemon = True
    server_thread.start()

    # Run the ML pipeline
    run_pipeline()