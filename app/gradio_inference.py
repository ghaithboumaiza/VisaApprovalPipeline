import os
import gradio as gr
import joblib
import pandas as pd
from google.cloud import storage
import csv

# Configuration
GCS_BUCKET_NAME = "loanapprovalprediction"  # Your GCS bucket
BEST_MODEL_PATH = "models/best_model.pkl"   # Path to the model in GCS
GCS_ENCODER_DIR = "encoders/"              # Path in GCS for encoders
GCS_SCALER_PATH = "scaler/scaler.pkl"      # Path in GCS for scaler
SAVED_INFERENCES_PATH = "/tmp/saved_inferences.csv"  # Path to save inferences locally
GCS_SAVED_INFERENCES_PATH = "feedback/saved_inferences.csv"  # Path in GCS for saved inferences
LOCAL_MODEL_PATH = "/tmp/best_model.pkl"   # Local path for the model
LOCAL_ENCODER_DIR = "/tmp/encoders/"       # Local directory for encoders
LOCAL_SCALER_PATH = "/tmp/scaler.pkl"      # Local path for scaler

# Ensure the saved inferences file exists
if not os.path.exists(SAVED_INFERENCES_PATH):
    pd.DataFrame(columns=[
        "Gender", "Married", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Credit_History", "Property_Area", "Dependents", "Loan_Amount_Term", "Prediction"
    ]).to_csv(SAVED_INFERENCES_PATH, index=False)

def save_inference(data, prediction):
    """
    Save inference data and prediction to a CSV file.
    """
    header = [
        "Gender", "Married", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Credit_History", "Property_Area", "Dependents", "Loan_Amount_Term", "Prediction"
    ]
    # Append data to CSV
    file_exists = os.path.isfile(SAVED_INFERENCES_PATH)
    with open(SAVED_INFERENCES_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)  # Write header if file doesn't exist
        # Ensure the row matches the header structure
        if len(data) + 1 == len(header):  # +1 for the prediction column
            writer.writerow(data + [prediction])
        else:
            print("Skipping malformed data:", data)

def upload_to_gcs(local_file_path, bucket_name, gcs_file_path):
    """
    Upload a file to Google Cloud Storage.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_file_path)
    blob.upload_from_filename(local_file_path)
    print(f"Uploaded {local_file_path} to gs://{bucket_name}/{gcs_file_path}")

def download_from_gcs(bucket_name, source_path, destination_path):
    """
    Download a file from Google Cloud Storage.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_path)
    blob.download_to_filename(destination_path)
    print(f"Downloaded {source_path} to {destination_path}")
def download_directory_from_gcs(bucket_name, gcs_dir, local_dir):
    """
    Download all files from a GCS directory to a local directory.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_dir)
    os.makedirs(local_dir, exist_ok=True)  # Ensure local directory exists
    for blob in blobs:
        if not blob.name.endswith('/'):  # Ignore directory placeholders
            local_path = os.path.join(local_dir, os.path.relpath(blob.name, gcs_dir))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)  # Ensure subdirectories exist
            blob.download_to_filename(local_path)
            print(f"Downloaded {blob.name} to {local_path}")

def load_resources():
    """
    Download and load the model, encoders, and scaler from GCS.
    """
    # Download model
    download_from_gcs(GCS_BUCKET_NAME, BEST_MODEL_PATH, LOCAL_MODEL_PATH)
    model = joblib.load(LOCAL_MODEL_PATH)
    print("Model loaded.")

    # Download encoders directory
    download_directory_from_gcs(GCS_BUCKET_NAME, GCS_ENCODER_DIR, LOCAL_ENCODER_DIR)
    encoders = {}
    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
        encoder_path = os.path.join(LOCAL_ENCODER_DIR, f"{col}_encoder.pkl")
        encoders[col] = joblib.load(encoder_path)
    print("Encoders loaded.")

    # Download scaler
    download_from_gcs(GCS_BUCKET_NAME, GCS_SCALER_PATH, LOCAL_SCALER_PATH)
    scaler = joblib.load(LOCAL_SCALER_PATH)
    print("Scaler loaded.")

    return model, encoders, scaler

def predict(
    Gender, Married, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History, Property_Area
):
    """
    Perform prediction on user input.
    """
    model, encoders, scaler = load_resources()

    # Create a DataFrame for the user input
    input_data = pd.DataFrame(
        [[Gender, Married, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History, Property_Area]],
        columns=[
            "Gender",
            "Married",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Credit_History",
            "Property_Area",
        ],
    )

    # Add missing features with default values
    input_data["Dependents"] = 0  # Default value
    input_data["Loan_Amount_Term"] = 360.0  # Default value

    # Apply encoders
    categorical_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
    for col in categorical_cols:
        encoder = encoders[col]
        input_data[col] = encoder.transform([input_data[col][0]])

    # Convert numerical inputs to float
    numerical_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Credit_History", "Loan_Amount_Term"]
    input_data[numerical_cols] = input_data[numerical_cols].astype(float)

    # Apply scaler to numerical features
    input_data[["LoanAmount", "ApplicantIncome", "CoapplicantIncome"]] = scaler.transform(
        input_data[["LoanAmount", "ApplicantIncome", "CoapplicantIncome"]]
    )

    # Reorder columns to match model's feature names
    input_data = input_data[model.feature_names_in_]

    # Perform prediction
    prediction = model.predict(input_data)
    result = "Loan Approved" if prediction[0] == 1 else "Loan Denied"

    # Save the input and prediction
    save_inference(
        [Gender, Married, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History, Property_Area, 0, 360.0],
        result,
    )

    return result

def feedback():
    """
    Display and allow modification of saved inferences.
    """
    # Ensure the CSV file exists and is valid
    if os.path.exists(SAVED_INFERENCES_PATH):
        try:
            saved_data = pd.read_csv(SAVED_INFERENCES_PATH)
        except pd.errors.ParserError:
            print("CSV file is corrupted. Resetting file.")
            # Reinitialize the CSV with the correct structure
            saved_data = pd.DataFrame(columns=[
                "Gender", "Married", "Education", "Self_Employed",
                "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Credit_History", "Property_Area", "Dependents", "Loan_Amount_Term", "Prediction"
            ])
            saved_data.to_csv(SAVED_INFERENCES_PATH, index=False)
    else:
        saved_data = pd.DataFrame(columns=[
            "Gender", "Married", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Credit_History", "Property_Area", "Dependents", "Loan_Amount_Term", "Prediction"
        ])
        saved_data.to_csv(SAVED_INFERENCES_PATH, index=False)

    def update_feedback(updated_data):
        """
        Update and save the feedback data to GCS.
        """
        updated_data.to_csv(SAVED_INFERENCES_PATH, index=False)
        upload_to_gcs(SAVED_INFERENCES_PATH, GCS_BUCKET_NAME, GCS_SAVED_INFERENCES_PATH)
        return "Feedback data saved to cloud!"

    # Create the Gradio interface
    return gr.Interface(
        fn=update_feedback,
        inputs=gr.DataFrame(value=saved_data, interactive=True),
        outputs="text",
        title="Feedback",
        description="Modify predictions and save updates to the cloud."
    )

if __name__ == "__main__":
    # Define Prediction Tab
    predict_tab = gr.Interface(
        fn=predict,
        inputs=[
            gr.Dropdown(["Male", "Female"], label="Gender"),
            gr.Dropdown(["Yes", "No"], label="Married"),
            gr.Dropdown(["Graduate", "Not Graduate"], label="Education"),
            gr.Dropdown(["Yes", "No"], label="Self Employed"),
            gr.Number(label="Applicant Income"),
            gr.Number(label="Coapplicant Income"),
            gr.Number(label="Loan Amount"),
            gr.Dropdown([0, 1], label="Credit History"),
            gr.Dropdown(["Urban", "Semiurban", "Rural"], label="Property Area"),
        ],
        outputs="text",
        title="Loan Approval Prediction",
        description="Enter loan details to check if the loan will be approved.",
    )

    # Define Feedback Tab
    feedback_tab = feedback()

    # Launch Gradio with Tabs
    gr.TabbedInterface(
        [predict_tab, feedback_tab],
        ["Prediction", "Feedback"]
    ).launch(server_name="0.0.0.0", server_port=8080)