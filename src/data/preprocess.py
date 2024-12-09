import pandas as pd
from google.cloud import storage
from sklearn.preprocessing import LabelEncoder, StandardScaler


def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Download a file from GCS to the local container."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}")


import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_data(input_path, output_path, encoder_dir="encoders/", scaler_dir="scaler/"):
    # Load the dataset
    data = pd.read_csv(input_path)

    # Drop unnecessary columns
    if 'Loan_ID' in data.columns:
        data.drop('Loan_ID', axis=1, inplace=True)

    # Fill missing values
    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)

    # Encode categorical variables and save encoders
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder

    # Save encoders
    os.makedirs(encoder_dir, exist_ok=True)
    for col, encoder in encoders.items():
        joblib.dump(encoder, os.path.join(encoder_dir, f"{col}_encoder.pkl"))

    # Scale numerical features and save scaler
    scaler = StandardScaler()
    numerical_cols = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Save scaler
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, 'scaler.pkl'))

    # Save processed data
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")