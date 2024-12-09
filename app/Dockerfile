# Base image with Python 3.9
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Set the PYTHONPATH to include the current directory
ENV PYTHONPATH=/app

# Install system dependencies for XGBoost and CatBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY . .

# Set Google Cloud credentials
COPY loanapprovalprediction-442813-de39e566af07.json /app/gcs-key.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/gcs-key.json

# Expose the default Gradio port
EXPOSE 8080

# Command to run the Gradio app
CMD ["python", "gradio_inference.py"]