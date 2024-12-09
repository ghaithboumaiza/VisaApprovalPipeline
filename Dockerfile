# Base image with Python 3.9
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Set the PYTHONPATH to include the src directory
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

# Copy the project files into the container
COPY . .

# Set Google Cloud credentials environment variable (key injected at runtime)
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/gcs-key.json

# Expose the default Cloud Run port
ENV PORT=8080

# Default command to run when the container starts
CMD ["python", "src/pipeline/run_pipeline.py"]