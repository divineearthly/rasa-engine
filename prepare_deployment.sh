#!/bin/bash

# Define the deployment directory name
DEPLOY_DIR="rasa_net_deployment"

# Define the local path to the exported model
LOCAL_EXPORTED_MODEL_PATH="./rasa_net_exported_model"

# Define the local path for requirements.in if you used pip-tools
LOCAL_REQUIREMENTS_IN="./requirements.in"

echo "Preparing deployment directory: $DEPLOY_DIR"

# Create the deployment directory
mkdir -p "$DEPLOY_DIR"

# Copy the exported model to the deployment directory
if [ -d "$LOCAL_EXPORTED_MODEL_PATH" ]; then
    cp -r "$LOCAL_EXPORTED_MODEL_PATH" "$DEPLOY_DIR/"
    echo "Copied model from $LOCAL_EXPORTED_MODEL_PATH to $DEPLOY_DIR/"
else
    echo "Error: Model directory $LOCAL_EXPORTED_MODEL_PATH not found." >&2
    exit 1
fi

# Create requirements.txt inside the deployment directory
# If you used pip-tools, you might run 'pip-compile' here if pip-tools is installed globally
# For simplicity, we'll use a basic list of core dependencies needed for the FastAPI app.
cat > "$DEPLOY_DIR/requirements.txt" << EOF
transformers
datasets
accelerate
bitsandbytes
scikit-learn
torch
fastapi
uvicorn
EOF
echo "Generated $DEPLOY_DIR/requirements.txt"

# Create app.py inside the deployment directory
cat > "$DEPLOY_DIR/app.py" << EOF
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Define the path where the model is copied inside the Docker container
MODEL_LOAD_PATH = "./rasa_net_exported_model"

# Load tokenizer and model globally to avoid reloading on each request
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_LOAD_PATH)

    # Move model to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"Model and tokenizer loaded successfully on device: {device}")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    # In a real production scenario, you might want to log this error and exit
    # For demonstration, we'll proceed, but inference will likely fail.

# Retrieve id_to_label and label_to_id from the model's configuration.
# These are automatically loaded by from_pretrained if saved in config.json.
# This ensures consistency with how the model was trained.
if hasattr(model.config, 'id2label'):
    id_to_label = model.config.id2label
    label_to_id = {v: k for k, v in id_to_label.items()}
    print("Label mappings loaded from model config.")
else:
    # Fallback if id2label is not in config (should not happen if saved correctly)
    # This part should ideally match the rasa_labels defined during training
    rasa_labels = [
        "Shringara", "Hasya", "Karuna", "Raudra", "Vira",
        "Bhayanaka", "Bibhatsa", "Adbhuta", "Shanta"
    ]
    id_to_label = {i: label for i, label in enumerate(rasa_labels)}
    label_to_id = {label: i for i, label in enumerate(rasa_labels)}
    print("Label mappings manually recreated (check model config for best practice).")

# Self-healing confidence threshold
CONFIDENCE_THRESHOLD = 0.5 # You can make this configurable via environment variables too

app = FastAPI()

# Define the request body model
class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_rasa(input: TextInput):
    # Tokenize the input text
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)

    # Move inputs to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits and apply softmax to get probabilities
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)[0] # Get probabilities for the single input

    # Collect all Rasa label probabilities
    rasa_results = []
    for i in range(len(id_to_label)): # Iterate through all possible labels
        rasa_label = id_to_label[i]
        confidence = probabilities[i].item()
        rasa_results.append({"rasa_label": rasa_label, "confidence": confidence})

    # Sort the results by confidence in descending order
    sorted_results = sorted(rasa_results, key=lambda x: x['confidence'], reverse=True)

    if not sorted_results:
        return {
            "text": input.text,
            "analysis": [],
            "status": "Error: Could not obtain Rasa predictions."
        }

    top_prediction = sorted_results[0]
    top_rasa_label = top_prediction['rasa_label']
    top_confidence = top_prediction['confidence']

    status_message = ""
    if top_confidence < CONFIDENCE_THRESHOLD:
        status_message = f"Uncertainty detected (Confidence: {top_confidence:.2f} < Threshold: {CONFIDENCE_THRESHOLD:.2f}). Consider clarifying or asking for more details. Top suggested Rasa: {top_rasa_label}"
    else:
        status_message = f"Confident prediction. Top Rasa: {top_rasa_label} (Confidence: {top_confidence:.2f})"

    return {
        "text": input.text,
        "analysis": sorted_results[:3], # Return top 3 for consistency
        "status": status_message
    }

# This block allows you to run the FastAPI app directly for local testing
# In a Docker container, uvicorn will be started via the CMD instruction
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
echo "Generated $DEPLOY_DIR/app.py"

# Create Dockerfile inside the deployment directory
cat > "$DEPLOY_DIR/Dockerfile" << EOF
# Use a slim Python base image for a smaller image size
FROM python:3.10-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file and install Python dependencies
# We assume requirements.txt has been generated from previous steps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your saved model and tokenizer files into the container
# The 'rasa_net_exported_model' directory contains all necessary model assets
COPY rasa_net_exported_model/ ./rasa_net_exported_model/

# Copy your FastAPI application code
# This 'app.py' file will define your API endpoints and model inference logic
COPY app.py .

# Expose the port on which the FastAPI application will run
EXPOSE 8000

# Command to run the application using uvicorn (an ASGI server)
# 'app:app' refers to the 'app' variable within the 'app.py' file
# '--host 0.0.0.0' makes the server accessible from outside the container
# '--port 8000' specifies the port it listens on
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
echo "Generated $DEPLOY_DIR/Dockerfile"

echo "Deployment file preparation complete in directory: $DEPLOY_DIR"

# Instructions for the user
echo "\nNext steps:"
echo "1. Navigate to the '$DEPLOY_DIR' directory: cd $DEPLOY_DIR"
echo "2. Build the Docker image: docker build -t rasa-net-api ."
echo "3. Run the Docker container (for local testing): docker run -p 8000:8000 rasa-net-api"
echo "4. Deploy to a cloud platform (e.g., Google Cloud Run, Vertex AI Endpoints) by pushing the image to a container registry and deploying it."
