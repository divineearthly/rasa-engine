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
