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
