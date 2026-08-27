#!/bin/bash

# Function to pull the model
pull_model() {
    echo "Pulling model ${MODEL_NAME}..."
    ollama pull ${MODEL_NAME}
    if [ $? -ne 0 ]; then
        echo "Failed to pull model ${MODEL_NAME}"
        exit 1
    fi
}

# Start Ollama server in the background
ollama serve &

# Wait for the server to start
echo "Waiting for Ollama server to start..."
sleep 10  # Adjust this value if needed

# Check if the model exists, pull if it doesn't
if ! ollama list | grep -q "${MODEL_NAME}"; then
    pull_model
fi

# Bring the Ollama server to the foreground
wait