#!/bin/bash

# Setup script for Ollama models
echo "Setting up Ollama models for Pest Management AI..."

# Wait for Ollama service to be ready
echo "Waiting for Ollama service to start..."
until curl -f http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "Waiting for Ollama to be ready..."
    sleep 5
done

echo "Ollama is ready! Pulling required models..."

# Pull recommended models for pest management
echo "Pulling llama3.2 model (recommended for general chat)..."
curl -X POST http://localhost:11434/api/pull -d '{"name": "llama3.2"}'

echo "Pulling llava model (for image analysis)..."
curl -X POST http://localhost:11434/api/pull -d '{"name": "llava"}'

echo "Setup complete! Available models:"
curl http://localhost:11434/api/tags

echo ""
echo "You can now use the following models:"
echo "- llama3.2: General purpose chat model"
echo "- llava: Vision model for image analysis"
echo ""
echo "The backend service will use llama3.2 by default."
echo "To use a different model, send a request to /api/chat with the 'model' parameter."