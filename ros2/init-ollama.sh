#!/bin/bash
# Script to initialize Ollama with the required model
# THIS SCRIPT IS COMMENTED OUT - NOW USING CHATGPT/OPENAI INSTEAD OF OLLAMA
# This should be run after the Ollama container is up

# set -e
#
# OLLAMA_HOST=${OLLAMA_HOST:-"http://localhost:11434"}
# MODEL=${OLLAMA_MODEL:-"llama3.1:8b"}
#
# echo "Waiting for Ollama to be ready..."
# until curl -sf "$OLLAMA_HOST/api/health" > /dev/null; do
#     echo "Waiting for Ollama service..."
#     sleep 2
# done
#
# echo "Ollama is ready!"
# echo "Pulling model: $MODEL"
# curl -X POST "$OLLAMA_HOST/api/pull" -d "{\"name\": \"$MODEL\"}"
#
# echo ""
# echo "Model $MODEL has been pulled successfully!"
# echo "You can now start using the RAG service."

echo "This script is no longer needed - the RAG service now uses ChatGPT/OpenAI API."
echo "Make sure your OPENAI_API_KEY is set in the .env file."
