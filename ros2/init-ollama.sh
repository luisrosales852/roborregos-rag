#!/bin/bash
# Script to initialize Ollama with llama3.1:8b and gemma:300m models
# This should be run after the Ollama container is up

set -e

OLLAMA_HOST=${OLLAMA_HOST:-"http://localhost:11434"}
LLM_MODEL=${OLLAMA_LLM_MODEL:-"llama3.1:8b"}
EMBEDDING_MODEL=${OLLAMA_EMBEDDING_MODEL:-"gemma:300m"}

echo "Waiting for Ollama to be ready..."
until curl -sf "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; do
    echo "Waiting for Ollama service..."
    sleep 2
done

echo "Ollama is ready!"
echo "Pulling LLM model: $LLM_MODEL (this may take a while, ~8GB download)"
ollama pull "$LLM_MODEL"

echo ""
echo "Pulling embedding model: $EMBEDDING_MODEL (this may take a while, ~300MB download)"
ollama pull "$EMBEDDING_MODEL"

echo ""
echo "Both models have been pulled successfully!"
echo "  - LLM (chat): $LLM_MODEL"
echo "  - Embeddings: $EMBEDDING_MODEL"
echo "You can now start using the RAG service with these models."
