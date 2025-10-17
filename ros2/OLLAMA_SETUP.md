# ROS2 RAG Service - Setup Guide

> **⚠️ DEPRECATION NOTICE**: This guide previously documented Ollama setup. The system has been **migrated to use ChatGPT/OpenAI API** instead of Ollama.
>
> **Current Configuration**: The ROS2 RAG service now uses OpenAI's `gpt-4o-mini` model for LLM operations and `text-embedding-3-large` for embeddings.
>
> For the updated setup instructions with ChatGPT/OpenAI, see the section below.

---

## NEW: ChatGPT/OpenAI Setup (Current Configuration)

### Prerequisites
- Docker and Docker Compose installed
- OpenAI API Key (get one at https://platform.openai.com/api-keys)

### Configuration

The `.env` file should contain:

```bash
# ChatGPT/OpenAI Configuration (ACTIVE)
OPENAI_API_KEY=your_openai_api_key_here

# Redis Configuration
REDIS_URL=redis://redis:6379

# ROS2 Configuration
ROS_DOMAIN_ID=0
```

**Important**: Replace `your_openai_api_key_here` with your actual OpenAI API key.

### Setup Instructions

1. **Set your OpenAI API Key**:
   ```bash
   cd ros2
   # Edit .env file and add your OPENAI_API_KEY
   nano .env
   ```

2. **Start the Services**:
   ```bash
   docker-compose up -d
   ```

   This will start:
   - Redis (for caching)
   - ROS2 RAG Service (using ChatGPT/OpenAI)

3. **Verify the Setup**:
   ```bash
   docker-compose ps
   docker logs rag_service_node
   ```

4. **Test the RAG Service**:
   ```bash
   # Run a test query
   docker exec -it rag_service_node ros2 service call /rag_query rag_interfaces/srv/RAGQuery "{question: 'What is your name?'}"
   ```

### Benefits of ChatGPT/OpenAI over Ollama
- ✅ No local model downloads required (~4-8GB saved)
- ✅ Faster responses (no local model loading)
- ✅ Lower hardware requirements (no 8GB RAM minimum)
- ✅ Access to latest GPT models
- ✅ High-quality embeddings with text-embedding-3-large
- ⚠️ Requires internet connection
- ⚠️ API usage costs (but very affordable with gpt-4o-mini)

### Cost Considerations
- **gpt-4o-mini**: $0.15 per 1M input tokens, $0.60 per 1M output tokens
- **text-embedding-3-large**: $0.13 per 1M tokens
- Typical RAG query: ~$0.001-0.005 (cached queries are free)

---

## DEPRECATED: Ollama Setup (No Longer Active)

The content below is preserved for reference but is **no longer used** in the current system.

This guide explains how to run the ROS2 RAG service using Ollama instead of OpenAI in a containerized environment.

## Architecture

The system consists of three main services:
- **Redis**: Caching layer for embeddings and Q&A pairs
- **Ollama**: Local LLM service running inside Docker
- **ROS2 RAG Service**: The main service that uses Ollama for RAG operations

## Prerequisites

- Docker and Docker Compose installed
- At least 8GB of RAM available for Ollama models
- Sufficient disk space for Ollama models (~4-8GB depending on model)

## Configuration

### Environment Variables

The `.env` file contains the following configurations:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2

# Redis Configuration
REDIS_URL=redis://redis:6379

# ROS2 Configuration
ROS_DOMAIN_ID=0
```

You can change `OLLAMA_MODEL` to use different models available in Ollama:
- `llama3.2` (default, ~4GB)
- `llama3.2:1b` (smaller, faster)
- `mistral` (~4GB)
- `phi3` (~2.3GB)
- See [Ollama Library](https://ollama.com/library) for more options

## Setup Instructions

### 1. Start the Services

```bash
cd ros2
docker-compose up -d
```

This will start:
- Redis (for caching)
- Ollama (LLM service)
- ROS2 RAG Service

### 2. Pull the Ollama Model

After the containers are running, you need to pull the model into the Ollama container:

**Option A: Using the provided script**
```bash
chmod +x init-ollama.sh
./init-ollama.sh
```

**Option B: Manual setup**
```bash
# Enter the Ollama container
docker exec -it rag_ollama ollama pull llama3.2

# Or pull a different model
docker exec -it rag_ollama ollama pull mistral
```

**Option C: From outside the container**
```bash
curl -X POST http://localhost:11434/api/pull -d '{"name": "llama3.2"}'
```

### 3. Verify the Setup

Check that all services are running:
```bash
docker-compose ps
```

Check Ollama logs:
```bash
docker logs rag_ollama
```

Check RAG service logs:
```bash
docker logs rag_service_node
```

### 4. Test the RAG Service

You can test the service using the ROS2 client:

```bash
# Run the client container
docker-compose --profile client up rag_client

# Or use ros2 service call directly from another terminal
docker exec -it rag_service_node ros2 service call /rag_query rag_interfaces/srv/RAGQuery "{question: 'What is your name?'}"
```

## Troubleshooting

### Ollama Container Won't Start
- Check if port 11434 is already in use
- Ensure you have enough RAM (at least 8GB recommended)
- Check logs: `docker logs rag_ollama`

### Model Download Fails
- Check internet connection
- Try a smaller model first (e.g., `llama3.2:1b`)
- Ensure sufficient disk space

### RAG Service Can't Connect to Ollama
- Verify Ollama is running: `curl http://localhost:11434/api/health`
- Check that the model is pulled: `docker exec -it rag_ollama ollama list`
- Ensure the `OLLAMA_BASE_URL` in `.env` matches the docker-compose service name

### Slow Performance
- Use a smaller model (e.g., `phi3` or `llama3.2:1b`)
- Increase Docker memory allocation
- Consider using GPU acceleration (requires NVIDIA GPU and nvidia-docker)

## GPU Acceleration (Optional)

To use GPU acceleration with Ollama, modify the `docker-compose.yml`:

```yaml
ollama:
  image: ollama/ollama:latest
  container_name: rag_ollama
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  # ... rest of config
```

Make sure you have:
- NVIDIA GPU
- NVIDIA Docker runtime installed
- Proper GPU drivers

## Switching Models

To switch to a different model:

1. Update the `.env` file with the new model name:
   ```bash
   OLLAMA_MODEL=mistral
   ```

2. Pull the new model:
   ```bash
   docker exec -it rag_ollama ollama pull mistral
   ```

3. Restart the RAG service:
   ```bash
   docker-compose restart rag_service
   ```

4. Clear the vector databases if needed (they use the same embeddings model):
   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

## Performance Considerations

- **First Query**: The first query will be slower as the model needs to load into memory
- **Caching**: Redis caches embeddings and Q&A pairs, speeding up repeated queries
- **Model Size**: Smaller models are faster but may have lower quality responses
- **Vector DBs**: The vector databases are persisted, so they only need to be built once

## Stopping the Services

```bash
# Stop services but keep data
docker-compose down

# Stop services and remove volumes (clears cache and vector DBs)
docker-compose down -v
```

## Monitoring

Check resource usage:
```bash
docker stats
```

View real-time logs:
```bash
docker-compose logs -f
```

## Additional Resources

- [Ollama Documentation](https://ollama.com)
- [Ollama Model Library](https://ollama.com/library)
- [LangChain Ollama Integration](https://python.langchain.com/docs/integrations/llms/ollama)
