docker exec -it rag_service_node /ros2_entrypoint.sh bash
ros2 run rag_service rag_client_example

  docker exec -it rag_ollama ollama pull llama3.2
  
  docker exec rag_redis redis-cli FLUSHALL

To Start Using It:

  cd ros2

  # Start all services
  docker-compose up -d

  # Pull the Ollama model (llama3.2 by default)
  chmod +x init-ollama.sh
  ./init-ollama.sh

  # Or manually:
  docker exec -it rag_ollama ollama pull llama3.1:8b
  # Check logs
  docker logs rag_service_node -f

  The system now runs completely locally without needing any API keys! All three services (Redis, Ollama, and the ROS2 RAG service)
  communicate within the Docker network.


    docker-compose restart rag_service