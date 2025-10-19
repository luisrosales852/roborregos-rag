#Roborregos Rag project Inmortal and Reflex

Hi, this is my rag project for submission to the roborregos team. Im using 2 vector databases here, one with information about the catalogue of a construction company
and the other with information about how I met my friend "El inmortal" and information about our and his adventures.
Ill explain here a little bit about the project and how to run it.

This RAG system implements
- Query Translation with step back prompting and CRAG
- Task distinction
- Caching with Redis (question-answer pairs as well as embeddings)
- ros 2 implementation (containirized)
- Uses docker for simplicity exporting it

For this rag system I extensively used langchain libraries and everything it has to offer. It has 2 folders, langchain and ros2. In the langchain folder theres the standard implementation.
TO use it first you will have to do two things
- Add a .env with OPENAI_API_KEY and REDIS_URL = redis://localhost:6379 . This is the default and what I will use. You must put the OPEN_AI_APIKEY yourself
- Run this   docker run -d --name redis -p 6379:6379 redis:latest

Then just start the main.py server and thats it.

For ROS 2 implementation and what this project is all about this is how to run it.
- Add a .env file one folder below the ROS 2 folder, so on the same folder level as rag_service. It must have REDIS_URL= redis://localhost:6379 and the OPEN_AI_APIKEY

Now docker commands
- Navigate to ros2 folder using cd
- Run this: docker compose build
- Run this: docker-compose up
- Run this: docker exec -it rag_service_node /ros2_entrypoint.sh bash (This is to get into the entry point)
- Run this: ros2 run rag_service rag_client

