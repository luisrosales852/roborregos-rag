# Cloud Run image for the LangChain RAG service.
# Build context is the repo root, because the prebuilt vector stores live there.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROMA_DB1_PATH=/app/chroma_knowledge1_db \
    CHROMA_DB2_PATH=/app/chroma_knowledge2_db

WORKDIR /app

COPY langchain/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the source PDFs (main.py resolves them relative to itself).
COPY langchain/main.py langchain/caching.py langchain/serve.py ./
COPY langchain/knowledge1.pdf langchain/knowledge2.pdf ./

# Prebuilt vector stores. Baking these in means the container needs no disk and
# never re-embeds the PDFs on a cold start, which would cost OpenAI spend and
# add ~30s to the first request.
COPY chroma_knowledge1_db ./chroma_knowledge1_db
COPY chroma_knowledge2_db ./chroma_knowledge2_db

# Chroma opens its sqlite store read-write, so the runtime user must own it.
RUN useradd --create-home app && chown -R app:app /app
USER app

EXPOSE 8080

# Cloud Run injects PORT; the default keeps `docker run -p 8080:8080` working.
CMD ["sh", "-c", "exec uvicorn serve:app --host 0.0.0.0 --port ${PORT:-8080}"]
