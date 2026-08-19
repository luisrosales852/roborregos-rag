"""HTTP wrapper around the RAG pipeline, for deploying to Cloud Run.

Importing `main` builds the entire pipeline (PDF load, splits, vector stores, BM25
retrievers, chains) exactly once at process start, so requests only pay for the
query itself. `python main.py` still works as the original interactive CLI.
"""

import os
import time

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import main as rag

API_KEY = os.getenv("API_KEY")

app = FastAPI(
    title="Roborregos RAG",
    description="Inmortal + Reflex retrieval-augmented generation service",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    response_time: float


def _check_api_key(provided: str | None) -> None:
    """No-op when API_KEY is unset, so local runs need no header."""
    if API_KEY and provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.get("/health")
def health():
    """Liveness probe. Returns 200 as soon as the pipeline finished importing."""
    return {"status": "ok", "cache": rag.redis_client is not None}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)

    started = time.monotonic()
    try:
        answer = rag.routeQuestion(request.question)
    except Exception as exc:  # surfaced as 500 with the reason in Cloud Run logs
        print(f"Query failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    return QueryResponse(answer=answer, response_time=time.monotonic() - started)


if __name__ == "__main__":
    import uvicorn

    # Cloud Run injects PORT; 8080 is its default and a sane local fallback.
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
