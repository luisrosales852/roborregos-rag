from typing import List, Optional
from openai import OpenAI

import os
class EmbeddingModel:
    """OpenAI Embedding Model wrapper"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-large"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"✗ Failed to generate embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"✗ Failed to generate embeddings: {e}")
            raise