import hashlib
import json
import pickle
from typing import Optional, Dict
import redis
from langchain_community.cache import RedisCache
from langchain.globals import set_llm_cache
from langchain_community.storage import RedisStore
from langchain.embeddings import CacheBackedEmbeddings
class RAGCacheManager:
    def __init__(self, 
                 redis_url = "redis://localhost:6379",
                 cache_prefix = "rag_cache",
                 max_qa_pairs = 10000):
        
        self.redis_url = redis_url
        self.cache_prefix = cache_prefix
        self.max_qa_pairs = max_qa_pairs
        
        # Initialize Redis client
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        
        # Test Redis connection
        try:
            self.redis_client.ping()
            print("Redis connection established")
        except redis.ConnectionError:
            print("Redis connection failed - make sure Redis is running")
            raise
        
        # Set up embeddings store with Redis (this caches embeddings automatically)
        self.embeddings_store = RedisStore(
            client=self.redis_client,
            namespace="embeddings"
        )
    def setup_cached_embeddings(self, base_embeddings):
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            base_embeddings,
            self.embeddings_store,
            namespace=f"{self.cache_prefix}:{base_embeddings.model}"
        )
        return cached_embeddings
    
    def _generate_qa_key(self, question: str) -> str:
        hash_key = hashlib.md5(question.encode()).hexdigest()
        return f"{self.cache_prefix}:qa:{hash_key}"
    
    def _get_current_qa_count(self) -> int:
        pattern = f"{self.cache_prefix}:qa:*"
        return len(list(self.redis_client.scan_iter(match=pattern)))
    
    def get_cached_answer(self, question: str) -> Optional[str]:

        cache_key = self._generate_qa_key(question)
        
        cached_answer = self.redis_client.get(cache_key)
        if cached_answer:
            print(f"Found cached answer for question hash: {cache_key[-8:]}...")
            return cached_answer.decode('utf-8') if isinstance(cached_answer, bytes) else cached_answer
        
        print(f"No cached answer for question hash: {cache_key[-8:]}...")
        return None
    
    def cache_qa_pair(self, question: str, answer: str):
    
        cache_key = self._generate_qa_key(question)
    
        is_new_entry = not self.redis_client.exists(cache_key)
    
        if is_new_entry:
            current_count = self._get_current_qa_count()
        
            if current_count >= self.max_qa_pairs:
                print(f"📊 Cache at capacity ({current_count}/{self.max_qa_pairs})")
            
                # Get first (oldest) key from scan and evict it
                pattern = f"{self.cache_prefix}:qa:*"
                key_iterator = self.redis_client.scan_iter(match=pattern, count=100)
            
                try:
                    oldest_key = next(key_iterator)
                    self.redis_client.delete(oldest_key)
                    print("Evicted one")
                
                except StopIteration:
                    print(" No keys available to evict")
    
        # Cache the Q&A pair
        self.redis_client.set(cache_key, answer)
    
        current_count = self._get_current_qa_count()
        print(f" Cached Q&A pair: ...{cache_key[-8:]} (Total: {current_count}/{self.max_qa_pairs})")
    
    def clear_qa_cache(self):
        pattern = f"{self.cache_prefix}:qa:*"
        keys_to_delete = list(self.redis_client.scan_iter(match=pattern))
        
        if keys_to_delete:
            self.redis_client.delete(*keys_to_delete)
            print(f"Deleted {len(keys_to_delete)} Q&A cache entries")
        else:
            print("No Q&A cache entries to delete")
    