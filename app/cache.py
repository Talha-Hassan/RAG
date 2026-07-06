import hashlib
import time
from typing import Optional
from app.config import get_settings

settings = get_settings()

class ResponseCache:
    """A simple in-memory cache for storing responses with expiration."""

    def __init__(self):
        self.ttl = settings.cache_ttl_seconds
        self.cache : dict[str,dict] = {}
        self.hits = 0 
        self.misses = 0 
    
    def make_key(self, prompt: str) -> str:
        """Generate a unique key for the given prompt."""
        query = prompt.strip().lower()
        return hashlib.sha256(query.encode()).hexdigest()
    
    async def get(self, prompt: str) -> Optional[str]:
        """Retrieve a cached response for the given prompt, if it exists and is not expired."""
        key = self.make_key(prompt)

        # print(f"Cache lookup for key: {key}")

        if key in self.cache:
            entity = self.cache[key]
            if time.time() - entity['timestamp'] < self.ttl:
                self.hits += 1
                return entity['response']
            else:
                # Entry has expired
                del self.cache[key]

        self.misses += 1
        # print(f"Cache miss for key: {key}")
        return None
    
    def set(self, prompt: str, response: str):
        """Store a response in the cache with the current timestamp."""
        key = self.make_key(prompt)
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
        }

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.2%}",
            'size': len(self.cache)
        }