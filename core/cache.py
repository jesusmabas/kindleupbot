# kindleupbot/core/cache.py
import time
import asyncio
from typing import Optional, Any

class CacheManager:
    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self.cache[key] = (value, time.time())
        if ttl:
            asyncio.create_task(self._auto_expire(key, ttl))

    async def _auto_expire(self, key: str, ttl: int):
        await asyncio.sleep(ttl)
        self.cache.pop(key, None)

    def clear(self):
        self.cache.clear()

# Instancia global para ser usada en todo el bot
from ..config import settings
cache_manager = CacheManager(default_ttl=settings.CACHE_DURATION)