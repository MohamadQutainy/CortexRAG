import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional

from rag.observability.logger import get_logger

logger = get_logger("cache")


class Cache:
 

    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 24, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_hours * 3600
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache enabled at: {self.cache_dir}")

    def _make_key(self, prefix: str, data: str) -> str:
      
        content = f"{prefix}:{data}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _get_path(self, key: str) -> Path:
        
        return self.cache_dir / f"{key}.json"

    def get(self, prefix: str, data: str) -> Optional[Any]:
   
        if not self.enabled:
            return None

        key = self._make_key(prefix, data)
        path = self._get_path(key)

        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)

            
            if time.time() - cached.get("timestamp", 0) > self.ttl_seconds:
                path.unlink(missing_ok=True)
                logger.debug(f"Cache expired: {prefix}")
                return None

            logger.debug(f"✅ Cache hit: {prefix}")
            return cached["value"]

        except (json.JSONDecodeError, KeyError):
            path.unlink(missing_ok=True)
            return None

    def set(self, prefix: str, data: str, value: Any):
  
        if not self.enabled:
            return

        key = self._make_key(prefix, data)
        path = self._get_path(key)

        cached = {
            "timestamp": time.time(),
            "prefix": prefix,
            "value": value,
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cached, f, ensure_ascii=False)
            logger.debug(f"💾 Saved to cache: {prefix}")
        except (TypeError, OSError) as e:
            logger.warning(f"Failed to save to cache: {e}")

    def clear(self):
        """مسح جميع البيانات المخزنة"""
        if self.cache_dir.exists():
            for file in self.cache_dir.glob("*.json"):
                file.unlink()
            logger.info("Cache cleared")


# --- مثيل عام (singleton) ---
_global_cache: Optional[Cache] = None


def get_cache() -> Cache:
   
    global _global_cache
    if _global_cache is None:
        from config import get_config
        cfg = get_config()
        cache_cfg = cfg["cache"]
        _global_cache = Cache(
            cache_dir=cache_cfg["directory"],
            ttl_hours=cache_cfg["ttl_hours"],
            enabled=cache_cfg["enabled"],
        )
    return _global_cache
