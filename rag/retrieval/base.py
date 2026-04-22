from abc import ABC, abstractmethod
from typing import List

from rag.chunking.base import ChunkResult


class BaseRetriever(ABC):
  
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[ChunkResult]:
    
        pass
