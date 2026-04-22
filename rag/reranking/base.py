
from abc import ABC, abstractmethod
from typing import List

from rag.chunking.base import ChunkResult


class BaseReranker(ABC):
   

    @abstractmethod
    def rerank(self, query: str, chunks: List[ChunkResult], top_k: int = None) -> List[ChunkResult]:
 
        pass
