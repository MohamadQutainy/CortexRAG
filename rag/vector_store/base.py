from abc import ABC, abstractmethod
from typing import List, Optional

from rag.chunking.base import ChunkResult


class SearchResult:

    def __init__(self, document: str, metadata: dict, score: float = 0.0):
        self.document = document
        self.metadata = metadata
        self.score = score

    def to_chunk_result(self) -> ChunkResult:
        """تحويل إلى ChunkResult للتوافق مع باقي النظام"""
        return ChunkResult(page_content=self.document, metadata=self.metadata)


class BaseVectorStore(ABC):


    @abstractmethod
    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: Optional[List[str]] = None,
    ):
     
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 10) -> List[SearchResult]:
 
        pass

    @abstractmethod
    def get_all_documents(self) -> List[SearchResult]:

        pass

    @abstractmethod
    def delete_collection(self):
   
        pass

    @abstractmethod
    def count(self) -> int:
       
        pass
