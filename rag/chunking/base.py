from abc import ABC, abstractmethod
from typing import List, Dict
from pydantic import BaseModel


class ChunkResult(BaseModel):
    
    page_content: str
    metadata: dict


class BaseChunker(ABC):


    @abstractmethod
    def chunk(self, document: Dict[str, str]) -> List[ChunkResult]:
  
        pass

    def chunk_many(self, documents: List[Dict[str, str]]) -> List[ChunkResult]:
      
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks
