from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
 

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
    
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
  
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        
        pass
