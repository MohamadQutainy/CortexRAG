from typing import List, Optional

from rag.vector_store.base import BaseVectorStore, SearchResult
from rag.observability.logger import get_logger

logger = get_logger("vector_store.pinecone")


class PineconeStore(BaseVectorStore):


    def __init__(self, collection_name: str = "docs", api_key: str = None, environment: str = None):
        self.collection_name = collection_name



        logger.warning("Pinecone is not enabled — requires API key and pinecone-client")

    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: Optional[List[str]] = None,
    ):
        
        raise NotImplementedError(
            "Pinecone storage is not enabled. To enable it:\n"
            "1. Run: pip install pinecone-client\n"
            "2. Add PINECONE_API_KEY to your .env file\n"
            "3. Complete the implementation in pinecone_store.py"
        )

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[SearchResult]:
        raise NotImplementedError("Pinecone search is not implemented yet.")

    def get_all_documents(self) -> List[SearchResult]:
        raise NotImplementedError("Pinecone does not support direct full document retrieval in this implementation.")

    def delete_collection(self):
        raise NotImplementedError("Pinecone index deletion is not implemented yet.")

    def count(self) -> int:
        return 0
