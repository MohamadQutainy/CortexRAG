from typing import List, Optional

from rag.vector_store.base import BaseVectorStore, SearchResult
from rag.observability.logger import get_logger

logger = get_logger("vector_store.milvus")


class MilvusStore(BaseVectorStore):


    def __init__(
        self,
        collection_name: str = "docs",
        host: str = "localhost",
        port: int = 19530,
        dimension: int = 3072,
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.dimension = dimension



        logger.warning("Milvus is not enabled — requires Milvus server and pymilvus")

    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: Optional[List[str]] = None,
    ):

     raise NotImplementedError(
            "Milvus storage is not enabled. To enable it:\n"
            "1. Start a Milvus server (Docker or Zilliz Cloud)\n"
            "2. Run: pip install pymilvus\n"
            "3. Complete the implementation in milvus_store.py"
        )

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[SearchResult]:
        raise NotImplementedError("Milvus search is not implemented yet.")

    def get_all_documents(self) -> List[SearchResult]:
        raise NotImplementedError("Milvus document retrieval is not implemented yet.")

    def delete_collection(self):
        raise NotImplementedError("Milvus collection deletion is not implemented yet.")

    def count(self) -> int:
        return 0