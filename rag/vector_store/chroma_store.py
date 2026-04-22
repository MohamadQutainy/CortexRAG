from typing import List, Optional

from chromadb import PersistentClient

from rag.vector_store.base import BaseVectorStore, SearchResult
from rag.observability.logger import get_logger, timed

logger = get_logger("vector_store.chroma")


class ChromaStore(BaseVectorStore):


    def __init__(self, db_path: str, collection_name: str = "docs"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)
        logger.info(f"ChromaDB ready: {db_path}/{collection_name} ({self.count()} documents)")

    @timed(label="إضافة للـ ChromaDB")
    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: Optional[List[str]] = None,
    ):
        
        if ids is None:
            ids = [str(i) for i in range(len(documents))]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(documents)} documents to ChromaDB")

    @timed(label="search ChromaDB")
    def search(self, query_embedding: List[float], top_k: int = 10) -> List[SearchResult]:
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
               
                score = 1.0 / (1.0 + distance)
                search_results.append(SearchResult(document=doc, metadata=meta, score=score))

        return search_results

    def get_all_documents(self) -> List[SearchResult]:
    
        results = self.collection.get()
        search_results = []
        if results["documents"]:
            for doc, meta in zip(results["documents"], results["metadatas"]):
                search_results.append(SearchResult(document=doc, metadata=meta, score=0.0))
        return search_results

    def delete_collection(self):
        """حذف المجموعة بالكامل"""
        if self.collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection deleted: {self.collection_name}")
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def count(self) -> int:
     
        return self.collection.count()
