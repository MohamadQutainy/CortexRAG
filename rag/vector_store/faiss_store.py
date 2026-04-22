import json
from pathlib import Path
from typing import List, Optional

from rag.vector_store.base import BaseVectorStore, SearchResult
from rag.observability.logger import get_logger, timed

logger = get_logger("vector_store.faiss")


class FAISSStore(BaseVectorStore):


    def __init__(self, db_path: str, collection_name: str = "docs"):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.index_path = self.db_path / f"{collection_name}.faiss"
        self.meta_path = self.db_path / f"{collection_name}_meta.json"

        self.db_path.mkdir(parents=True, exist_ok=True)

        self._index = None
        self._documents = []
        self._metadatas = []

     
        self._load()

    def _load(self):
        
        try:
            import faiss
        except ImportError:
            logger.warning("faiss is not installed — pip install faiss-cpu")
            return

        if self.index_path.exists() and self.meta_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._documents = data["documents"]
                self._metadatas = data["metadatas"]
            logger.info(f"FAISS index loaded: {len(self._documents)} documents")

    def _save(self):
        
        import faiss
        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump({"documents": self._documents, "metadatas": self._metadatas}, f, ensure_ascii=False)

    @timed(label="Add to FAISS")
    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: Optional[List[str]] = None,
    ):
        
        import faiss
        import numpy as np

        vectors = np.array(embeddings, dtype=np.float32)
        dimension = vectors.shape[1]

        if self._index is None:
            
            self._index = faiss.IndexFlatL2(dimension)

        self._index.add(vectors)
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)
        self._save()

        logger.info(f"Added {len(documents)} documents to FAISS")

    @timed(label="بحث FAISS")
    def search(self, query_embedding: List[float], top_k: int = 10) -> List[SearchResult]:
        """البحث في FAISS بمتجه التضمين"""
        import numpy as np

        if self._index is None or self._index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        distances, indices = self._index.search(query, min(top_k, self._index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            score = 1.0 / (1.0 + float(dist))
            results.append(SearchResult(
                document=self._documents[idx],
                metadata=self._metadatas[idx],
                score=score,
            ))

        return results

    def get_all_documents(self) -> List[SearchResult]:
        
        return [
            SearchResult(document=doc, metadata=meta, score=0.0)
            for doc, meta in zip(self._documents, self._metadatas)
        ]

    def delete_collection(self):
      
        self._index = None
        self._documents = []
        self._metadatas = []
        self.index_path.unlink(missing_ok=True)
        self.meta_path.unlink(missing_ok=True)
        logger.info("FAISS index deleted")

    def count(self) -> int:
        return len(self._documents)
