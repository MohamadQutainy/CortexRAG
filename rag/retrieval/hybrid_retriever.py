from typing import List, Dict

from rag.retrieval.base import BaseRetriever
from rag.retrieval.vector_retriever import VectorRetriever
from rag.retrieval.bm25_retriever import BM25Retriever
from rag.embeddings.base import BaseEmbedder
from rag.vector_store.base import BaseVectorStore
from rag.chunking.base import ChunkResult
from rag.observability.logger import get_logger, timed

logger = get_logger("retrieval.hybrid")


RRF_K = 60


class HybridRetriever(BaseRetriever):


    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        top_k: int = 10,
        alpha: float = 0.7,
    ):
        self.top_k = top_k
        self.alpha = alpha

        self.vector_retriever = VectorRetriever(embedder, vector_store, top_k=top_k)
        self.bm25_retriever = BM25Retriever(vector_store, top_k=top_k)

    def _rrf_merge(
        self,
        vector_results: List[ChunkResult],
        bm25_results: List[ChunkResult],
    ) -> List[ChunkResult]:
 
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, ChunkResult] = {}

     
        for rank, chunk in enumerate(vector_results):
            key = chunk.page_content[:100]  
            rrf_score = self.alpha * (1.0 / (RRF_K + rank + 1))
            scores[key] = scores.get(key, 0) + rrf_score
            chunk_map[key] = chunk

        for rank, chunk in enumerate(bm25_results):
            key = chunk.page_content[:100]
            rrf_score = (1.0 - self.alpha) * (1.0 / (RRF_K + rank + 1))
            scores[key] = scores.get(key, 0) + rrf_score
            if key not in chunk_map:
                chunk_map[key] = chunk

       
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        return [chunk_map[k] for k in sorted_keys]

    @timed(label=" hybrid retrieval ")
    def retrieve(self, query: str, top_k: int = None) -> List[ChunkResult]:
      
        k = top_k or self.top_k

        
        vector_results = self.vector_retriever.retrieve(query, top_k=k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=k)

        # دمج النتائج بـ RRF
        merged = self._rrf_merge(vector_results, bm25_results)

        logger.info(
            f"hybrid retrieval  : {len(vector_results)} vector + {len(bm25_results)} BM25 → {min(len(merged), k)} result"
        )
        return merged[:k]
