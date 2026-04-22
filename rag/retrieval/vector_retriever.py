from typing import List

from rag.retrieval.base import BaseRetriever
from rag.embeddings.base import BaseEmbedder
from rag.vector_store.base import BaseVectorStore
from rag.chunking.base import ChunkResult
from rag.observability.logger import get_logger, timed

logger = get_logger("retrieval.vector")


class VectorRetriever(BaseRetriever):


    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore, top_k: int = 10):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    @timed(label="Vector Retrieval")
    def retrieve(self, query: str, top_k: int = None) -> List[ChunkResult]:
        k = top_k or self.top_k

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(query_embedding, top_k=k)

        chunks = [r.to_chunk_result() for r in results]
        logger.info(f"Retrieved {len(chunks)} vector results for query: {query[:50]}...")
        return chunks
