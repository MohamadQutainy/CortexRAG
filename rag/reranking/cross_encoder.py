from typing import List

from rag.reranking.base import BaseReranker
from rag.chunking.base import ChunkResult
from rag.observability.logger import get_logger, timed

logger = get_logger("reranking.cross_encoder")


class CrossEncoderReranker(BaseReranker):
 

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Cross-Encoder loaded: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "you should install sentence-transformers:\n"
                    "pip install sentence-transformers"
                )

    @timed(label="Coss-Encoder")
    def rerank(self, query: str, chunks: List[ChunkResult], top_k: int = None) -> List[ChunkResult]:
        if not chunks:
            return []

        self._load_model()

        pairs = [(query, chunk.page_content) for chunk in chunks]

        scores = self._model.predict(pairs)

        scored_chunks = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        reranked = [chunk for chunk, score in scored_chunks]

        logger.info(f"Reranked {len(reranked)} chunks with Cross-Encoder")
        return reranked[:top_k] if top_k else reranked
