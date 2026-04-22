from typing import List

from rag.retrieval.base import BaseRetriever
from rag.vector_store.base import BaseVectorStore
from rag.chunking.base import ChunkResult
from rag.observability.logger import get_logger, timed

logger = get_logger("retrieval.bm25")


class BM25Retriever(BaseRetriever):
 

    def __init__(self, vector_store: BaseVectorStore, top_k: int = 10):
        self.vector_store = vector_store
        self.top_k = top_k
        self._bm25 = None
        self._documents = None
        self._metadatas = None

    def _build_index(self):
       
        if self._bm25 is not None:
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("you should install: rank_bm25:\npip install rank_bm25")

       
        all_docs = self.vector_store.get_all_documents()

        if not all_docs:
            logger.warning("No documents to build BM25 index")
            self._documents = []
            self._metadatas = []
            return

        self._documents = [doc.document for doc in all_docs]
        self._metadatas = [doc.metadata for doc in all_docs]

   
        tokenized = [doc.lower().split() for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)

        logger.info(f"BM25 index built from {len(self._documents)} documents")

    @timed(label="retrieval BM25")
    def retrieve(self, query: str, top_k: int = None) -> List[ChunkResult]:
      
        k = top_k or self.top_k
        self._build_index()

        if not self._documents or self._bm25 is None:
            return []


        query_tokens = query.lower().split()

      
        scores = self._bm25.get_scores(query_tokens)

    
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

     
        chunks = []
        for idx, score in scored_indices[:k]:
            if score > 0:
                chunks.append(ChunkResult(
                    page_content=self._documents[idx],
                    metadata=self._metadatas[idx],
                ))

        logger.info(f"Retrieved {len(chunks)} BM25 results for query: {query[:50]}...")
        return chunks

    def refresh_index(self):
        self._bm25 = None
        self._documents = None
        self._metadatas = None
        self._build_index()
