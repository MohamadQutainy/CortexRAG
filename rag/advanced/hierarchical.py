from typing import List, Tuple
from litellm import completion
from tenacity import retry, wait_exponential, stop_after_attempt

from rag.chunking.base import ChunkResult
from rag.embeddings.base import BaseEmbedder
from rag.vector_store.base import BaseVectorStore
from rag.observability.logger import get_logger, timed

logger = get_logger("advanced.hierarchical")


class HierarchicalRAG:


    def __init__(
        self,
        embedder: BaseEmbedder,
        summary_store: BaseVectorStore,
        detail_store: BaseVectorStore,
        llm_model: str = "openai/gpt-4.1-nano",
        summary_top_k: int = 5,
        detail_top_k: int = 10,
    ):
        self.embedder = embedder
        self.summary_store = summary_store
        self.detail_store = detail_store
        self.llm_model = llm_model
        self.summary_top_k = summary_top_k
        self.detail_top_k = detail_top_k

    @retry(wait=wait_exponential(multiplier=1, min=10, max=240), stop=stop_after_attempt(3))
    def generate_summary(self, document_text: str, source: str) -> str:

        prompt = f"""
Summarize the following document in 2-3 sentences.
Focus on the key information, entities, and facts.
Document source: {source}

Document:
{document_text[:3000]}

Summary:
"""
        response = completion(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    @timed(label="HierarchicalRAG Insert")
    def ingest_documents(self, documents: list, chunks: List[ChunkResult]):

        logger.info(f"Generating summaries for {len(documents)} documents...")
        summary_texts = []
        summary_metas = []

        for doc in documents:
            summary = self.generate_summary(doc["text"], doc["source"])
            summary_texts.append(summary)
            summary_metas.append({
                "source": doc["source"],
                "type": doc.get("type", "unknown"),
                "level": "summary",
            })


        summary_embeddings = self.embedder.embed(summary_texts)
        summary_ids = [f"summary_{i}" for i in range(len(summary_texts))]
        self.summary_store.add(summary_texts, summary_embeddings, summary_metas, summary_ids)


        detail_texts = [c.page_content for c in chunks]
        detail_metas = [c.metadata for c in chunks]
        detail_embeddings = self.embedder.embed(detail_texts)
        detail_ids = [f"detail_{i}" for i in range(len(detail_texts))]
        self.detail_store.add(detail_texts, detail_embeddings, detail_metas, detail_ids)

        logger.info(
            f"{len(summary_texts)} Summarizing + {len(detail_texts)} "
        )

    @timed(label="HierarchicalRAG Retrieval")
    def retrieve(self, query: str) -> List[ChunkResult]:

        query_embedding = self.embedder.embed_query(query)

        
        summary_results = self.summary_store.search(query_embedding, top_k=self.summary_top_k)

       
        relevant_sources = set()
        for result in summary_results:
            source = result.metadata.get("source", "")
            if source:
                relevant_sources.add(source)

        logger.info(f"First Level: {len(relevant_sources)} relevant documents")

        
        detail_results = self.detail_store.search(
            query_embedding, top_k=self.detail_top_k * 2
        )

       
        filtered = []
        for result in detail_results:
            source = result.metadata.get("source", "")
            if source in relevant_sources:
                filtered.append(result.to_chunk_result())

        
        if len(filtered) < self.detail_top_k:
            for result in detail_results:
                chunk = result.to_chunk_result()
                if chunk not in filtered:
                    filtered.append(chunk)
                if len(filtered) >= self.detail_top_k:
                    break

        logger.info(f"Second Level: {len(filtered[:self.detail_top_k])} final chunks")
        return filtered[:self.detail_top_k]
