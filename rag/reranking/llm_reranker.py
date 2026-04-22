import json
from typing import List

from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt

from rag.cache.cache import get_cache
from rag.chunking.base import ChunkResult
from rag.generation.prompts import RERANK_SYSTEM_PROMPT
from rag.observability.logger import get_logger, timed
from rag.reranking.base import BaseReranker

logger = get_logger("reranking.llm")


class RankOrder(BaseModel):
    

    order: list[int] = Field(
        description="Chunk ids ordered from most relevant to least relevant"
    )


class LLMReranker(BaseReranker):
    

    def __init__(self, model: str = "openai/gpt-4.1-nano"):
        self.model = model

    def _build_cache_key(self, query: str, chunks: List[ChunkResult]) -> str:
        payload = {
            "query": query,
            "model": self.model,
            "chunks": [
                {
                    "source": chunk.metadata.get("source", "unknown"),
                    "text": chunk.page_content[:250],
                }
                for chunk in chunks
            ],
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @retry(wait=wait_exponential(multiplier=1, min=10, max=240), stop=stop_after_attempt(3))
    @timed(label="reRanker LLM")
    def rerank(self, query: str, chunks: List[ChunkResult], top_k: int = None) -> List[ChunkResult]:
       
        if not chunks:
            return []

        cache = get_cache()
        cache_key = self._build_cache_key(query, chunks)
        cached = cache.get("rerank", cache_key)
        if cached is not None:
            try:
                reordered = [chunks[index - 1] for index in cached if 0 < index <= len(chunks)]
                if reordered:
                    return reordered[:top_k] if top_k else reordered
            except (IndexError, TypeError):
                pass

        user_prompt = f"Question:\n{query}\n\nRank all chunks by relevance:\n\n"
        for index, chunk in enumerate(chunks, start=1):
            user_prompt += f"# CHUNK ID: {index}\n\n{chunk.page_content}\n\n"
        user_prompt += "Reply only with the list of ranked chunk ids."

        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=RankOrder,
        )
        order = RankOrder.model_validate_json(response.choices[0].message.content).order

        cache.set("rerank", cache_key, order)

        reranked = [chunks[index - 1] for index in order if 0 < index <= len(chunks)]
        logger.info(f"Reranked {len(reranked)} chunks")
        return reranked[:top_k] if top_k else reranked
