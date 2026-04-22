import json
from typing import List, Tuple

from litellm import completion
from tenacity import retry, wait_exponential, stop_after_attempt

from rag.cache.cache import get_cache
from rag.chunking.base import ChunkResult
from rag.observability.logger import get_logger, timed

logger = get_logger("generation")


DEFAULT_SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing {company_name}.
Answer the user's question based on the provided context.
If you don't know the answer, say so.

{context}
"""


class RAGGenerator:

    def __init__(
        self,
        retriever,
        reranker=None,
        query_rewriter=None,
        query_expander=None,
        model: str = "openai/gpt-4.1-nano",
        company_name: str = "Insurellm",
        final_k: int = 10,
        system_prompt_template: str | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.query_rewriter = query_rewriter
        self.query_expander = query_expander
        self.model = model
        self.company_name = company_name
        self.final_k = final_k
        self.system_prompt_template = system_prompt_template or DEFAULT_SYSTEM_PROMPT

    def _merge_chunks(self, chunks_list: List[List[ChunkResult]]) -> List[ChunkResult]:
       
        seen = set()
        merged: List[ChunkResult] = []
        for chunks in chunks_list:
            for chunk in chunks:
                key = (
                    chunk.metadata.get("source", "unknown"),
                    chunk.metadata.get("chunk_id"),
                    chunk.page_content[:120],
                )
                if key not in seen:
                    seen.add(key)
                    merged.append(chunk)
        return merged

    def _build_context(self, chunks: List[ChunkResult]) -> str:
        return "\n\n".join(
            f"Extract from {chunk.metadata.get('source', 'unknown')}:\n{chunk.page_content}"
            for chunk in chunks
        )

    def _build_messages(self, question: str, context: str, history: list | None = None) -> list:
        system_prompt = self.system_prompt_template.format(
            company_name=self.company_name,
            context=context,
        )
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})
        return messages

    def _build_answer_cache_key(self, question: str, history: list | None = None) -> str:
      
        payload = {
            "question": question,
            "history": history or [],
            "model": self.model,
            "company_name": self.company_name,
            "final_k": self.final_k,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @timed(label="Context Retrieval")
    def fetch_context(self, question: str, history: list | None = None) -> List[ChunkResult]:

        all_chunks: List[List[ChunkResult]] = []

        original_chunks = self.retriever.retrieve(question)
        all_chunks.append(original_chunks)

        if self.query_rewriter:
            rewritten = self.query_rewriter.rewrite(question, history)
            if rewritten and rewritten != question:
                all_chunks.append(self.retriever.retrieve(rewritten))

        if self.query_expander:
            expanded = self.query_expander.expand(question)
            for expanded_query in expanded[1:]:
                all_chunks.append(self.retriever.retrieve(expanded_query))

        merged = self._merge_chunks(all_chunks)

        if self.reranker:
            merged = self.reranker.rerank(question, merged, top_k=self.final_k)
        else:
            merged = merged[: self.final_k]

        logger.info(f"Retrieved {len(merged)} final chunks")
        return merged

    @retry(wait=wait_exponential(multiplier=1, min=10, max=240), stop=stop_after_attempt(3))
    @timed(label="generate answer")
    def answer(
        self,
        question: str,
        history: list | None = None,
    ) -> Tuple[str, List[ChunkResult]]:
        cache = get_cache()
        cache_key = self._build_answer_cache_key(question, history)
        cached = cache.get("llm_answer", cache_key)
        if cached is not None:
            logger.info("Answer from cache")
            chunks = self.fetch_context(question, history)
            return cached, chunks

        chunks = self.fetch_context(question, history)
        context = self._build_context(chunks)
        messages = self._build_messages(question, context, history)

        response = completion(model=self.model, messages=messages)
        answer_text = response.choices[0].message.content

        cache.set("llm_answer", cache_key, answer_text)
        return answer_text, chunks
