import json

from litellm import completion
from tenacity import retry, wait_exponential, stop_after_attempt

from rag.cache.cache import get_cache
from rag.generation.prompts import REWRITE_PROMPT
from rag.observability.logger import get_logger, timed

logger = get_logger("query.rewriter")


class QueryRewriter:
   

    def __init__(self, model: str = "openai/gpt-4.1-nano"):
        self.model = model

    @retry(wait=wait_exponential(multiplier=1, min=10, max=240), stop=stop_after_attempt(3))
    @timed(label="Query Rewriter")
    def rewrite(self, question: str, history: list | None = None) -> str:
        
        history = history or []
        cache = get_cache()
        cache_key = json.dumps(
            {"question": question, "history": history, "model": self.model},
            ensure_ascii=False,
            sort_keys=True,
        )

        cached = cache.get("query_rewrite", cache_key)
        if cached is not None:
            return cached

        prompt = REWRITE_PROMPT.format(history=history, question=question)
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        rewritten = response.choices[0].message.content.strip()

        cache.set("query_rewrite", cache_key, rewritten)
        logger.info(f"Rewriting: '{question[:40]}...' -> '{rewritten[:40]}...'")
        return rewritten
