import json
from typing import List

from litellm import completion
from tenacity import retry, wait_exponential, stop_after_attempt

from rag.cache.cache import get_cache
from rag.observability.logger import get_logger, timed

logger = get_logger("query.expander")


class QueryExpander:

    def __init__(self, model: str = "openai/gpt-4.1-nano", expansion_count: int = 3):
        self.model = model
        self.expansion_count = expansion_count

    @retry(wait=wait_exponential(multiplier=1, min=10, max=240), stop=stop_after_attempt(3))
    @timed(label="Query Expansion")
    def expand(self, question: str) -> List[str]:
        
        cache = get_cache()
        cache_key = json.dumps(
            {
                "question": question,
                "model": self.model,
                "expansion_count": self.expansion_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cached = cache.get("query_expansion", cache_key)
        if cached is not None:
            return cached

        prompt = f"""
Generate {self.expansion_count} different ways to ask the same question.
Each version should focus on different aspects or use different keywords.
Respond with ONLY the questions, one per line, no numbering.

Original question: {question}
"""
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        reply = response.choices[0].message.content.strip()

        expanded = [line.strip() for line in reply.splitlines() if line.strip()]
        result = [question] + expanded[: self.expansion_count]

        cache.set("query_expansion", cache_key, result)
        logger.info(f"Query expanded to {len(result)} variants")
        return result
