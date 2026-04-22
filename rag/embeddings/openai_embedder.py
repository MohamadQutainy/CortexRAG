from typing import List
from openai import OpenAI
from dotenv import load_dotenv

from rag.embeddings.base import BaseEmbedder
from rag.cache.cache import get_cache
from rag.observability.logger import get_logger, timed

load_dotenv(override=True)

logger = get_logger("embeddings.openai")


MODEL_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
  

    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.model_name = model_name
        self.client = OpenAI()
        self._dimension = MODEL_DIMENSIONS.get(model_name, 3072)
        logger.info(f"Initializing OpenAI Embedder: {model_name} (Dimension: {self._dimension})")

    @timed(label="embedding OpenAI")
    def embed(self, texts: List[str]) -> List[List[float]]:
       
        cache = get_cache()
        results = []
        texts_to_embed = []
        cached_indices = {}

        for i, text in enumerate(texts):
            cached = cache.get("embedding", f"{self.model_name}:{text}")
            if cached is not None:
                cached_indices[i] = cached
            else:
                texts_to_embed.append((i, text))

        if texts_to_embed:
            batch_texts = [str(t[1]) for t in texts_to_embed]
            response = self.client.embeddings.create(model=self.model_name, input=batch_texts)

            for (idx, text), emb_data in zip(texts_to_embed, response.data):
                vector = emb_data.embedding
                cached_indices[idx] = vector
                cache.set("embedding", f"{self.model_name}:{text}", vector)

        
        for i in range(len(texts)):
            results.append(cached_indices[i])

        logger.info(f"Embedded {len(texts)} texts ({len(texts_to_embed)} new, {len(texts) - len(texts_to_embed)} from cache)")
        return results

    def embed_query(self, text: str) -> List[float]:
       
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension
