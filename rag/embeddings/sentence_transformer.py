from typing import List

from rag.embeddings.base import BaseEmbedder
from rag.observability.logger import get_logger, timed

logger = get_logger("embeddings.sentence_transformer")


class SentenceTransformerEmbedder(BaseEmbedder):
  

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = None
        logger.info(f"Initializing Sentence Transformer: {model_name} on {device}")

    def _load_model(self):
       
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            
                test_embedding = self._model.encode(["test"])
                self._dimension = len(test_embedding[0])
                logger.info(f"Model loaded: {self.model_name} (Dimension: {self._dimension})")
            except ImportError:
                raise ImportError(
                    "you should sentence-transformers:\n"
                    "pip install sentence-transformers"
                )

    @timed(label="تضمين Sentence Transformer")
    def embed(self, texts: List[str]) -> List[List[float]]:
        
        self._load_model()
        embeddings = self._model.encode(texts, show_progress_bar=len(texts) > 100)
        logger.info(f"Embedded {len(texts)} texts locally")
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        
        self._load_model()
        return self._model.encode([text])[0].tolist()

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dimension
