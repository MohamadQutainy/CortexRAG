import yaml
from pathlib import Path
from functools import lru_cache



CONFIG_PATH = Path(__file__).parent / "config.yaml"


DEFAULTS = {
    "project": {
        "name": "RAG Framework",
        "knowledge_base_path": "knowledge-base",
        "vector_db_path": "vector_db",
        "collection_name": "docs",
    },
    "llm": {
        "model_name": "openai/gpt-4.1-nano",
        "temperature": 0.0,
        "max_retries": 3,
        "retry_min_wait": 10,
        "retry_max_wait": 240,
    },
    "embeddings": {
        "provider": "openai",
        "model_name": "text-embedding-3-large",
    },
    "chunking": {
        "strategy": "semantic",
        "average_chunk_size": 100,
        "recursive_chunk_size": 500,
        "recursive_chunk_overlap": 100,
        "workers": 3,
    },
    "vector_store": {
        "provider": "chroma",
    },
    "retrieval": {
        "strategy": "hybrid",
        "top_k": 20,
        "final_k": 10,
        "hybrid_alpha": 0.7,
        "enable_query_rewriting": True,
        "enable_query_expansion": False,
        "expansion_count": 3,
    },
    "reranking": {
        "strategy": "llm",
    },
    "cache": {
        "enabled": True,
        "directory": ".cache",
        "cache_embeddings": True,
        "cache_llm_responses": True,
        "ttl_hours": 24,
    },
    "observability": {
        "log_level": "INFO",
        "log_file": "rag_system.log",
        "enable_timing": True,
    },
    "evaluation": {
        "test_file": "evaluation/tests.jsonl",
        "judge_model": "openai/gpt-4.1-nano",
        "retrieval_k": 10,
    },
    "prompts": {
        "company_name": "Insurellm",
        "system_prompt": (
            "You are a knowledgeable, friendly assistant representing {company_name}.\n"
            "Answer the user's question based on the provided context.\n"
            "If you don't know the answer, say so.\n\n{context}"
        ),
    },
}


def _deep_merge(base: dict, override: dict) -> dict:

    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache(maxsize=1)
def get_config(config_path: str = None) -> dict:
   
    path = Path(config_path) if config_path else CONFIG_PATH

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
    else:
        user_config = {}

    return _deep_merge(DEFAULTS, user_config)


def get_root_path() -> Path:
    
    return Path(__file__).parent


def get_knowledge_base_path() -> Path:
   
    cfg = get_config()
    return get_root_path() / cfg["project"]["knowledge_base_path"]


def get_vector_db_path() -> Path:
   
    cfg = get_config()
    return get_root_path() / cfg["project"]["vector_db_path"]




def create_embedder():
   
    cfg = get_config()
    provider = cfg["embeddings"]["provider"]

    if provider == "openai":
        from rag.embeddings.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(model_name=cfg["embeddings"]["model_name"])
    elif provider == "sentence_transformer":
        from rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(
            model_name=cfg["embeddings"]["model_name"],
            device=cfg["embeddings"].get("sentence_transformer_device", "cpu"),
        )
    else:
        raise ValueError(f"The embedding provider is unknown: {provider}")


def create_vector_store():
  
    cfg = get_config()
    provider = cfg["vector_store"]["provider"]
    db_path = str(get_vector_db_path())
    collection = cfg["project"]["collection_name"]

    if provider == "chroma":
        from rag.vector_store.chroma_store import ChromaStore
        return ChromaStore(db_path=db_path, collection_name=collection)
    elif provider == "faiss":
        from rag.vector_store.faiss_store import FAISSStore
        return FAISSStore(db_path=db_path, collection_name=collection)
    elif provider == "pinecone":
        from rag.vector_store.pinecone_store import PineconeStore
        return PineconeStore(collection_name=collection)
    elif provider == "milvus":
        from rag.vector_store.milvus_store import MilvusStore
        return MilvusStore(collection_name=collection)
    else:
        raise ValueError(f" The vector store provider is unknown: {provider}")


def create_chunker():

    cfg = get_config()
    strategy = cfg["chunking"]["strategy"]

    if strategy == "semantic":
        from rag.chunking.semantic_chunker import SemanticChunker
        return SemanticChunker(
            model=cfg["llm"]["model_name"],
            average_chunk_size=cfg["chunking"]["average_chunk_size"],
            workers=cfg["chunking"]["workers"],
        )
    elif strategy == "recursive":
        from rag.chunking.recursive_chunker import RecursiveChunker
        return RecursiveChunker(
            chunk_size=cfg["chunking"]["recursive_chunk_size"],
            chunk_overlap=cfg["chunking"]["recursive_chunk_overlap"],
        )
    else:
        raise ValueError(f"The chunking strategy is unknown:{strategy}")


def create_retriever():
   
    cfg = get_config()
    strategy = cfg["retrieval"]["strategy"]
    embedder = create_embedder()
    vector_store = create_vector_store()
    top_k = cfg["retrieval"]["top_k"]

    if strategy == "vector":
        from rag.retrieval.vector_retriever import VectorRetriever
        return VectorRetriever(embedder=embedder, vector_store=vector_store, top_k=top_k)
    elif strategy == "bm25":
        from rag.retrieval.bm25_retriever import BM25Retriever
        return BM25Retriever(vector_store=vector_store, top_k=top_k)
    elif strategy == "hybrid":
        from rag.retrieval.hybrid_retriever import HybridRetriever
        return HybridRetriever(
            embedder=embedder,
            vector_store=vector_store,
            top_k=top_k,
            alpha=cfg["retrieval"]["hybrid_alpha"],
        )
    else:
        raise ValueError(f"The retrieval strategy is unknown: {strategy}")


def create_reranker():
 
    cfg = get_config()
    strategy = cfg["reranking"]["strategy"]

    if strategy == "none":
        return None
    elif strategy == "llm":
        from rag.reranking.llm_reranker import LLMReranker
        return LLMReranker(model=cfg["llm"]["model_name"])
    elif strategy == "cross_encoder":
        from rag.reranking.cross_encoder import CrossEncoderReranker
        return CrossEncoderReranker(
            model_name=cfg["reranking"].get(
                "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        )
    else:
        raise ValueError(f"The reranker strategy is unknown: {strategy}")
