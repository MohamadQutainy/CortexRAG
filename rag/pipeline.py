from config import create_reranker, create_retriever, get_config
from rag.generation.generator import RAGGenerator
from rag.query.expander import QueryExpander
from rag.query.rewriter import QueryRewriter


def build_generator() -> RAGGenerator:
   
    cfg = get_config()
    retrieval_cfg = cfg["retrieval"]
    llm_model = cfg["llm"]["model_name"]

    query_rewriter = None
    if retrieval_cfg.get("enable_query_rewriting", True):
        query_rewriter = QueryRewriter(model=llm_model)

    query_expander = None
    if retrieval_cfg.get("enable_query_expansion", False):
        query_expander = QueryExpander(
            model=llm_model,
            expansion_count=retrieval_cfg.get("expansion_count", 3),
        )

    return RAGGenerator(
        retriever=create_retriever(),
        reranker=create_reranker(),
        query_rewriter=query_rewriter,
        query_expander=query_expander,
        model=llm_model,
        company_name=cfg["prompts"]["company_name"],
        final_k=retrieval_cfg["final_k"],
        system_prompt_template=cfg["prompts"]["system_prompt"],
    )
