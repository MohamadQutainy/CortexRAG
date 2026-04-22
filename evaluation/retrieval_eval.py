import math
from typing import List

from pydantic import BaseModel, Field

from evaluation.test import TestQuestion
from rag.chunking.base import ChunkResult
from rag.observability.logger import timed


class RetrievalEval(BaseModel):

    hit_at_k: bool = Field(description="Binary flag: true if at least one relevant chunk was found in the top-k results")
    mrr: float = Field(description="Mean Reciprocal Rank (MRR) based on the rank of the first relevant chunk")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (nDCG) reflecting the ranking quality")
    keywords_found: int = Field(description="Number of ground-truth keywords present in the retrieved context")
    total_keywords: int = Field(description="Total number of ground-truth keywords for the query")
    keyword_recall: float = Field(description="Ratio of retrieved keywords to total ground-truth keywords (0.0 to 1.0)")


def _doc_contains_keyword(doc: ChunkResult, keyword: str) -> bool:
    return keyword.lower() in doc.page_content.lower()


def _is_relevant(doc: ChunkResult, keywords: List[str]) -> bool:
    return any(_doc_contains_keyword(doc, keyword) for keyword in keywords)


def _calculate_dcg(relevances: List[int]) -> float:
    return sum(relevance / math.log2(index + 2) for index, relevance in enumerate(relevances))


@timed(label="retrieval evaluate")
def evaluate_retrieval(
    test: TestQuestion,
    retrieved_docs: List[ChunkResult],
    k: int = 10,
) -> RetrievalEval:
    top_docs = retrieved_docs[:k]
    relevances = [1 if _is_relevant(doc, test.keywords) else 0 for doc in top_docs]

    first_relevant_rank = next(
        (index for index, relevance in enumerate(relevances, start=1) if relevance == 1),
        None,
    )
    hit_at_k = first_relevant_rank is not None
    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    dcg = _calculate_dcg(relevances)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = _calculate_dcg(ideal_relevances)
    ndcg = dcg / idcg if idcg > 0 else 0.0

    keywords_found = sum(
        1
        for keyword in test.keywords
        if any(_doc_contains_keyword(doc, keyword) for doc in top_docs)
    )
    total_keywords = len(test.keywords)
    keyword_recall = keywords_found / total_keywords if total_keywords else 0.0

    return RetrievalEval(
        hit_at_k=hit_at_k,
        mrr=mrr,
        ndcg=ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_recall=keyword_recall,
    )
