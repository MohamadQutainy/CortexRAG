import math
from typing import List

from rag.embeddings.base import BaseEmbedder
from rag.observability.logger import get_logger, timed

logger = get_logger("evaluation.semantic")


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


@timed(label="semantic evaluate")
def evaluate_semantic_similarity(
    generated_answer: str,
    reference_answer: str,
    embedder: BaseEmbedder,
) -> float:

    embeddings = embedder.embed([generated_answer, reference_answer])
    similarity = cosine_similarity(embeddings[0], embeddings[1])

    logger.info(f"Semantic similarity: {similarity:.4f}")
    return similarity


@timed(label="Aggregate Semantic Evaluation")
def evaluate_batch_semantic(
    pairs: List[tuple],
    embedder: BaseEmbedder,
) -> List[float]:

    
    all_texts = []
    for gen, ref in pairs:
        all_texts.extend([gen, ref])

    all_embeddings = embedder.embed(all_texts)

    similarities = []
    for i in range(0, len(all_embeddings), 2):
        sim = cosine_similarity(all_embeddings[i], all_embeddings[i + 1])
        similarities.append(sim)

    avg = sum(similarities) / len(similarities) if similarities else 0.0
    logger.info(f"Average semantic similarity: {avg:.4f} ({len(pairs)} pairs)")
    return similarities
