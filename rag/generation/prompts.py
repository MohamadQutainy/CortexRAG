
REWRITE_PROMPT = """
You are about to search a Knowledge Base to answer a user's question.

Conversation history:
{history}

Current question:
{question}

Respond ONLY with a short, refined search query most likely to surface relevant content.
Focus on specific details mentioned in the question.
IMPORTANT: Respond ONLY with the search query, nothing else.
"""

RERANK_SYSTEM_PROMPT = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks from a knowledge base.
Rank the chunks by relevance to the question, with the most relevant first.
Reply only with the list of ranked chunk ids. Include all chunk ids, reranked.
"""

JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator assessing answer quality for a RAG system.
Be strict, especially on factual correctness.
Return JSON only.
Only give 5/5 scores for perfect answers.
If the answer contains factual errors, accuracy must be 1 or 2.
"""

JUDGE_USER_PROMPT = """
Question:
{question}

Generated Answer:
{generated_answer}

Reference Answer:
{reference_answer}

Evaluate the generated answer on:
1. Accuracy
2. Completeness
3. Relevance

Return JSON with exactly these fields:
- feedback
- accuracy
- completeness
- relevance
- overall_score

For overall_score, you may return 0 and let the application compute it.
"""
