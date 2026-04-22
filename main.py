import argparse

from dotenv import load_dotenv

load_dotenv(override=True)

from config import create_chunker, create_embedder, create_vector_store, get_config, get_knowledge_base_path
from rag.observability.logger import get_logger, setup_logging
from rag.pipeline import build_generator


def cmd_ingest():
  
    from rag.ingestion.loader import load_documents
    from rag.ingestion.preprocessor import preprocess_documents

    logger = get_logger("main.ingest")
    logger.info("=" * 60)
    logger.info("Starting ingestion process")
    logger.info("=" * 60)

    kb_path = get_knowledge_base_path()
    documents = load_documents(kb_path)
    if not documents:
        logger.error(f"No documents found in: {kb_path}")
        return

    documents = preprocess_documents(documents)
    chunker = create_chunker()
    chunks = chunker.chunk_many(documents)
    logger.info(f"Created {len(chunks)} chunks")

    embedder = create_embedder()
    vector_store = create_vector_store()


    vector_store.delete_collection()

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    all_embeddings = []
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        all_embeddings.extend(embedder.embed(batch))

    ids = [str(index) for index in range(len(chunks))]
    vector_store.add(texts, all_embeddings, metadatas, ids)

    logger.info("=" * 60)
    logger.info(f"Ingestion successful: {vector_store.count()} documents in the database")
    logger.info("=" * 60)


def cmd_query(question: str):
  
    logger = get_logger("main.query")
    logger.info(f"Question: {question}")

    generator = build_generator()
    answer, chunks = generator.answer(question)

    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print(f"{'=' * 60}")
    print(f"\nAnswer:\n{answer}")
    print(f"\n{'=' * 60}")
    print(f"Sources ({len(chunks)}):")
    for index, chunk in enumerate(chunks[:5], start=1):
        print(f"  [{index}] {chunk.metadata.get('source', 'unknown')}")
    print(f"{'=' * 60}\n")


def cmd_evaluate(test_number: int | None = None, sample: int | None = None):
    
    from evaluation.answer_eval import evaluate_answer
    from evaluation.retrieval_eval import evaluate_retrieval
    from evaluation.test import load_tests

    logger = get_logger("main.evaluate")
    cfg = get_config()
    tests = load_tests(cfg["evaluation"]["test_file"])
    generator = build_generator()
    retrieval_k = cfg["evaluation"]["retrieval_k"]

    if test_number is not None:
        if test_number < 0 or test_number >= len(tests):
            print(f"Error: Question number must be between 0 and {len(tests) - 1}")
            return

        test = tests[test_number]
        chunks = generator.fetch_context(test.question)
        retrieval_result = evaluate_retrieval(test, chunks, k=retrieval_k)
        answer, _ = generator.answer(test.question)
        answer_result = evaluate_answer(
            test,
            answer,
            judge_model=cfg["evaluation"]["judge_model"],
        )

        print(f"\n{'=' * 60}")
        print(f"Question #{test_number}")
        print(f"{'=' * 60}")
        print(f"Question: {test.question}")
        print(f"Category: {test.category}")
        print(f"Reference Answer: {test.reference_answer}")
        print("\n--- Retrieval Evaluation ---")
        print(f"Hit@{retrieval_k}: {retrieval_result.hit_at_k}")
        print(f"MRR@{retrieval_k}: {retrieval_result.mrr:.4f}")
        print(f"nDCG@{retrieval_k}: {retrieval_result.ndcg:.4f}")
        print(
            f"Recall@{retrieval_k} Keywords: "
            f"{retrieval_result.keywords_found}/{retrieval_result.total_keywords} "
            f"({retrieval_result.keyword_recall:.1%})"
        )
        print("\n--- Answer Evaluation ---")
        print(f"Generated Answer: {answer}")
        print(f"\nOverall Score: {answer_result.overall_score:.2f}/5")
        print(f"Feedback: {answer_result.feedback}")
        print(f"Accuracy: {answer_result.accuracy:.1f}/5")
        print(f"Completeness: {answer_result.completeness:.1f}/5")
        print(f"Relevance: {answer_result.relevance:.1f}/5")
        print(f"{'=' * 60}\n")
        return

    eval_tests = tests[:sample] if sample else tests
    print(f"\nEvaluating {len(eval_tests)} questions...")

    totals = {
        "hit_at_k": 0,
        "mrr": 0.0,
        "ndcg": 0.0,
        "keyword_recall": 0.0,
        "accuracy": 0.0,
        "completeness": 0.0,
        "relevance": 0.0,
        "overall_score": 0.0,
    }
    count = 0

    for index, test in enumerate(eval_tests, start=1):
        print(f"\n[{index}/{len(eval_tests)}] {test.question[:50]}...")
        try:
            chunks = generator.fetch_context(test.question)
            retrieval_result = evaluate_retrieval(test, chunks, k=retrieval_k)

            answer, _ = generator.answer(test.question)
            answer_result = evaluate_answer(
                test,
                answer,
                judge_model=cfg["evaluation"]["judge_model"],
            )

            totals["hit_at_k"] += int(retrieval_result.hit_at_k)
            totals["mrr"] += retrieval_result.mrr
            totals["ndcg"] += retrieval_result.ndcg
            totals["keyword_recall"] += retrieval_result.keyword_recall
            totals["accuracy"] += answer_result.accuracy
            totals["completeness"] += answer_result.completeness
            totals["relevance"] += answer_result.relevance
            totals["overall_score"] += answer_result.overall_score
            count += 1

            print(
                f"  Hit@{retrieval_k}={int(retrieval_result.hit_at_k)} | "
                f"MRR={retrieval_result.mrr:.2f} | "
                f"nDCG={retrieval_result.ndcg:.2f} | "
                f"Recall={retrieval_result.keyword_recall:.0%} | "
                f"Overall={answer_result.overall_score:.1f}/5"
            )
        except Exception as exc:
            logger.error(f"Error in question {index - 1}: {exc}")

    if count == 0:
        print("\nNo evaluations completed successfully.")
        return

    print(f"\n{'=' * 60}")
    print(f"Overall Results ({count} questions)")
    print(f"{'=' * 60}")
    print(f"Average Hit@{retrieval_k}: {totals['hit_at_k'] / count:.2%}")
    print(f"Average MRR@{retrieval_k}: {totals['mrr'] / count:.4f}")
    print(f"Average nDCG@{retrieval_k}: {totals['ndcg'] / count:.4f}")
    print(f"Average Keyword Recall@{retrieval_k}: {totals['keyword_recall'] / count:.2%}")
    print(f"Average Accuracy: {totals['accuracy'] / count:.2f}/5")
    print(f"Average Completeness: {totals['completeness'] / count:.2f}/5")
    print(f"Average Relevance: {totals['relevance'] / count:.2f}/5")
    print(f"Average Overall Score: {totals['overall_score'] / count:.2f}/5")
    print(f"{'=' * 60}\n")


def main():
    
    parser = argparse.ArgumentParser(
        description="RAG Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest
  python main.py query "Who founded Insurellm?"
  python main.py evaluate --test 5
  python main.py evaluate --sample 10
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="The operation to perform")
    
    # Ingest command
    subparsers.add_parser("ingest", help="Ingest documents and build the knowledge base")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question to the system")
    query_parser.add_argument("question", type=str, help="The text of the query/question")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run system evaluation metrics")
    eval_parser.add_argument("--test", type=int, default=None, help="Index of a specific test case to run")
    eval_parser.add_argument("--sample", type=int, default=None, help="Number of random samples to evaluate")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cfg = get_config()
    setup_logging(
        log_level=cfg["observability"]["log_level"],
        log_file=cfg["observability"]["log_file"],
    )

    if args.command == "ingest":
        cmd_ingest()
    elif args.command == "query":
        cmd_query(args.question)
    elif args.command == "evaluate":
        cmd_evaluate(test_number=args.test, sample=args.sample)


if __name__ == "__main__":
    main()
