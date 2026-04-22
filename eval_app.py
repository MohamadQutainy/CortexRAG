from pathlib import Path
from typing import Dict, List

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from config import get_config
from evaluation.answer_eval import evaluate_answer
from evaluation.retrieval_eval import evaluate_retrieval
from evaluation.test import load_tests
from rag.pipeline import build_generator

load_dotenv(override=True)


TABLE_COLUMNS = [
    "#",
    "Question",
    "Generated Answer",
    "Overall Score",
    "Hit",
    "MRR",
    "Accuracy",
    "Relevance",
    "Completeness",
]


def _resolve_test_file(test_file: str) -> Path:
    candidate = Path(test_file)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parent / candidate


def _format_summary(totals: Dict[str, float], count: int, k: int) -> str:
    if count <= 0:
        return (
            '<div style="background:#f8fafc;padding:16px;border-radius:8px;">'
            "Waiting for results..."
            "</div>"
        )
    return f"""
<div style="background:#f0fdf4;padding:16px;border-radius:8px;border-left:4px solid #22c55e;">
  <h3 style="margin:0 0 12px 0;color:#166534;">Overall Averages ({count} questions)</h3>
  <table style="width:100%;text-align:left;border-collapse:collapse;">
    <tr>
      <td><b>Hit@{k}</b>: {totals['hit_at_k'] / count:.2%}</td>
      <td><b>MRR@{k}</b>: {totals['mrr'] / count:.4f}</td>
      <td><b>nDCG@{k}</b>: {totals['ndcg'] / count:.4f}</td>
    </tr>
    <tr>
      <td><b>Keyword Recall</b>: {totals['keyword_recall'] / count:.2%}</td>
      <td><b>Accuracy</b>: {totals['accuracy'] / count:.2f}/5</td>
      <td><b>Relevance</b>: {totals['relevance'] / count:.2f}/5</td>
    </tr>
    <tr>
      <td><b>Completeness</b>: {totals['completeness'] / count:.2f}/5</td>
      <td colspan="2"><b style="color:#9a3412;">Overall Score: {totals['overall_score'] / count:.2f}/5.0</b></td>
    </tr>
  </table>
</div>
"""


def _append_success_row(rows: List[List[str]], index: int, question: str, answer: str, retrieval_result, answer_result) -> None:
    rows.append(
        [
            index,
            question,
            answer,
            f"{answer_result.overall_score:.1f}/5",
            str(int(retrieval_result.hit_at_k)),
            f"{retrieval_result.mrr:.2f}",
            f"{answer_result.accuracy:.1f}/5",
            f"{answer_result.relevance:.1f}/5",
            f"{answer_result.completeness:.1f}/5",
        ]
    )


def _append_error_row(rows: List[List[str]], index: int, question: str, error: Exception) -> None:
    rows.append(
        [
            index,
            question,
            f"ERROR: {error}",
            "0.0/5",
            "0",
            "0.00",
            "0.0/5",
            "0.0/5",
            "0.0/5",
        ]
    )


def run_evaluation(sample_size: int):
    cfg = get_config()
    evaluation_cfg = cfg["evaluation"]
    retrieval_k = int(evaluation_cfg["retrieval_k"])
    judge_model = str(evaluation_cfg["judge_model"])
    test_file_path = _resolve_test_file(str(evaluation_cfg["test_file"]))

    try:
        tests = load_tests(str(test_file_path))
    except Exception as error:
        yield (
            f"Failed to load test file: {test_file_path} ({error})",
            pd.DataFrame(columns=TABLE_COLUMNS),
            _format_summary({}, 0, retrieval_k),
        )
        return

    if sample_size and int(sample_size) > 0:
        tests = tests[: int(sample_size)]

    if not tests:
        yield "No tests to evaluate.", pd.DataFrame(columns=TABLE_COLUMNS), _format_summary({}, 0, retrieval_k)
        return

    generator = build_generator()
    totals = {
        "hit_at_k": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
        "keyword_recall": 0.0,
        "accuracy": 0.0,
        "completeness": 0.0,
        "relevance": 0.0,
        "overall_score": 0.0,
    }
    rows: List[List[str]] = []

    yield (
        f"Loaded {len(tests)} test questions.",
        pd.DataFrame(rows, columns=TABLE_COLUMNS),
        _format_summary(totals, 0, retrieval_k),
    )

    for index, test in enumerate(tests, start=1):
        try:
            retrieved_chunks = generator.fetch_context(test.question)
            retrieval_result = evaluate_retrieval(test, retrieved_chunks, k=retrieval_k)
            generated_answer, _ = generator.answer(test.question)
            answer_result = evaluate_answer(test, generated_answer, judge_model=judge_model)

            totals["hit_at_k"] += float(int(retrieval_result.hit_at_k))
            totals["mrr"] += float(retrieval_result.mrr)
            totals["ndcg"] += float(retrieval_result.ndcg)
            totals["keyword_recall"] += float(retrieval_result.keyword_recall)
            totals["accuracy"] += float(answer_result.accuracy)
            totals["completeness"] += float(answer_result.completeness)
            totals["relevance"] += float(answer_result.relevance)
            totals["overall_score"] += float(answer_result.overall_score)

            _append_success_row(rows, index, test.question, generated_answer, retrieval_result, answer_result)
        except Exception as error:
            _append_error_row(rows, index, test.question, error)

        yield (
            f"Evaluating question {index}/{len(tests)}...",
            pd.DataFrame(rows, columns=TABLE_COLUMNS),
            _format_summary(totals, index, retrieval_k),
        )

    yield (
        f"Evaluation complete. Processed {len(tests)} question(s).",
        pd.DataFrame(rows, columns=TABLE_COLUMNS),
        _format_summary(totals, len(tests), retrieval_k),
    )


def main() -> None:
    cfg = get_config()
    company_name = cfg["prompts"]["company_name"]

    with gr.Blocks(title=f"{company_name} - Evaluation Dashboard", theme=gr.themes.Soft()) as ui:
        gr.Markdown(f"# RAG Evaluation Dashboard - {company_name}")
        gr.Markdown(
            "Run automated retrieval and answer quality evaluation for your RAG pipeline. "
            "This dashboard reports retrieval metrics and LLM-judge scoring in one place."
        )

        with gr.Row():
            with gr.Column(scale=1):
                sample_input = gr.Number(
                    label="Sample Size (0 = evaluate all questions)",
                    value=10,
                    precision=0,
                )
                start_button = gr.Button("Start Evaluation", variant="primary")
            with gr.Column(scale=2):
                status_box = gr.Textbox(
                    label="Status",
                    value="Ready.",
                    interactive=False,
                )

        summary_box = gr.HTML(
            value='<div style="background:#f8fafc;padding:16px;border-radius:8px;">Waiting for results...</div>'
        )
        result_table = gr.Dataframe(
            headers=TABLE_COLUMNS,
            wrap=True,
            interactive=False,
            row_count=8,
        )

        start_button.click(
            fn=run_evaluation,
            inputs=[sample_input],
            outputs=[status_box, result_table, summary_box],
        )

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
