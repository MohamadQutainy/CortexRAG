from pydantic import BaseModel, Field
from litellm import completion
from tenacity import retry, wait_exponential, stop_after_attempt

from evaluation.test import TestQuestion
from rag.generation.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT
from rag.observability.logger import get_logger, timed

logger = get_logger("evaluation.answer")


class AnswerEval(BaseModel):
    

    feedback: str = Field(description="Qualitative feedback on the answer quality and grounding")
    accuracy: float = Field(description="Factuality and grounding score (1 to 5)")
    completeness: float = Field(description="Coverage of all query sub-points (1 to 5)")
    relevance: float = Field(description="Alignment with user intent and conciseness (1 to 5)")
    overall_score: float = Field(description="Weighted aggregate score (1 to 5)")


def _calculate_overall_score(accuracy: float, completeness: float, relevance: float) -> float:
    return round((accuracy * 0.5) + (completeness * 0.3) + (relevance * 0.2), 2)


@retry(wait=wait_exponential(multiplier=1, min=10, max=240), stop=stop_after_attempt(3))
@timed(label="تقييم الإجابة")
def evaluate_answer(
    test: TestQuestion,
    generated_answer: str,
    judge_model: str = "openai/gpt-4.1-nano",
) -> AnswerEval:
    
    user_prompt = JUDGE_USER_PROMPT.format(
        question=test.question,
        generated_answer=generated_answer,
        reference_answer=test.reference_answer,
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = completion(
        model=judge_model,
        messages=messages,
        response_format=AnswerEval,
    )

    raw = AnswerEval.model_validate_json(response.choices[0].message.content)
    result = AnswerEval(
        feedback=raw.feedback,
        accuracy=raw.accuracy,
        completeness=raw.completeness,
        relevance=raw.relevance,
        overall_score=_calculate_overall_score(
            raw.accuracy,
            raw.completeness,
            raw.relevance,
        ),
    )

    logger.info(
        "تقييم: "
        f"accuracy={result.accuracy:.1f} | "
        f"completeness={result.completeness:.1f} | "
        f"relevance={result.relevance:.1f} | "
        f"overall={result.overall_score:.2f}"
    )
    return result
