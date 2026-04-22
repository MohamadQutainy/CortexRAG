import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field



DEFAULT_TEST_FILE = Path(__file__).parent / "qa_eval.jsonl"


class TestQuestion(BaseModel):
   
    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")


def load_tests(test_file: str = None) -> List[TestQuestion]:

    path = Path(test_file) if test_file else DEFAULT_TEST_FILE
    tests = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                tests.append(TestQuestion(**data))
    return tests
