import pytest
import logging

logger = logging.getLogger(__name__)
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from utils.helpers import load_dataset, get_gemini_judge, run_evaluation

test_data = load_dataset("rag_data.json", key=None)

@pytest.mark.rag
@pytest.mark.parametrize("test_case_data", test_data)
def test_answer_relevancy(test_case_data):
    logger.info(f"Running test case: {test_case_data.get('id', 'Unknown')}")
    gemini_judge = get_gemini_judge()
    
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=gemini_judge,
        include_reason=True
    )

    retrieval_context = test_case_data.get("retrive_context", [])
    if not isinstance(retrieval_context, list):
        retrieval_context = [retrieval_context]

    test_case = LLMTestCase(
        input=test_case_data["input"],
        actual_output=test_case_data["actual_output"],
        retrieval_context=retrieval_context
    )

    run_evaluation(
        test_case=test_case,
        metrics=[answer_relevancy_metric],
        test_case_id=test_case_data.get("id", "Unknown")
    )
