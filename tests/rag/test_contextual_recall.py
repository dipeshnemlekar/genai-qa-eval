import pytest
import logging

logger = logging.getLogger(__name__)
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric
from utils.helpers import load_dataset, get_gemini_judge, run_test_with_retry

test_data = load_dataset("testdata.json", "golden")

@pytest.mark.rag
@pytest.mark.parametrize("test_case_data", test_data)
def test_contextual_recall(test_case_data):
    """
    Contextual Recall evaluates whether the retrieval context contains all the necessary
    information required to arrive at the expected output. It measures whether the RAG
    system successfully recalled all relevant facts.
    """
    logger.info(f"Running test case: {test_case_data.get('id', 'Unknown')}")
    gemini_judge = get_gemini_judge()

    contextual_recall_metric = ContextualRecallMetric(
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
        expected_output=test_case_data.get("expected_output", ""),
        retrieval_context=retrieval_context
    )

    run_test_with_retry(
        test_case=test_case,
        metrics=[contextual_recall_metric],
        test_case_id=test_case_data.get("id", "Unknown")
    )
