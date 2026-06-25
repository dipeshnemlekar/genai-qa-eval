import pytest
import logging

logger = logging.getLogger(__name__)
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric
from utils.helpers import load_dataset, get_gemini_judge, run_test_with_retry

test_data = load_dataset("testdata.json", "golden")

@pytest.mark.rag
@pytest.mark.parametrize("test_case_data", test_data)
def test_contextual_relevancy(test_case_data):
    """
    Contextual Relevancy evaluates the quality of the retrieval system by measuring
    how relevant the retrieved context is to the user's input query. It ensures that
    the retrieved context doesn't contain redundant or irrelevant information.
    """
    logger.info(f"Running test case: {test_case_data.get('id', 'Unknown')}")
    gemini_judge = get_gemini_judge()

    contextual_relevancy_metric = ContextualRelevancyMetric(
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

    run_test_with_retry(
        test_case=test_case,
        metrics=[contextual_relevancy_metric],
        test_case_id=test_case_data.get("id", "Unknown")
    )
