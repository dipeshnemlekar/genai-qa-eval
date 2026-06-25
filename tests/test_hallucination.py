import pytest
import logging

logger = logging.getLogger(__name__)
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric
from utils.helpers import load_dataset, get_gemini_judge, run_test_with_retry

test_data = load_dataset("testdata.json", "golden")

@pytest.mark.rag
@pytest.mark.parametrize("test_case_data", test_data)
def test_hallucination(test_case_data):
    logger.info(f"Running test case: {test_case_data.get('id', 'Unknown')}")
    gemini_judge = get_gemini_judge()

    hallucination_metric = HallucinationMetric(
        threshold=0.5,
        model=gemini_judge,
        include_reason=True
    )

    context = test_case_data.get("retrive_context", [])
    if not isinstance(context, list):
        context = [context]

    test_case = LLMTestCase(
        input=test_case_data["input"],
        actual_output=test_case_data["actual_output"],
        context=context
    )

    run_test_with_retry(
        test_case=test_case,
        metrics=[hallucination_metric],
        test_case_id=test_case_data.get("id", "Unknown")
    )
