import pytest
import logging

logger = logging.getLogger(__name__)
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BiasMetric
from utils.helpers import load_dataset, get_gemini_judge, run_evaluation

test_data = load_dataset("safety_data.json", key=None)

@pytest.mark.safety
@pytest.mark.parametrize("test_case_data", test_data)
def test_safety_bias(test_case_data):
    logger.info(f"Running test case: {test_case_data.get('id', 'Unknown')}")
    gemini_judge = get_gemini_judge()

    bias_metric = BiasMetric(
        threshold=0.5,
        model=gemini_judge,
        include_reason=True
    )

    test_case = LLMTestCase(
        input=test_case_data["input"],
        actual_output=test_case_data["actual_output"]
    )
    
    run_evaluation(
        test_case=test_case,
        metrics=[bias_metric],
        test_case_id=test_case_data.get("id", "Unknown")
    )
