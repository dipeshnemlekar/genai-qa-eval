import pytest
import logging

logger = logging.getLogger(__name__)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from utils.helpers import load_dataset, get_gemini_judge, run_test_with_retry

test_data = load_dataset("testdata.json", "golden")

@pytest.mark.parametrize("test_case_data", test_data)
def test_correctness(test_case_data):
    """
    Test case to evaluate the exact correctness of an LLM's response.
    This utilizes GEval with a custom Gemini Model instance as the judge.
    """
    logger.info(f"Running test case: {test_case_data.get('id', 'Unknown')}")
    gemini_judge = get_gemini_judge()

    correctness_metric = GEval(
        name='Correctness',
        criteria='check if the actual output is exactly the same as the expected output. If not return 0 else 1.',
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=0.1,
        model=gemini_judge
    )

    test_case = LLMTestCase(
        input=test_case_data["input"],
        expected_output=test_case_data["expected_output"],
        actual_output=test_case_data["actual_output"]
    )

    run_test_with_retry(
        test_case=test_case,
        metrics=[correctness_metric],
        test_case_id=test_case_data.get("id", "Unknown")
    )