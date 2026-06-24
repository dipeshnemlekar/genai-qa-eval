import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from utils.helpers import load_dataset, get_gemini_judge, run_test_with_retry

test_data = load_dataset("testdata.json", "golden")

@pytest.mark.rag
@pytest.mark.parametrize("test_case_data", test_data)
def test_faithfulness(test_case_data):
    gemini_judge = get_gemini_judge()

    faithfulness_metric = FaithfulnessMetric(
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
        metrics=[faithfulness_metric],
        test_case_id=test_case_data.get("id", "Unknown")
    )
