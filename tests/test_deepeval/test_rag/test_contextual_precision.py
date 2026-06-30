import pytest
import logging

logger = logging.getLogger(__name__)
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from utils.helpers import load_dataset, get_gemini_judge, run_evaluation

test_data = load_dataset("rag_data.json", key=None)

@pytest.mark.rag
@pytest.mark.parametrize("test_case_data", test_data)
def test_contextual_precision(test_case_data):
    """
    Contextual Precision evaluates whether the relevant information in the retrieval context
    is precise and ranked highly. It compares the retrieval context against the expected output
    to determine if the RAG system retrieved highly relevant contexts without much noise.
    """
    logger.info(f"Running test case: {test_case_data.get('id', 'Unknown')}")
    gemini_judge = get_gemini_judge()

    contextual_precision_metric = ContextualPrecisionMetric(
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

    run_evaluation(
        test_case=test_case,
        metrics=[contextual_precision_metric],
        test_case_id=test_case_data.get("id", "Unknown")
    )
