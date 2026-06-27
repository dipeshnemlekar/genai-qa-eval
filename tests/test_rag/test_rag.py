import pytest
import logging

logger = logging.getLogger(__name__)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from utils.helpers import load_dataset, get_gemini_judge, run_evaluation

test_data = load_dataset("rag_data.json", key=None)

@pytest.mark.rag
@pytest.mark.parametrize("test_case_data", test_data)
def test_rag(test_case_data):
    """
    End-to-end RAG quality test using dataset.
    Evaluates an LLM response against Faithfulness, Answer Relevancy, and Professionalism.
    """
    logger.info(f"Running test case: {test_case_data.get('id', 'Unknown')}")
    gemini_judge = get_gemini_judge()

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model=gemini_judge,
        include_reason=True
    )

    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=gemini_judge,
        include_reason=True
    )

    professionalism_metric = GEval(
        name="Professionalism",
        criteria="Determine if the output is professional.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        model=gemini_judge,
        threshold=0.7
    )

    retrieval_context = test_case_data.get("retrive_context", [])
    if not isinstance(retrieval_context, list):
        retrieval_context = [retrieval_context]

    test_case = LLMTestCase(
        input=test_case_data["input"],
        actual_output=test_case_data["actual_output"],
        retrieval_context=retrieval_context,
        expected_output=test_case_data.get("expected_output", "")
    )

    run_evaluation(
        test_case=test_case,
        metrics=[
            faithfulness_metric,
            answer_relevancy_metric,
            professionalism_metric
        ],
        test_case_id=test_case_data.get("id", "Unknown")
    )
