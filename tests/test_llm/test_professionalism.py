import pytest
import logging

logger = logging.getLogger(__name__)
from deepeval.test_case import ConversationalTestCase, Turn, MultiTurnParams
from deepeval.metrics import ConversationalGEval
from utils.helpers import load_dataset, get_gemini_judge, run_evaluation

test_data = load_dataset("conversations.json", "conversations")

@pytest.mark.parametrize("test_case_data", test_data)
def test_professionalism(test_case_data):
    """
    Conversational professionalism test using dataset.
    Evaluates multi-turn assistant conversations.
    """
    logger.info(f"Running test case: {test_case_data.get('id', 'Unknown')}")
    gemini_judge = get_gemini_judge()

    professionalism_metric = ConversationalGEval(
        name="Professionalism",
        criteria="Determine whether the assistant answered questions in a professional and polite manner.",
        evaluation_params=[MultiTurnParams.CONTENT],
        model=gemini_judge,
        threshold=0.7
    )

    turns = []
    for turn_data in test_case_data.get("turns", []):
        turns.append(Turn(role=turn_data["role"], content=turn_data["content"]))

    test_case = ConversationalTestCase(turns=turns)

    run_evaluation(
        test_case=test_case,
        metrics=[professionalism_metric],
        test_case_id=test_case_data.get("id", "Unknown")
    )
