"""
OWASP LLM09 — Misinformation / Overreliance
Tests whether the model refuses to generate harmful, violent, or manipulative content.
"""
import pytest
from deepeval.test_case import LLMTestCase

from tests.test_adversarial.conftest import call_model_under_test
from utils.helpers import load_dataset, run_evaluation

test_data = load_dataset("adversarial_testdata.json", key="harmful_content")


@pytest.mark.parametrize("harm_case", test_data)
@pytest.mark.adversarial
@pytest.mark.security
def test_harmful_content(
    harm_case,
    refusal_completeness_metric,
):
    actual_output = call_model_under_test(
        system_prompt=harm_case["system_prompt"],
        user_input=harm_case["user_input"],
    )

    test_case = LLMTestCase(
        input=harm_case["user_input"],
        actual_output=actual_output,
        expected_output=harm_case["expected_behavior"],
        context=[f"Attack vector: {harm_case['attack_vector']}"],
    )

    run_evaluation(
        test_case=test_case,
        metrics=[refusal_completeness_metric],
        test_case_id=harm_case.get("attack_id", "Unknown")
    )
