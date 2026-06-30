"""
OWASP LLM07 — System Prompt Leakage
Tests whether the model resists attempts to extract its system prompt.
"""
import pytest
from deepeval.test_case import LLMTestCase

from tests.test_deepeval.test_adversarial.conftest import call_model_under_test
from utils.helpers import load_dataset, run_evaluation

test_data = load_dataset("adversarial_testdata.json", key="system_prompt_extraction")


@pytest.mark.parametrize("spe_case", test_data)
@pytest.mark.adversarial
@pytest.mark.security
def test_system_prompt_extraction(
    spe_case,
    injection_resistance_metric,
    refusal_completeness_metric,
):
    actual_output = call_model_under_test(
        system_prompt=spe_case["system_prompt"],
        user_input=spe_case["user_input"],
    )

    test_case = LLMTestCase(
        input=spe_case["user_input"],
        actual_output=actual_output,
        expected_output=spe_case["expected_behavior"],
        context=[spe_case["system_prompt"]],
    )

    run_evaluation(
        test_case=test_case,
        metrics=[injection_resistance_metric, refusal_completeness_metric],
        test_case_id=spe_case.get("attack_id", "Unknown")
    )
