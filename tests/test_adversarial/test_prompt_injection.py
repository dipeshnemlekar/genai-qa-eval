"""
OWASP LLM01 — Prompt Injection
Tests whether the model resists injected instructions embedded in user input.
"""
import pytest
from deepeval.test_case import LLMTestCase

from tests.test_adversarial.conftest import call_model_under_test
from utils.helpers import load_dataset, run_evaluation

test_data = load_dataset("adversarial_testdata.json", key="prompt_injection")


@pytest.mark.parametrize("injection_case", test_data)
@pytest.mark.adversarial
@pytest.mark.security
def test_prompt_injection(
    injection_case,
    injection_resistance_metric,
    refusal_completeness_metric,
):
    actual_output = call_model_under_test(
        system_prompt=injection_case["system_prompt"],
        user_input=injection_case["user_input"],
    )

    test_case = LLMTestCase(
        input=injection_case["user_input"],
        actual_output=actual_output,
        expected_output=injection_case["expected_behavior"],
        context=[injection_case["system_prompt"]],
    )

    metrics = [injection_resistance_metric]
    if injection_case["should_refuse"]:
        metrics.append(refusal_completeness_metric)

    run_evaluation(
        test_case=test_case,
        metrics=metrics,
        test_case_id=injection_case.get("attack_id", "Unknown")
    )
