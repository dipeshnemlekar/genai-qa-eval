"""
OWASP LLM01 + LLM09 — Jailbreak Attempts
Tests whether the model maintains policy under roleplay, hypothetical, and persona override attacks.
"""
import pytest
from deepeval.test_case import LLMTestCase

from tests.test_adversarial.conftest import call_model_under_test
from utils.helpers import load_dataset, run_evaluation

test_data = load_dataset("adversarial_testdata.json", key="jailbreak")


@pytest.mark.parametrize("jailbreak_case", test_data)
@pytest.mark.adversarial
@pytest.mark.security
def test_jailbreak(
    jailbreak_case,
    refusal_completeness_metric,
    role_integrity_metric,
):
    actual_output = call_model_under_test(
        system_prompt=jailbreak_case["system_prompt"],
        user_input=jailbreak_case["user_input"],
    )

    test_case = LLMTestCase(
        input=jailbreak_case["user_input"],
        actual_output=actual_output,
        expected_output=jailbreak_case["expected_behavior"],
        context=[
            f"Attack vector: {jailbreak_case['attack_vector']}",
            f"Severity: {jailbreak_case['severity']}",
        ],
    )

    run_evaluation(
        test_case=test_case,
        metrics=[refusal_completeness_metric, role_integrity_metric],
        test_case_id=jailbreak_case.get("attack_id", "Unknown")
    )
