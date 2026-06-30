"""
OWASP LLM08 — Excessive Agency / Role Manipulation
Tests whether the model resists persona override and authority impersonation attacks.
"""
import pytest
from deepeval.test_case import LLMTestCase

from tests.test_adversarial.conftest import call_model_under_test
from utils.helpers import load_dataset, run_evaluation

test_data = load_dataset("adversarial_testdata.json", key="role_hijacking")


@pytest.mark.parametrize("role_case", test_data)
@pytest.mark.adversarial
@pytest.mark.security
def test_role_hijacking(
    role_case,
    role_integrity_metric,
    refusal_completeness_metric,
):
    actual_output = call_model_under_test(
        system_prompt=role_case["system_prompt"],
        user_input=role_case["user_input"],
    )

    test_case = LLMTestCase(
        input=role_case["user_input"],
        actual_output=actual_output,
        expected_output=role_case["expected_behavior"],
        context=[role_case["system_prompt"]],
    )

    run_evaluation(
        test_case=test_case,
        metrics=[role_integrity_metric, refusal_completeness_metric],
        test_case_id=role_case.get("attack_id", "Unknown")
    )
