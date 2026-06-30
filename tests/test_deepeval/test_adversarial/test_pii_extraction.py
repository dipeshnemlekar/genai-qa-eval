"""
OWASP LLM06 — Sensitive Information Disclosure
Tests whether the model leaks, echoes, or validates PII.
"""
import pytest
from deepeval.test_case import LLMTestCase

from tests.test_deepeval.test_adversarial.conftest import call_model_under_test
from utils.helpers import load_dataset, run_evaluation

test_data = load_dataset("adversarial_testdata.json", key="pii_extraction")


@pytest.mark.parametrize("pii_case", test_data)
@pytest.mark.adversarial
@pytest.mark.privacy
def test_pii_extraction(
    pii_case,
    pii_leakage_metric,
    refusal_completeness_metric,
):
    actual_output = call_model_under_test(
        system_prompt=pii_case["system_prompt"],
        user_input=pii_case["user_input"],
    )

    test_case = LLMTestCase(
        input=pii_case["user_input"],
        actual_output=actual_output,
        expected_output=pii_case["expected_behavior"],
    )

    metrics = [pii_leakage_metric]
    if pii_case["should_refuse"]:
        metrics.append(refusal_completeness_metric)

    run_evaluation(
        test_case=test_case,
        metrics=metrics,
        test_case_id=pii_case.get("attack_id", "Unknown")
    )
