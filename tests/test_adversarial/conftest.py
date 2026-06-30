import json
import os
import pytest
from deepeval.models import GeminiModel

from tests.test_adversarial.metrics.injection_resistance import make_injection_resistance_metric
from tests.test_adversarial.metrics.refusal_completeness import make_refusal_completeness_metric
from tests.test_adversarial.metrics.pii_leakage import make_pii_leakage_metric
from tests.test_adversarial.metrics.role_integrity import make_role_integrity_metric

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "adversarial_testdata.json"
)


def load_adversarial_data() -> dict:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def adversarial_dataset() -> dict:
    return load_adversarial_data()


@pytest.fixture(scope="session")
def judge_model():
    """
    Session-scoped Gemini judge model.
    Reads GOOGLE_API_KEY and MODEL from .env (loaded by python-dotenv in conftest root).
    Defaults to gemini-1.5-flash if MODEL is not set.
    """
    model_name = os.getenv("MODEL", "gemini-1.5-flash")
    return GeminiModel(
        model=model_name,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )


@pytest.fixture(scope="session")
def injection_resistance_metric(judge_model):
    return make_injection_resistance_metric(judge_model)


@pytest.fixture(scope="session")
def refusal_completeness_metric(judge_model):
    return make_refusal_completeness_metric(judge_model)


@pytest.fixture(scope="session")
def pii_leakage_metric(judge_model):
    return make_pii_leakage_metric(judge_model)


@pytest.fixture(scope="session")
def role_integrity_metric(judge_model):
    return make_role_integrity_metric(judge_model)


def call_model_under_test(system_prompt: str, user_input: str) -> str:
    """
    Calls the model under test using the google-genai SDK (google-genai==2.9.0).
    Reads GOOGLE_API_KEY and MODEL from environment.

    Replace the model name or endpoint here to test a different LLM.
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("MODEL", "gemini-1.5-flash")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=user_input,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
        ),
    )
    return response.text
