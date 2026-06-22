import os
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.models import GeminiModel

load_dotenv()

def test_correctness():
    # Initialize the judge using the model you actually have quota for
    gemini_judge = GeminiModel(
        model=os.environ["MODEL"],
        api_key=os.environ["GOOGLE_API_KEY"]
    )

    correctness_metric = GEval(
        name='Correctness',
        criteria='check if the actual output is exactly the same as the expected output. If not return 0 else 1.',
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=0.1,
        model=gemini_judge
    )

    test_case = LLMTestCase(
        input="1+2",
        expected_output="3",
        actual_output="3"
    )

    assert_test(test_case, [correctness_metric])