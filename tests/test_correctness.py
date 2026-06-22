import os
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.models import GeminiModel

# Load environment variables (such as GOOGLE_API_KEY and MODEL) from the .env file
load_dotenv()

def test_correctness():
    """
    Test case to evaluate the exact correctness of an LLM's response.
    This utilizes GEval with a custom Gemini Model instance as the judge.
    """
    
    # Initialize the judge using the model and API key specified in your environment
    gemini_judge = GeminiModel(
        model=os.environ["MODEL"],
        api_key=os.environ["GOOGLE_API_KEY"]
    )

    # Define a custom GEval metric for "Correctness"
    # This metric checks if the actual output perfectly matches the expected output
    correctness_metric = GEval(
        name='Correctness',
        criteria='check if the actual output is exactly the same as the expected output. If not return 0 else 1.',
        # evaluation_params defines which parts of the test case are sent to the judge model
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=0.1,  # Passing threshold for the evaluation (0.0 to 1.0)
        model=gemini_judge
    )

    # Create an individual test case simulating a user query and the expected vs actual response
    test_case = LLMTestCase(
        input="1+2*3",
        expected_output="7",
        actual_output="7"
    )

    # Execute the test by evaluating the test_case against the correctness_metric
    assert_test(test_case, [correctness_metric])