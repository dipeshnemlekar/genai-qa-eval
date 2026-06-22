import os
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams # Import Params
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.models import GeminiModel

# 1. Load the .env file
#load_dotenv()
# Get the key from the environment variable
#api_key = os.getenv("GOOGLE_API_KEY")

def test_rag():
    # 2. Use a valid model name (e.g., gemini-1.5-flash or gemini-2.0-flash)
    gemini_judge = GeminiModel(
        model=os.environ["MODEL"], 
        api_key=os.environ["GOOGLE_API_KEY"]
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model=gemini_judge,
        include_reason=True
    )
    
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=gemini_judge,
        include_reason=True
    )
    

    # 3. Fixing the GEval error from your previous screenshot:
    # You MUST provide 'evaluation_params'
    professionalism_metric = GEval(
        name="Professionalism",
        criteria="Determine if the output is professional.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT], # This fixes the ValueError
        model=gemini_judge,
        threshold=0.7
    )

    test_case = LLMTestCase(
        input="What is the capital of India?",
        actual_output="New Delhi is the capital of India.",
        retrieval_context=["India is a country in South Asia. Its capital is New Delhi."],
        expected_output="New Delhi"
    )

    # Run the tests with all metrics
    assert_test(test_case, [
        faithfulness_metric, 
        answer_relevancy_metric, 
        professionalism_metric
    ])