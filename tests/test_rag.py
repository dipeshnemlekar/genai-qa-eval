import os
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.models import GeminiModel

# Load environment variables (GOOGLE_API_KEY, MODEL) from the .env file
load_dotenv()


def test_rag():
    """
    End-to-end RAG quality test.
    Evaluates an LLM response against three complementary metrics:
      1. Faithfulness  – Is the answer grounded in the retrieved context?
      2. Answer Relevancy – Does the answer actually address the user's question?
      3. Professionalism – Is the response written in a professional tone?
    """

    # ---------- Judge Model ----------
    # A shared Gemini instance used by every metric to score the response
    gemini_judge = GeminiModel(
        model=os.environ["MODEL"],
        api_key=os.environ["GOOGLE_API_KEY"]
    )

    # ---------- Metrics ----------

    # Faithfulness: checks that the actual output does not hallucinate
    # beyond what was provided in the retrieval_context
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model=gemini_judge,
        include_reason=True  # Attach a human-readable explanation to the score
    )

    # Answer Relevancy: checks that the actual output is relevant
    # to the original user input / question
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=gemini_judge,
        include_reason=True
    )

    # Professionalism: a custom GEval metric that assesses tone and style
    professionalism_metric = GEval(
        name="Professionalism",
        criteria="Determine if the output is professional.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],  # Only the answer is needed for tone evaluation
        model=gemini_judge,
        threshold=0.7
    )

    # ---------- Test Case ----------
    # Simulates a RAG pipeline: user asks a question, retrieval_context is fetched,
    # and the LLM produces an actual_output that we compare against expected_output
    test_case = LLMTestCase(
        input="What is the capital of India?",
        actual_output="New Delhi is the capital of India.",
        retrieval_context=["India is a country in South Asia. Its capital is New Delhi."],
        expected_output="New Delhi"
    )

    # ---------- Assertion ----------
    # Run all three metrics against the test case; the test fails if any metric
    # scores below its configured threshold
    assert_test(test_case, [
        faithfulness_metric,
        answer_relevancy_metric,
        professionalism_metric
    ])