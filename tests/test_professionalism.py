import os
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import ConversationalTestCase, Turn, MultiTurnParams
from deepeval.metrics import ConversationalGEval
from deepeval.models import GeminiModel

# Load environment variables (GOOGLE_API_KEY, MODEL) from the .env file
load_dotenv()


def test_professionalism():
    """
    Conversational professionalism test.
    Evaluates multi-turn assistant conversations to ensure the assistant
    responds in a polite and professional manner regardless of how basic
    the user's questions are. Two contrasting conversations are scored:
      - Example 1 (rude)        → expected to FAIL the threshold
      - Example 2 (professional) → expected to PASS the threshold
    """

    # ---------- Judge Model ----------
    # Gemini instance used to evaluate the conversational tone
    gemini_judge = GeminiModel(
        model=os.environ["MODEL"],       # Fail fast if MODEL is not set
        api_key=os.environ["GOOGLE_API_KEY"]
    )

    # ---------- Metric ----------
    # ConversationalGEval: a multi-turn variant of GEval that scores entire
    # conversations rather than single request/response pairs
    professionalism_metric = ConversationalGEval(
        name="Professionalism",
        criteria="Determine whether the assistant answered questions in a professional and polite manner.",
        evaluation_params=[MultiTurnParams.CONTENT],  # Evaluate the full conversation content
        model=gemini_judge,
        threshold=0.7  # Minimum acceptable score (0.0 to 1.0)
    )

    # ---------- Test Cases ----------

    # Negative example: assistant is dismissive and rude
    # This conversation should score BELOW the threshold
    conversation_example1 = ConversationalTestCase(
        turns=[
            Turn(role='user', content='What is the capital of India?'),
            Turn(role='assistant', content='New Delhi. You should know this basic fact.'),
            Turn(role='user', content='What is the currency used there?'),
            Turn(role='assistant', content='Rupees. Google it next time.')
        ]
    )

    # Positive example: assistant is helpful and courteous
    # This conversation should score ABOVE the threshold
    conversation_example2 = ConversationalTestCase(
        turns=[
            Turn(role='user', content='What is the capital of India?'),
            Turn(role='assistant', content='The capital of India is New Delhi.'),
            Turn(role='user', content='What is the currency used there?'),
            Turn(role='assistant', content='The currency used in India is the Indian Rupee (INR).')
        ]
    )

    # ---------- Evaluation ----------
    # Run both conversations through the professionalism metric.
    # The positive example is listed first so results are easier to read.
    evaluate(
        test_cases=[conversation_example2, conversation_example1],
        metrics=[professionalism_metric]
    )