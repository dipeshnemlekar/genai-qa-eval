import os
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import ConversationalTestCase, Turn, LLMTestCaseParams
from deepeval.metrics import ConversationalGEval
from deepeval.models import GeminiModel

# 1. Load the .env file so os.environ can find your keys
load_dotenv()

def test_professionalism():
    gemini_judge = GeminiModel(
        model=os.environ.get("MODEL"), 
        api_key=os.environ.get("GOOGLE_API_KEY")
    )

    professionalism_metric = ConversationalGEval(
        name="Professionalism",
        criteria="Determine whether the assistant answered the questions of the user in a professional and polite manner.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=gemini_judge,
        threshold=0.7 # Add a threshold (0.0 to 1.0)
    )

    conversation_example1 = ConversationalTestCase(
        turns=[
            Turn(role='user', content='What is the capital of India?'),
            Turn(role='assistant', content='New Delhi. You should know this basic fact.'),
            Turn(role='user', content='What is the currency used there?'),
            Turn(role='assistant', content='Rupees. Google it next time.')
        ]
    )

    conversation_example2 = ConversationalTestCase(
        turns=[
            Turn(role='user', content='What is the capital of India?'),
            Turn(role='assistant', content='The capital of India is New Delhi.'),
            Turn(role='user', content='What is the currency used there?'),
            Turn(role='assistant', content='The currency used in India is the Indian Rupee (INR).')
        ]
    )

    # You can pass multiple test cases to a single evaluate call
    evaluate(
        test_cases=[conversation_example2, conversation_example1],
        metrics=[professionalism_metric]
    )