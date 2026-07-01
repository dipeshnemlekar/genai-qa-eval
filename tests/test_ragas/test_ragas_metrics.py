import pytest
import os
import logging
from dotenv import load_dotenv

from utils.helpers import load_dataset, llm_eval_results
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate

try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    HAS_RAGAS_WRAPPERS = True
except ImportError:
    HAS_RAGAS_WRAPPERS = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

load_dotenv()
logger = logging.getLogger(__name__)

# Load the dataset (rag_data.json)
rag_test_data = load_dataset("rag_data.json", key=None)

@pytest.fixture(scope="module")
def ragas_llm():
    if HAS_GOOGLE_GENAI and "GOOGLE_API_KEY" in os.environ:
        model_name = os.environ.get("MODEL", "gemini-1.5-pro")
        llm = ChatGoogleGenerativeAI(model=model_name)
        if HAS_RAGAS_WRAPPERS:
            return LangchainLLMWrapper(llm)
        return llm
    print("Warning: ragas_llm is None. HAS_GOOGLE_GENAI=", HAS_GOOGLE_GENAI)
    return None

@pytest.fixture(scope="module")
def ragas_embeddings():
    if HAS_GOOGLE_GENAI and "GOOGLE_API_KEY" in os.environ:
        emb = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT"
        )
        if HAS_RAGAS_WRAPPERS:
            return LangchainEmbeddingsWrapper(emb)
        return emb
    return None

@pytest.mark.parametrize("rag_case", rag_test_data)
def test_ragas_metrics(rag_case, ragas_llm, ragas_embeddings):
    """
    Evaluates a RAG test case using the Ragas framework.
    We test faithfulness, answer_relevancy, and context_precision.
    """
    test_case_id = rag_case.get("id", "Unknown")
    
    # 1. Map to SingleTurnSample
    sample = SingleTurnSample(
        user_input=rag_case.get("input", ""),
        response=rag_case.get("actual_output", ""),
        retrieved_contexts=rag_case.get("retrive_context", []),
        reference=rag_case.get("ground_truth", "")
    )
    dataset = EvaluationDataset(samples=[sample])
    
    # 2. Evaluate using Ragas
    metrics = [faithfulness, answer_relevancy, context_precision]
    
    kwargs = {}
    if ragas_llm:
        kwargs['llm'] = ragas_llm
    if ragas_embeddings:
        kwargs['embeddings'] = ragas_embeddings
        
    try:
        result = evaluate(
            dataset,
            metrics=metrics,
            raise_exceptions=False,
            **kwargs
        )
        
        # 3. Process results
        scores = result.to_pandas().iloc[0]
        
        f_score = scores.get('faithfulness', 0)
        a_score = scores.get('answer_relevancy', 0)
        c_score = scores.get('context_precision', 0)
        
        # Make sure they are numbers
        f_score = f_score if not set([f_score]).issubset({None}) else 0
        a_score = a_score if not set([a_score]).issubset({None}) else 0
        c_score = c_score if not set([c_score]).issubset({None}) else 0
        
        reason = f"Faithfulness: {f_score:.2f}, Relevancy: {a_score:.2f}, Precision: {c_score:.2f}"
        
        # Store for dashboard and CSV reporting
        llm_eval_results[test_case_id] = {
            "score": 1 if min(f_score, a_score, c_score) >= 0.5 else 0,
            "reason": reason,
            "input": sample.user_input,
            "expected_output": sample.reference,
            "actual_output": sample.response
        }
        
        # Assertions
        assert f_score >= 0.5, f"Faithfulness too low: {f_score}"
        assert a_score >= 0.5, f"Answer relevancy too low: {a_score}"
        assert c_score >= 0.5, f"Context precision too low: {c_score}"
        
    except AssertionError as e:
        # Re-raise AssertionErrors so pytest registers a failure correctly
        raise
    except Exception as e:
        logger.error(f"Ragas evaluation failed for {test_case_id}: {e}")
        llm_eval_results[test_case_id] = {
            "score": 0,
            "reason": f"Error: {str(e)}",
            "input": sample.user_input,
            "expected_output": sample.reference,
            "actual_output": sample.response
        }
        pytest.fail(f"Ragas evaluation failed: {e}")
