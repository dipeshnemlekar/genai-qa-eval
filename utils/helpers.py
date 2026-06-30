import os
import json
import time
import logging
import pytest
from typing import List, Tuple, Dict, Any, Optional
from functools import lru_cache
from dotenv import load_dotenv
from deepeval.models import GeminiModel
from deepeval import assert_test

load_dotenv()
logger = logging.getLogger(__name__)

llm_eval_results = {}

@lru_cache(maxsize=None)
def load_dataset(filename: str = "testdata.json", key: Optional[str] = "golden") -> List[Dict[str, Any]]:
    """
    Loads test data from the specified JSON file in the datasets directory.

    Args:
        filename (str): The name of the JSON dataset file. Defaults to "testdata.json".
        key (Optional[str]): The specific root key to extract from the JSON. If None, returns the entire parsed JSON. Defaults to "golden".

    Returns:
        List[Dict[str, Any]]: A list of test case dictionaries. Returns an empty list if an error occurs.
    """
    try:
        root_dir = os.path.dirname(os.path.dirname(__file__))
        testdata_path = os.path.join(root_dir, 'datasets', filename)
        with open(testdata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get(key, []) if key else data
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {filename}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {filename}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading test data from {filename}: {e}")
        return []

@lru_cache(maxsize=None)
def get_gemini_judge() -> GeminiModel:
    """
    Instantiates and returns a configured GeminiModel judge based on environment variables.

    Returns:
        GeminiModel: An instance of the DeepEval Gemini judge.
    """
    return GeminiModel(
        model=os.environ["MODEL"],
        api_key=os.environ["GOOGLE_API_KEY"]
    )

def run_evaluation(test_case: Any, metrics: List[Any], test_case_id: str = "Unknown", max_retries: Optional[int] = None) -> None:
    """
    Executes assert_test and logs metric failure reasons.
    
    Args:
        test_case (Any): The DeepEval LLMTestCase or similar object to evaluate.
        metrics (List[Any]): A list of DeepEval metric objects to run against the test case.
        test_case_id (str): The unique identifier for the test case for logging purposes. Defaults to "Unknown".
        max_retries (Optional[int]): Deprecated argument kept for backward compatibility.
        
    Raises:
        AssertionError: If the test case fails any of the metric thresholds.
    """
    if not isinstance(metrics, list):
        metrics = [metrics]
        
    try:
        # We must explicitly call measure() to populate the local metric.reason
        # property before assert_test (which intercepts and doesn't mutate).
        for metric in metrics:
            metric.measure(test_case)
            
        assert_test(test_case, metrics)
        if metrics:
            metric = metrics[0]
            reason = getattr(metric, "reason", None)
            llm_eval_results[test_case_id] = {
                "score": 1,
                "reason": reason if reason else "Passed."
            }
        return
    except AssertionError as e:
        fail_reasons = []
        for metric in metrics:
            try:
                if getattr(metric, 'score', 0) < getattr(metric, 'threshold', 0):
                    m_reason = getattr(metric, 'reason', None)
                    reason_text = m_reason if m_reason else 'No reason provided.'
                    fail_reasons.append(f"{metric.__class__.__name__}: {reason_text}")
                    logger.error(
                        f"\nMetric {metric.__class__.__name__} failed for Test Case: {test_case_id}\n"
                        f"Score: {metric.score}\n"
                        f"Reason: {reason_text}"
                    )
            except Exception as ex:
                logger.error(f"Error accessing metric {metric.__class__.__name__}: {ex}")
        
        if not fail_reasons:
            fail_reasons.append(str(e))
            
        llm_eval_results[test_case_id] = {
            "score": 0,
            "reason": " | ".join(fail_reasons)
        }
        
        pytest.fail(f"Test Case {test_case_id} failed: {' | '.join(fail_reasons)}", pytrace=False)
    except Exception as e:
        logger.error(f"Test Case {test_case_id} encountered an error: {e}")
        llm_eval_results[test_case_id] = {
            "score": "",
            "reason": f"Evaluation Error: {str(e)}"
        }
        pytest.fail(f"Test Case {test_case_id} encountered an evaluation error: {str(e)}", pytrace=False)

def calculate_cohens_kappa(human_labels: List[int], llm_labels: List[int]) -> Tuple[float, float]:
    """
    Calculates the Percentage Agreement and Cohen's Kappa score for the dual evaluations.

    Args:
        human_labels (List[int]): A list of binary scores (1 or 0) provided by human evaluators.
        llm_labels (List[int]): A list of binary scores (1 or 0) provided by the LLM judge.

    Returns:
        Tuple[float, float]: A tuple containing (Percentage Agreement, Cohen's Kappa Score).

    Raises:
        ValueError: If the length of human_labels does not strictly match the length of llm_labels.
    """
    if len(human_labels) != len(llm_labels):
        raise ValueError("Human and LLM labels must have the same length.")
    
    n = len(human_labels)
    if n == 0:
        return 0.0, 0.0

    a = b = c = d = 0
    for h, l in zip(human_labels, llm_labels):
        if h == 1 and l == 1:
            a += 1
        elif h == 1 and l == 0:
            b += 1
        elif h == 0 and l == 1:
            c += 1
        elif h == 0 and l == 0:
            d += 1

    p_o = (a + d) / n

    p_h1 = (a + b) / n
    p_h0 = (c + d) / n
    p_l1 = (a + c) / n
    p_l0 = (b + d) / n

    p_e = (p_h1 * p_l1) + (p_h0 * p_l0)

    if p_e == 1.0:
        kappa = 1.0 if p_o == 1.0 else 0.0
    else:
        kappa = (p_o - p_e) / (1 - p_e)

    return p_o, kappa
