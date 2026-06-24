import os
import json
import time
import logging
from dotenv import load_dotenv
from deepeval.models import GeminiModel
from deepeval import assert_test

load_dotenv()
logger = logging.getLogger(__name__)

def load_dataset(filename="testdata.json", key="golden"):
    """
    Loads test data from the datasets directory.
    If 'key' is provided, it returns that key from the json dictionary.
    """
    try:
        # __file__ is utils/helpers.py, root is one level up
        root_dir = os.path.dirname(os.path.dirname(__file__))
        testdata_path = os.path.join(root_dir, 'datasets', filename)
        with open(testdata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get(key, []) if key else data
    except Exception as e:
        logger.error(f"Error loading test data from {filename}: {e}")
        return []

def get_gemini_judge():
    """
    Returns a configured GeminiModel judge based on environment variables.
    """
    return GeminiModel(
        model=os.environ["MODEL"],
        api_key=os.environ["GOOGLE_API_KEY"]
    )

def run_test_with_retry(test_case, metrics, test_case_id="Unknown", max_retries=3):
    """
    Executes assert_test with global retry logic for 429 Too Many Requests errors.
    Logs metric failure reasons and re-raises AssertionError.
    """
    # Ensure metrics is a list
    if not isinstance(metrics, list):
        metrics = [metrics]
        
    retries = 0
    while retries < max_retries:
        try:
            assert_test(test_case, metrics)
            return  # Success
        except AssertionError as e:
            # The test failed the score threshold
            for metric in metrics:
                try:
                    metric.measure(test_case)
                    if getattr(metric, 'score', 0) < getattr(metric, 'threshold', 0):
                        logger.error(
                            f"\nMetric {metric.__class__.__name__} failed for Test Case: {test_case_id}\n"
                            f"Score: {metric.score}\n"
                            f"Reason: {metric.reason}"
                        )
                except Exception as ex:
                    logger.error(f"Error measuring metric {metric.__class__.__name__}: {ex}")
            # Re-raise the assertion error to ensure pytest marks it as FAILED
            raise e
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "resource exhausted" in error_str or "too many requests" in error_str:
                logger.warning(f"Received 429 quota error. Waiting 55 seconds before retry {retries + 1}/{max_retries}...")
                time.sleep(55)
                retries += 1
            else:
                # Re-raise other exceptions immediately
                raise e
                
    raise Exception(f"Failed to execute test case {test_case_id} after {max_retries} retries due to 429 errors.")
