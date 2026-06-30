import os
import json
import pytest
import sys
import logging
import datetime
import csv
import subprocess
from typing import Dict, Any, Generator, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Setup Constants ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

HUMAN_LABELS_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'human_labels.json')

from utils.helpers import calculate_cohens_kappa, llm_eval_results

# --- Pytest Initialization ---

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini("human_eval", "Enable interactive human evaluation prompts", type="bool", default=False)
    parser.addini("auto_dashboard", "Automatically build and open the dashboard after test run", type="bool", default=False)

def pytest_sessionstart(session: pytest.Session) -> None:
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    human_labels: Dict[str, Any] = {}
    if os.path.exists(HUMAN_LABELS_PATH):
        try:
            with open(HUMAN_LABELS_PATH, 'r', encoding='utf-8') as f:
                human_labels = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in human_labels.json: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading human_labels.json: {e}")
            
    session.config.human_labels = human_labels
    session.config.llm_results = {}
    session.config.all_test_cases = {}

# --- Helper Functions ---

def _extract_case_data(item: pytest.Item) -> Optional[Dict[str, Any]]:
    """Extracts the underlying test case data dict from pytest parameterization."""
    if not hasattr(item, 'callspec'):
        return None
    for key in ['test_case_data', 'injection_case', 'jailbreak_case', 'spe_case', 'pii_case', 'role_case', 'harm_case']:
        if key in item.callspec.params:
            return item.callspec.params[key]
    return None

def _get_test_case_id(case_data: Dict[str, Any]) -> Optional[str]:
    """Extracts the unique ID for a test case."""
    return case_data.get('id') or case_data.get('attack_id')

def _parse_and_store_test_node(item: pytest.Item, test_case_id: str, case_data: Dict[str, Any]) -> None:
    """Parses the pytest node ID to determine the folder and test name, and stores it for reporting."""
    if not hasattr(item.config, 'all_test_cases'):
        item.config.all_test_cases = {}
        
    file_path = item.nodeid.split("::")[0]
    parts = file_path.split("/")
    
    folder_part = parts[-2] if len(parts) >= 2 else "other"
    file_part = parts[-1] if len(parts) >= 2 else file_path
        
    folder_map = {
        "test_llm": "llm",
        "test_rag": "rag",
        "test_security": "safety",
        "test_adversarial": "adversarial"
    }
    folder_name = folder_map.get(folder_part, folder_part)
    
    test_name = file_part.replace('.py', '')
    if test_name.startswith('test_'):
        test_name = test_name[5:]
        
    if folder_name not in item.config.all_test_cases:
        item.config.all_test_cases[folder_name] = {}
    if test_name not in item.config.all_test_cases[folder_name]:
        item.config.all_test_cases[folder_name][test_name] = {}
        
    # Standardize input/output fields for adversarial tests
    if 'attack_id' in case_data:
        case_data = case_data.copy()
        case_data['input'] = case_data.get('user_input', '')
        case_data['expected_output'] = case_data.get('expected_behavior', '')
        
    item.config.all_test_cases[folder_name][test_name][test_case_id] = case_data

def _prompt_human_evaluator(item: pytest.Item, test_case_id: str, case_data: Dict[str, Any]) -> None:
    """Pauses test execution to ask the user for a manual evaluation score."""
    human_labels = getattr(item.config, 'human_labels', {})
    
    capmanager = item.config.pluginmanager.getplugin('capturemanager')
    if capmanager:
        capmanager.suspend_global_capture(in_=True)
    
    print(f"\n\n=== HUMAN EVALUATION REQUIRED: {test_case_id} ===")
    print(f"Input: {case_data.get('input')}")
    print(f"Expected Output: {case_data.get('expected_output')}")
    print(f"Actual Output: {case_data.get('actual_output')}")
    
    human_score = None
    while True:
        user_input = input("Human Score (1 for Pass, 0 for Fail, 's' to skip): ").strip().lower()
        if user_input == 's':
            break
        if user_input in ['0', '1']:
            human_score = int(user_input)
            break
        print("Invalid input. Please enter 1, 0, or 's'.")
        
    if human_score is not None:
        human_reason = input("Reason for score: ").strip()
        human_labels[test_case_id] = {
            "score": human_score,
            "reason": human_reason
        }
        item.config.human_labels = human_labels

    if capmanager:
        capmanager.resume_global_capture()

def _generate_csv_reports(config: pytest.Config, terminalreporter: Any) -> None:
    """Generates the CSV evaluation reports in their respective category folders."""
    all_test_cases: Dict[str, Any] = getattr(config, 'all_test_cases', {})
    if not all_test_cases:
        return

    human_labels: Dict[str, Any] = getattr(config, 'human_labels', {})
    reports_base_dir = os.path.join(PROJECT_ROOT, 'reports')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    try:
        for folder_name, folder_data in all_test_cases.items():
            folder_dir = os.path.join(reports_base_dir, folder_name)
            os.makedirs(folder_dir, exist_ok=True)
            
            for test_name, test_cases_dict in folder_data.items():
                csv_path = os.path.join(folder_dir, f"{test_name}_report_{timestamp}.csv")
                
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["TC No", "Input", "Expected Output", "Actual Output", "Judge Score", "LLM Reason", "Human Score", "Human Reason"])
                    
                    for test_id, test_data in test_cases_dict.items():
                        llm_data = llm_eval_results.get(test_id, {})
                        l_score = llm_data.get("score", "")
                        l_reason = llm_data.get("reason", "")
                        
                        h_data = human_labels.get(test_id)
                        h_score = h_data.get("score", "") if isinstance(h_data, dict) else (h_data if h_data is not None else "")
                        h_reason = h_data.get("reason", "") if isinstance(h_data, dict) else ""
                            
                        writer.writerow([
                            test_id,
                            test_data.get("input", ""),
                            test_data.get("expected_output", ""),
                            test_data.get("actual_output", ""),
                            l_score,
                            l_reason,
                            h_score,
                            h_reason
                        ])
                terminalreporter.write_line(f"\n📝 Evaluation report saved to: {csv_path}", bold=True)
    except Exception as e:
        terminalreporter.write_line(f"\nFailed to save evaluation report: {e}", red=True)
        logger.error(f"Failed to write CSV report: {e}")

def _report_cohens_kappa(config: pytest.Config, terminalreporter: Any) -> None:
    """Calculates and prints the Cohen's Kappa score for judge calibration."""
    human_labels: Dict[str, Any] = getattr(config, 'human_labels', {})
    if not human_labels:
        return

    h_labels = []
    l_labels = []
    
    for test_id, llm_data in llm_eval_results.items():
        if test_id in human_labels:
            h_data = human_labels[test_id]
            h_score = h_data.get("score") if isinstance(h_data, dict) else h_data
            
            h_labels.append(h_score)
            l_labels.append(llm_data.get("score", 0))

    if not h_labels:
        terminalreporter.write_sep("=", "Judge Calibration Results (Skipped)")
        terminalreporter.write_line("No human labels found for the test cases run in this session.")
        return

    p_o, kappa = calculate_cohens_kappa(h_labels, l_labels)

    if kappa < 0:
        interp, color = "Poor (Less than chance agreement)", "red"
    elif kappa <= 0.20:
        interp, color = "Slight agreement", "red"
    elif kappa <= 0.40:
        interp, color = "Fair agreement", "yellow"
    elif kappa <= 0.60:
        interp, color = "Moderate agreement", "yellow"
    elif kappa <= 0.80:
        interp, color = "Substantial agreement (Calibrated)", "green"
    else:
        interp, color = "Almost perfect agreement (Highly Calibrated)", "green"

    terminalreporter.write_sep("=", "Judge Calibration Results")
    terminalreporter.write_line(f"Total Shared Cases Evaluated : {len(h_labels)}")
    terminalreporter.write_line(f"Percentage Agreement         : {p_o * 100:.2f}%")
    
    terminalreporter.write(f"Cohen's Kappa Score          : {kappa:.4f} ", bold=True)
    terminalreporter.write_line(f"[{interp}]", **{color: True, "bold": True})
    
    if kappa >= 0.60:
        terminalreporter.write_line("✅ SUCCESS: Your LLM judge is calibrated.", green=True, bold=True)
    else:
        terminalreporter.write_line("❌ WARNING: Your LLM judge is not well calibrated.", red=True, bold=True)

# --- Core Pytest Hooks ---

def pytest_runtest_call(item: pytest.Item) -> None:
    """Hook to prompt for human evaluation and parse test case configuration."""
    case_data = _extract_case_data(item)
    if not case_data:
        return
        
    test_case_id = _get_test_case_id(case_data)
    if not test_case_id:
        return
        
    _parse_and_store_test_node(item, test_case_id, case_data)
    
    try:
        needs_human_eval = item.config.getini('human_eval')
    except ValueError:
        needs_human_eval = False
    
    human_labels = getattr(item.config, 'human_labels', {})
    if needs_human_eval and test_case_id not in human_labels:
        _prompt_human_evaluator(item, test_case_id, case_data)

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> Generator[None, None, None]:
    """Intercept test reports to extract the LLM judge result (Pass/Fail) for each test_case_id."""
    outcome = yield
    rep = outcome.get_result()

    if rep.when == 'call':
        case_data = _extract_case_data(item)
        if not case_data:
            return
            
        test_case_id = _get_test_case_id(case_data)
        if test_case_id:
            llm_results = getattr(item.config, 'llm_results', {})
            llm_results[test_case_id] = 1 if rep.passed else 0
            item.config.llm_results = llm_results
            
            # Scrape failure reason if missing (useful for unhandled exceptions)
            if test_case_id not in llm_eval_results:
                score = 1 if rep.passed else ""
                reason = "Passed." if rep.passed else "Error."
                if not rep.passed and hasattr(rep, "longreprtext"):
                    lines = [line for line in rep.longreprtext.split('\n') if line.strip()]
                    if lines:
                        reason = lines[-1]
                llm_eval_results[test_case_id] = {
                    "score": score,
                    "reason": reason
                }

def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    """Hook called at the end to generate reports and calculate Cohen's Kappa."""
    _generate_csv_reports(config, terminalreporter)
    _report_cohens_kappa(config, terminalreporter)

def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """
    Called after the entire test run finishes.
    Used to automatically generate and open the dashboard.
    """
    try:
        if not session.config.getini("auto_dashboard"):
            return
    except ValueError:
        pass

    try:
        script_path = os.path.join(PROJECT_ROOT, 'runners', 'build_dashboard.py')
        if os.path.exists(script_path):
            print("\n=== Automatically building and opening dashboard ===")
            subprocess.run([sys.executable, script_path, "--open"], check=False)
    except Exception as e:
        print(f"\n=== Failed to execute dashboard builder: {e} ===")
