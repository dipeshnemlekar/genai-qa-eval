import os
import sys
import json
import logging
from typing import List, Tuple

# Add the project root to sys.path so we can import from utils
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from utils.helpers import load_dataset, get_gemini_judge, calculate_cohens_kappa

# Set library loggers to WARNING to reduce verbosity in CLI
logging.getLogger('deepeval').setLevel(logging.WARNING)
logging.getLogger('utils.helpers').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

def main():
    print("=== Human-in-the-Loop Judge Calibration ===")
    print("This script will present you with test cases.")
    print("You will act as the human judge (1 = Pass, 0 = Fail).")
    print("The LLM-as-a-judge will also score the cases.")
    print("We will then compute Cohen's Kappa to measure alignment.\n")

    # Load dataset (we'll use 'golden' as the default category for Correctness evaluation)
    data = load_dataset("testdata.json", "golden")
    
    # We want up to 20 samples
    samples = data[:20] if len(data) > 20 else data
    
    if not samples:
        print("No test data found in 'golden' category.")
        return

    gemini_judge = get_gemini_judge()
    correctness_metric = GEval(
        name='Correctness',
        criteria='Determine whether the actual output is factually correct based on the expected output. '
                 'The actual output should be considered correct if it conveys the same meaning and facts '
                 'as the expected output, even if worded differently. Penalize omissions or hallucinations.',
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=gemini_judge
    )

    human_labels = []
    llm_labels = []
    
    # Try loading existing labels to save time
    labels_path = os.path.join(project_root, 'datasets', 'human_labels.json')
    saved_labels = {}
    if os.path.exists(labels_path):
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                saved_labels = json.load(f)
        except Exception:
            pass

    for i, item in enumerate(samples):
        print(f"\n--- Test Case {i+1}/{len(samples)} (ID: {item.get('id', 'N/A')}) ---")
        print(f"Input: {item.get('input')}")
        print(f"Expected Output: {item.get('expected_output')}")
        print(f"Actual Output: {item.get('actual_output')}")
        
        # 1. Get human label
        human_label = None
        
        # Check if we have it saved
        if item.get('id') in saved_labels:
            use_saved = input(f"Use saved human label ({saved_labels[item['id']]}) for this case? (y/n/skip) [y]: ").strip().lower()
            if use_saved == 'y' or use_saved == '':
                human_label = saved_labels[item['id']]
            elif use_saved == 'skip' or use_saved == 's':
                continue
                
        if human_label is None:
            while True:
                user_input = input("Human Score (1 for Pass, 0 for Fail, 's' to skip): ").strip().lower()
                if user_input == 's':
                    break
                if user_input in ['0', '1']:
                    human_label = int(user_input)
                    break
                print("Invalid input. Please enter 1, 0, or 's'.")
            
        if human_label is None:
            continue # Skipped
            
        # 2. Get LLM label
        print("Evaluating with LLM Judge (GEval Correctness)...")
        test_case = LLMTestCase(
            input=item.get("input", ""),
            expected_output=item.get("expected_output", ""),
            actual_output=item.get("actual_output", ""),
            context=item.get("retrive_context")
        )
        
        try:
            correctness_metric.measure(test_case)
            llm_score = correctness_metric.score
            # Pass/Fail based on threshold
            llm_label = 1 if llm_score >= correctness_metric.threshold else 0
            print(f"LLM Judge Score: {llm_score} -> {'Pass' if llm_label else 'Fail'}")
            print(f"Reason: {correctness_metric.reason}")
        except Exception as e:
            print(f"Error running LLM judge: {e}")
            continue
            
        # Record labels
        human_labels.append(human_label)
        llm_labels.append(llm_label)
        
        # Save human label
        if item.get('id'):
            saved_labels[item['id']] = human_label

    # Save all labels
    try:
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(saved_labels, f, indent=2)
    except Exception as e:
        print(f"Could not save labels: {e}")

    # Compute metrics
    print("\n" + "="*40)
    print("Calibration Results")
    print("="*40)
    
    if not human_labels:
        print("No cases were evaluated.")
        return
        
    p_o, kappa = calculate_cohens_kappa(human_labels, llm_labels)
    
    print(f"Total Cases Evaluated: {len(human_labels)}")
    print(f"Percentage Agreement: {p_o * 100:.2f}%")
    print(f"Cohen's Kappa Score: {kappa:.4f}")
    
    # Interpretation
    if kappa < 0:
        interp = "Poor (Less than chance agreement)"
    elif kappa <= 0.20:
        interp = "Slight agreement"
    elif kappa <= 0.40:
        interp = "Fair agreement"
    elif kappa <= 0.60:
        interp = "Moderate agreement"
    elif kappa <= 0.80:
        interp = "Substantial agreement (Calibrated)"
    else:
        interp = "Almost perfect agreement (Highly Calibrated)"
        
    print(f"Interpretation: {interp}")
    print("="*40)
    
    if kappa >= 0.60:
        print("✅ SUCCESS: Your LLM judge is calibrated.")
    else:
        print("❌ WARNING: Your LLM judge is not well calibrated. Consider tuning your evaluation criteria, metrics, or prompts.")

if __name__ == "__main__":
    main()
