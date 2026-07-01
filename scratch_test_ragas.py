import json
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

def test_ragas_dataset():
    dataset_path = "datasets/rag_data.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = SingleTurnSample(
            user_input=item["input"],
            response=item["actual_output"],
            retrieved_contexts=item.get("retrive_context", []),
            reference=item.get("ground_truth", "")
        )
        samples.append(sample)

    eval_dataset = EvaluationDataset(samples=samples)
    print("Dataset prepared successfully with", len(eval_dataset), "samples.")
    print(eval_dataset[0])

if __name__ == "__main__":
    test_ragas_dataset()
