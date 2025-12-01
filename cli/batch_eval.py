#!/usr/bin/env python3
"""
Batch evaluate a trained model on multiple important datasets.

This script evaluates a single model on all key benchmark datasets
for comprehensive performance assessment and comparison.
"""

import sys
from pathlib import Path
import argparse

# Add the parent directory to sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from semantic_ranker.data import MSMARCODataLoader, CustomDataLoader
from semantic_ranker.evaluation import RankerEvaluator
from semantic_ranker.models import CrossEncoderModel

# Key datasets for comprehensive evaluation
KEY_DATASETS = [
    'msmarco_dev_benchmark_with_hard_negatives',  # Hard negatives performance
    'msmarco_dev_benchmark',                      # Standard MS MARCO benchmark
    'qa_mixed_giant',                            # Training data performance
    'natural_questions',                         # Open-ended QA performance
    'msmarco_nq_mixed'                           # Mixed domain performance
]


def evaluate_model_on_dataset(model_path: str, dataset_name: str):
    """
    Evaluate a model on a specific dataset.

    Args:
        model_path: Path to the model directory
        dataset_name: Name of the dataset to evaluate on

    Returns:
        dict: Evaluation results or None if failed
    """
    try:
        # Load model
        model = CrossEncoderModel.load(model_path)

        # Load dataset
        if dataset_name == 'msmarco':
            loader = MSMARCODataLoader()
            _, _, test_data = loader.load_and_split()
            eval_data = test_data[:100]  # Use first 100 for quick evaluation
        else:
            loader = CustomDataLoader()
            dataset_path = f"datasets/{dataset_name}.json"
            if not Path(dataset_path).exists():
                return None

            all_data = loader.load_from_json(dataset_path)
            eval_data = all_data[:100]  # Use first 100 samples

        if not eval_data:
            return None

        # Create evaluator
        evaluator = RankerEvaluator(model=model)

        # Evaluate
        results = evaluator.evaluate(eval_data)
        return results

    except Exception as e:
        return None


def print_dataset_results(dataset_name: str, results: dict):
    """Print concise results for a single dataset."""
    if not results:
        print(f"{dataset_name}: FAILED")
        return

    ndcg10 = results.get('ndcg@10', 0)
    mrr10 = results.get('mrr@10', 0)
    map10 = results.get('map@10', 0)

    print(f"{dataset_name}: NDCG@10={ndcg10:.4f} MRR@10={mrr10:.4f} MAP@10={map10:.4f}")


def print_summary(all_results: dict, model_name: str):
    """Print concise summary."""
    valid_results = [r for r in all_results.values() if r is not None]
    if valid_results:
        avg_ndcg10 = sum(r.get('ndcg@10', 0) for r in valid_results) / len(valid_results)
        avg_mrr10 = sum(r.get('mrr@10', 0) for r in valid_results) / len(valid_results)
        print(f"AVERAGE: NDCG@10={avg_ndcg10:.4f} MRR@10={avg_mrr10:.4f}")
    else:
        print("AVERAGE: NO VALID RESULTS")


def main():
    parser = argparse.ArgumentParser(description='Batch evaluate model on multiple datasets')
    parser.add_argument('--model-path', required=True,
                       help='Path to the model directory (e.g., models/my_model/best/)')
    parser.add_argument('--datasets', nargs='+',
                       help='Specific datasets to evaluate (default: all key datasets)')

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}")
        sys.exit(1)

    # Get model name for display
    model_name = model_path.parent.name if model_path.name == 'best' else model_path.name
    print(f"MODEL: {model_name}")

    # Determine which datasets to evaluate
    datasets_to_eval = args.datasets if args.datasets else KEY_DATASETS

    # Evaluate on all datasets
    all_results = {}

    for dataset in datasets_to_eval:
        results = evaluate_model_on_dataset(str(model_path), dataset)
        all_results[dataset] = results
        print_dataset_results(dataset, results)

    print_summary(all_results, model_name)


if __name__ == "__main__":
    main()