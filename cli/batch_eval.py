#!/usr/bin/env python3
"""
Batch evaluate a trained model on multiple important datasets.

This script evaluates a single model on all key benchmark datasets
for comprehensive performance assessment and comparison.
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Import shared utilities
from cli.utils import (
    setup_project_path,
    setup_logging,
    find_best_model,
    load_dataset_unified
)

# Setup project imports
setup_project_path()

# Setup logging
logger = setup_logging()

# Now import semantic_ranker modules
from semantic_ranker.evaluation import RankerEvaluator
from semantic_ranker.models import CrossEncoderModel
from semantic_ranker.data import MSMARCODataLoader, CustomDataLoader

# Key datasets for comprehensive evaluation
KEY_DATASETS = [ 
    'nq_eval',                         # Open-ended QA performance
    'msmarco_dev',                     # Comprehensive dataset performance
    'squad_eval',                      # Comprehensive dataset performance
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
        # Temporarily suppress verbose logging
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)

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

        # Restore logging level
        logging.getLogger().setLevel(original_level)

        return results

    except Exception as e:
        logger.error(f"Error evaluating {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_dataset_results(dataset_name: str, results: dict, logger=None):
    """Print concise results for a single dataset."""
    if not results:
        result_line = f"{dataset_name}: FAILED"
    else:
        ndcg10 = results.get('ndcg@10', 0)
        mrr10 = results.get('mrr@10', 0)
        map10 = results.get('map@10', 0)
        result_line = f"{dataset_name}: NDCG@10={ndcg10:.4f} MRR@10={mrr10:.4f} MAP@10={map10:.4f}"

    print(result_line)
    # Also write to file if logger has file handler
    if logger:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream.write(result_line + '\n')
                handler.stream.flush()


def print_summary(all_results: dict, model_name: str, logger=None):
    """Print concise summary."""
    valid_results = [r for r in all_results.values() if r is not None]
    if valid_results:
        avg_ndcg10 = sum(r.get('ndcg@10', 0) for r in valid_results) / len(valid_results)
        avg_mrr10 = sum(r.get('mrr@10', 0) for r in valid_results) / len(valid_results)
        result_line = f"AVERAGE: NDCG@10={avg_ndcg10:.4f} MRR@10={avg_mrr10:.4f}"
    else:
        result_line = "AVERAGE: NO VALID RESULTS"

    print(result_line)
    # Also write to file if logger has file handler
    if logger:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream.write(result_line + '\n')
                handler.stream.flush()


def main():
    parser = argparse.ArgumentParser(description='Batch evaluate model on multiple datasets')
    parser.add_argument('--model-path', required=True,
                       help='Path to the model directory (e.g., models/my_model/best/)')
    parser.add_argument('--datasets', nargs='+',
                       help='Specific datasets to evaluate (default: all key datasets)')
    parser.add_argument('--log-file', type=str,
                       help='Save results to log file (optional)')

    args = parser.parse_args()

    # Validate model path first
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}")
        sys.exit(1)

    # Get model name for display
    model_name = model_path.parent.name if model_path.name == 'best' else model_path.name

    # Setup logging (always generate log file)
    if args.log_file:
        log_filename = args.log_file
    else:  # Auto-generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"batch_eval_{model_name}_{timestamp}.log"

    # Setup logging with clean format for results
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)  # Suppress verbose messages

    # Create file handler for clean output
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)

    # Add file handler (keep existing console handler but modify format)
    logger.addHandler(file_handler)

    # Override the console format to be clean
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setFormatter(logging.Formatter('%(message)s'))

    # Print model header (quantum-style format)
    print(f"MODEL: {model_name}")
    file_handler.stream.write(f"MODEL: {model_name}\n")
    file_handler.stream.flush()

    # Determine which datasets to evaluate
    datasets_to_eval = args.datasets if args.datasets else KEY_DATASETS

    # Evaluate on all datasets
    all_results = {}

    for dataset in datasets_to_eval:
        results = evaluate_model_on_dataset(str(model_path), dataset)
        all_results[dataset] = results
        print_dataset_results(dataset, results, logger)

    print_summary(all_results, model_name, logger)


if __name__ == "__main__":
    main()