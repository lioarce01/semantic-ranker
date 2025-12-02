#!/usr/bin/env python3
"""
Evaluate a trained model (best model only).
Responsibility: Only evaluation, finds best model automatically.
"""

import argparse
from pathlib import Path

# Import shared utilities
from cli.utils import (
    setup_project_path,
    setup_logging,
    find_best_model,
    get_available_datasets,
    add_config_args,
    load_config_with_overrides
)

# Setup project imports
setup_project_path()

# Setup logging
logger = setup_logging()

# Now import semantic_ranker modules
from semantic_ranker.data import MSMARCODataLoader, CustomDataLoader
from semantic_ranker.evaluation import RankerEvaluator


def prepare_eval_data(dataset_name, num_samples=100):
    """Prepare evaluation data."""
    if dataset_name == 'msmarco':
        loader = MSMARCODataLoader()
        _, _, test_data = loader.load_and_split()
        eval_data = test_data[:num_samples]
    else:
        loader = CustomDataLoader()
        dataset_path = f"datasets/{dataset_name}.json"
        all_data = loader.load_from_json(dataset_path)
        eval_data = all_data[:num_samples]

    # Convert to evaluation format
    formatted_data = []
    for item in eval_data:
        query = item.get('query', '')

        # Handle both old format (positive/negatives) and new format (documents/labels)
        if 'documents' in item and 'labels' in item:
            # New format - already correct
            documents = item['documents']
            labels = item['labels']
        elif 'positive' in item:
            # Old format - convert
            positive = item.get('positive', '')
            negatives = item.get('negatives', [])

            if not positive:
                continue

            documents = [positive] + negatives[:3]  # Up to 3 negatives
            labels = [1] + [0] * len(documents[1:])
        else:
            # Unknown format
            continue

        if not query or not documents:
            continue

        formatted_data.append({
            'query': query,
            'documents': documents,
            'labels': labels
        })

    return formatted_data


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model')

    # Add config support
    parser = add_config_args(parser)

    # Add evaluation-specific arguments
    parser.add_argument('--model-path', help='Path to specific model directory (if not provided, uses best model)')
    parser.add_argument('--dataset', choices=get_available_datasets(),
                       default='msmarco', help='Dataset to use for evaluation')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples to evaluate')
    parser.add_argument('--metrics', nargs='+',
                       default=['ndcg@5', 'ndcg@10', 'mrr@10', 'map@10'],
                       help='Metrics to compute')
    parser.add_argument('--eval-batch-size', type=int, help='Override evaluation batch size')
    parser.add_argument('--num-samples', type=int, help='Override number of samples')

    args = parser.parse_args()

    # Load configuration with CLI overrides
    config = load_config_with_overrides(args)

    logger.info("=== Model Evaluation ===")
    print()

    # 1. Get model path
    if args.model_path:
        model_path = args.model_path
        if not Path(model_path).exists():
            logger.error(f"❌ Model path does not exist: {model_path}")
            return
        logger.info(f"📍 Using specified model: {model_path}")
    else:
        model_path = find_best_model()
        if not model_path:
            return

    # 2. Load model
    logger.info("Loading model...")
    evaluator = RankerEvaluator(model_path=model_path)

    # 3. Prepare evaluation data
    num_samples = args.num_samples or args.samples
    logger.info(f"Preparing evaluation data from {args.dataset}...")
    eval_data = prepare_eval_data(args.dataset, num_samples)
    logger.info(f"Prepared {len(eval_data)} evaluation queries")

    # 4. Evaluate (using optimized batch evaluation)
    logger.info("Running evaluation...")
    batch_size = args.eval_batch_size or config.evaluation.batch_size
    metrics = evaluator.evaluate(
        eval_data,
        metrics=args.metrics,
        batch_size=batch_size,
        query_batch_size=8  # Use optimized batch processing
    )

    # 5. Results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Performance interpretation
    ndcg_at_10 = metrics.get('ndcg@10', 0)
    if ndcg_at_10 > 0.85:
        print("\n🎉 EXCELLENT PERFORMANCE!")
    elif ndcg_at_10 > 0.70:
        print("\n👍 GOOD PERFORMANCE!")
    elif ndcg_at_10 > 0.50:
        print("\n⚠️ DECENT PERFORMANCE - Consider more training")
    else:
        print("\n❌ POOR PERFORMANCE - Model needs improvement")

    print("\n📊 NDCG@10 Score: Higher is better (max = 1.0)")


if __name__ == "__main__":
    main()
