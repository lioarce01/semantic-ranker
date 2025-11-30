#!/usr/bin/env python3
"""
Evaluate a trained model (best model only).
Responsibility: Only evaluation, finds best model automatically.
"""

import sys
import os
from pathlib import Path
import argparse

# Add the parent directory to sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import logging
from semantic_ranker.data import MSMARCODataLoader, CustomDataLoader
from semantic_ranker.evaluation import RankerEvaluator
from semantic_ranker.models import CrossEncoderModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_best_model():
    """Find the best model in the models directory."""
    models_dir = Path("./models")

    if not models_dir.exists():
        logger.error("❌ No models directory found. Train a model first.")
        return None

    best_models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            best_path = model_dir / "best"
            # Check for both regular models and LoRA models
            has_model = (best_path / "model.safetensors").exists() or (best_path / "adapter_model.safetensors").exists()
            if best_path.exists() and has_model:
                # Get modification time to find the most recent
                mtime = best_path.stat().st_mtime
                best_models.append((str(best_path), mtime, model_dir.name))

    if not best_models:
        logger.error("❌ No trained models found. Train a model first.")
        return None

    # Return the most recently trained model
    best_models.sort(key=lambda x: x[1], reverse=True)
    best_path, _, model_name = best_models[0]

    logger.info(f"📍 Found best model: {model_name}")
    return best_path


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
    parser.add_argument('--model-path', help='Path to specific model directory (if not provided, uses best model)')
    parser.add_argument('--dataset', choices=['msmarco'] + [f.stem for f in Path('datasets').glob('*.json')],
                       default='msmarco', help='Dataset to use for evaluation')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples to evaluate')
    parser.add_argument('--metrics', nargs='+',
                       default=['ndcg@5', 'ndcg@10', 'mrr@10', 'map@10'],
                       help='Metrics to compute')

    args = parser.parse_args()

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
    logger.info(f"Preparing evaluation data from {args.dataset}...")
    eval_data = prepare_eval_data(args.dataset, args.samples)
    logger.info(f"Prepared {len(eval_data)} evaluation queries")

    # 4. Evaluate
    logger.info("Running evaluation...")
    metrics = evaluator.evaluate(
        eval_data,
        metrics=args.metrics,
        batch_size=32
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
