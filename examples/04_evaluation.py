"""
Example: Evaluate a trained reranker model.
"""

import logging
from semantic_ranker.data import MSMARCODataLoader, DataPreprocessor
from semantic_ranker.evaluation import RankerEvaluator
from semantic_ranker.models import CrossEncoderModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_test_data(test_data, num_samples=100):
    """
    Prepare test data in evaluation format.

    Returns list of dicts with:
    - query: str
    - documents: List[str]
    - labels: List[int] (binary relevance)
    """
    eval_data = []

    for item in test_data[:num_samples]:
        query = item.get('query', '')
        positive = item.get('positive', item.get('document', ''))
        negative = item.get('negative', '')

        if not query or not positive:
            continue

        # Create a simple test case with 1 positive and 1 negative
        documents = [positive]
        labels = [1]

        if negative:
            documents.append(negative)
            labels.append(0)

        eval_data.append({
            'query': query,
            'documents': documents,
            'labels': labels
        })

    return eval_data


def main():
    """Evaluate a trained reranker."""

    logger.info("=== Reranker Evaluation Example ===\n")

    # 1. Load test data
    logger.info("Step 1: Loading test data...")
    loader = MSMARCODataLoader()
    _, _, test_data = loader.load_and_split(max_samples=500)

    # Prepare evaluation format
    eval_data = prepare_test_data(test_data, num_samples=100)
    logger.info(f"Prepared {len(eval_data)} evaluation samples")

    # 2. Load trained model
    logger.info("\nStep 2: Loading trained model...")

    # Option A: Load from saved checkpoint
    # evaluator = RankerEvaluator(model_path="./models/basic_reranker/final")

    # Option B: Use a pretrained cross-encoder
    model = CrossEncoderModel(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        num_labels=1
    )
    evaluator = RankerEvaluator(model=model)

    # 3. Evaluate
    logger.info("\nStep 3: Evaluating model...")

    metrics = evaluator.evaluate(
        eval_data,
        metrics=["ndcg@3", "ndcg@5", "ndcg@10", "mrr@10", "map@10", "hit_rate@5"],
        batch_size=32
    )

    logger.info("\n=== Evaluation Results ===")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")

    # 4. Per-query analysis (optional)
    logger.info("\n\nStep 4: Per-query analysis...")

    results = evaluator.evaluate(
        eval_data[:10],  # Analyze first 10 queries
        metrics=["ndcg@5", "mrr@10"],
        return_per_query=True
    )

    logger.info("\nPer-query metrics (first 10):")
    for i, query_metrics in enumerate(results['per_query'], 1):
        logger.info(f"Query {i}: {query_metrics}")

    # 5. Save results
    logger.info("\n\nStep 5: Saving results...")
    evaluator.save_results(metrics, "evaluation_results.json")

    logger.info("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
