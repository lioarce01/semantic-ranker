#!/usr/bin/env python3
"""
Evaluate trained model on BEIR benchmark datasets (simple version without BM25).
BEIR provides diverse out-of-domain datasets for robust evaluation.
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
)

# Setup project imports
setup_project_path()

# Setup logging (will be configured with file handler in main)
logger = logging.getLogger(__name__)

# Now import semantic_ranker modules
from semantic_ranker.models import CrossEncoderModel


def evaluate_on_beir(model_path: str, dataset_name: str, split: str = "test",
                     max_docs_per_query: int = 1000, score_batch_size: int = 32):
    """
    Evaluate model on BEIR dataset (simple version with batched scoring).

    Args:
        model_path: Path to trained model
        dataset_name: BEIR dataset name (e.g., 'nfcorpus', 'scifact', 'fiqa')
        split: Dataset split to use (default: 'test')
        max_docs_per_query: Max documents to score per query (default: 1000)
        score_batch_size: Batch size for cross-encoder scoring (default: 32)
    """
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval
        from tqdm import tqdm
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Run: pip install beir")
        sys.exit(1)

    logger.info(f"📥 Downloading BEIR dataset: {dataset_name}")

    # Download dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, "datasets/beir")

    # Load dataset
    logger.info(f"📂 Loading dataset from: {data_path}")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    logger.info(f"✅ Dataset loaded:")
    logger.info(f"   Queries: {len(queries)}")
    logger.info(f"   Corpus: {len(corpus)}")
    logger.info(f"   Relevance judgments: {len(qrels)}")

    # Load model
    logger.info(f"🤖 Loading model from: {model_path}")
    model = CrossEncoderModel.load(model_path)

    # Score documents with cross-encoder (batched for efficiency)
    corpus_items = list(corpus.items())

    # Decide whether to score all docs or limit
    if len(corpus_items) <= max_docs_per_query:
        logger.info(f"🔄 Scoring ALL {len(corpus_items)} documents per query (batch_size={score_batch_size})")
        total_predictions = len(queries) * len(corpus_items)
    else:
        logger.info(f"⚠️ Large corpus ({len(corpus_items)} docs), limiting to {max_docs_per_query} per query")
        total_predictions = len(queries) * max_docs_per_query

    logger.info(f"📊 Total predictions: {total_predictions:,} (~{total_predictions // score_batch_size:,} batches)")

    results = {}

    for query_id in tqdm(list(queries.keys()), desc="Scoring queries"):
        if query_id not in qrels:
            continue

        query_text = queries[query_id]

        # Get documents to score (all for small datasets, first N for large)
        docs_to_score = corpus_items if len(corpus_items) <= max_docs_per_query else corpus_items[:max_docs_per_query]

        # Prepare batches for scoring
        doc_ids = [doc_id for doc_id, _ in docs_to_score]
        doc_texts = [doc_data.get('text', '') for _, doc_data in docs_to_score]
        query_texts = [query_text] * len(doc_ids)

        # Score all documents in batches
        scores = model.predict(
            query_texts,
            doc_texts,
            batch_size=score_batch_size
        )

        # Create scored results
        doc_scores = {doc_id: float(score) for doc_id, score in zip(doc_ids, scores)}

        # Sort by score
        results[query_id] = dict(sorted(doc_scores.items(),
                                       key=lambda x: x[1],
                                       reverse=True))

    # Evaluate with BEIR metrics
    logger.info("📊 Computing BEIR metrics...")

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results,
        k_values=[1, 3, 5, 10, 20, 100]
    )

    # Print results (to both console and log)
    def log_and_print(msg):
        print(msg)
        logger.info(msg)

    log_and_print("\n" + "="*60)
    log_and_print(f"BEIR EVALUATION RESULTS: {dataset_name.upper()}")
    log_and_print("="*60)

    log_and_print("\n📈 NDCG@k:")
    for k, v in ndcg.items():
        log_and_print(f"  {k}: {v:.4f}")

    log_and_print("\n🎯 MAP@k:")
    for k, v in _map.items():
        log_and_print(f"  {k}: {v:.4f}")

    log_and_print("\n🔍 Recall@k:")
    for k, v in recall.items():
        log_and_print(f"  {k}: {v:.4f}")

    log_and_print("\n✓ Precision@k:")
    for k, v in precision.items():
        log_and_print(f"  {k}: {v:.4f}")

    # Interpretation
    ndcg_at_10 = ndcg.get('NDCG@10', 0)
    log_and_print("\n" + "="*60)
    if ndcg_at_10 > 0.50:
        log_and_print("🎉 EXCELLENT - Strong zero-shot performance!")
    elif ndcg_at_10 > 0.35:
        log_and_print("👍 GOOD - Decent zero-shot generalization")
    elif ndcg_at_10 > 0.20:
        log_and_print("⚠️ FAIR - Some domain adaptation may help")
    else:
        log_and_print("❌ POOR - Significant domain gap")
    log_and_print("="*60)

    return ndcg, _map, recall, precision


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained model on BEIR benchmark (simple version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available BEIR datasets:
  Small datasets (good for quick testing):
    - nfcorpus      : Nutrition/medical (3.6k docs)
    - scifact       : Scientific fact-checking (5.2k docs)
    - arguana       : Argument retrieval (8.7k docs)

  Medium datasets:
    - fiqa          : Financial QA (57k docs)
    - trec-covid    : COVID-19 research (171k docs)
    - dbpedia-entity: Entity retrieval (4.6M docs)

  Large datasets:
    - msmarco       : Passage ranking (8.8M docs) - your training data!
    - hotpotqa      : Multi-hop QA (5.2M docs)
        """
    )

    parser.add_argument('--model-path', required=True,
                       help='Path to trained model directory')
    parser.add_argument('--dataset', required=True,
                       choices=['nfcorpus', 'scifact', 'arguana', 'fiqa',
                               'trec-covid', 'hotpotqa', 'dbpedia-entity'],
                       help='BEIR dataset name')
    parser.add_argument('--split', default='test',
                       choices=['train', 'dev', 'test'],
                       help='Dataset split to use (default: test)')
    parser.add_argument('--max-docs', type=int, default=1000,
                       help='Max documents to score per query (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for cross-encoder scoring (default: 32)')

    args = parser.parse_args()

    # Setup logging to file
    model_name = Path(args.model_path).parent.name if Path(args.model_path).name == 'best' else Path(args.model_path).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"beir_eval_{args.dataset}_{model_name}_{timestamp}.log"

    # Configure logger with both console and file handlers
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("="*60)
    logger.info(f"BEIR EVALUATION LOG")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)

    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"❌ Model path does not exist: {model_path}")
        sys.exit(1)

    try:
        evaluate_on_beir(
            str(model_path),
            args.dataset,
            args.split,
            max_docs_per_query=args.max_docs,
            score_batch_size=args.batch_size
        )
        logger.info(f"\n✅ Evaluation completed successfully!")
        logger.info(f"📝 Results saved to: {log_filename}")
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
