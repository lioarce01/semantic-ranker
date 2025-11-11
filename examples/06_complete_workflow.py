"""
Complete workflow: Data loading -> Training -> Evaluation -> Optimization -> Deployment
"""

import logging
from pathlib import Path

from semantic_ranker.data import MSMARCODataLoader, DataPreprocessor
from semantic_ranker.training import CrossEncoderTrainer, HardNegativeMiner
from semantic_ranker.evaluation import RankerEvaluator
from semantic_ranker.rag import RAGPipeline
from semantic_ranker.optimization import ModelOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Complete end-to-end workflow for building a semantic reranker.
    """

    logger.info("="*60)
    logger.info("COMPLETE SEMANTIC RERANKER WORKFLOW")
    logger.info("="*60)

    # ========== PHASE 1: DATA PREPARATION ==========
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: DATA PREPARATION")
    logger.info("="*60)

    # Load data
    logger.info("\n1.1 Loading MS MARCO dataset...")
    loader = MSMARCODataLoader()
    train_data, val_data, test_data = loader.load_and_split(
        max_samples=2000,  # Use subset for demo
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    logger.info(f"Dataset loaded:")
    logger.info(f"  Train: {len(train_data)} samples")
    logger.info(f"  Val:   {len(val_data)} samples")
    logger.info(f"  Test:  {len(test_data)} samples")

    # Preprocess
    logger.info("\n1.2 Preprocessing data...")
    preprocessor = DataPreprocessor(
        tokenizer_name="distilbert-base-uncased",
        max_length=256,
        normalize_text=True
    )

    # Create basic triples
    train_triples = preprocessor.create_triples(
        train_data,
        negative_sampling="random",
        num_negatives=1
    )

    val_triples = preprocessor.create_triples(
        val_data,
        negative_sampling="random",
        num_negatives=1
    )

    # Balance dataset
    train_triples = preprocessor.balance_dataset(train_triples, balance_ratio=1.0)

    logger.info(f"Triples created: {len(train_triples)} train, {len(val_triples)} val")

    # Get statistics
    stats = preprocessor.get_statistics(train_triples)
    logger.info(f"\nDataset statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # ========== PHASE 2: TRAINING ==========
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: MODEL TRAINING")
    logger.info("="*60)

    logger.info("\n2.1 Training cross-encoder...")
    trainer = CrossEncoderTrainer(
        model_name="distilbert-base-uncased",
        num_labels=1,
        max_length=256,
        loss_function="bce",
        use_lora=False
    )

    history = trainer.train(
        train_samples=train_triples,
        val_samples=val_triples,
        epochs=2,  # Increase for production
        batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        output_dir="./models/workflow_reranker",
        save_best_model=True,
        eval_steps=100,
        logging_steps=50
    )

    logger.info(f"\nTraining complete!")
    logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        logger.info(f"  Best val loss: {min(history['val_loss']):.4f}")

    # ========== PHASE 3: EVALUATION ==========
    logger.info("\n" + "="*60)
    logger.info("PHASE 3: MODEL EVALUATION")
    logger.info("="*60)

    # Prepare test data
    logger.info("\n3.1 Preparing test data...")
    eval_data = []
    for item in test_data[:100]:
        query = item.get('query', '')
        positive = item.get('positive', item.get('document', ''))
        negative = item.get('negative', '')

        if query and positive:
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

    logger.info(f"Prepared {len(eval_data)} test samples")

    # Evaluate
    logger.info("\n3.2 Evaluating model...")
    evaluator = RankerEvaluator(
        model_path="./models/workflow_reranker/best"
    )

    metrics = evaluator.evaluate(
        eval_data,
        metrics=["ndcg@5", "ndcg@10", "mrr@10", "map@10", "hit_rate@5"],
        batch_size=32
    )

    logger.info("\nEvaluation results:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")

    # ========== PHASE 4: OPTIMIZATION ==========
    logger.info("\n" + "="*60)
    logger.info("PHASE 4: MODEL OPTIMIZATION")
    logger.info("="*60)

    logger.info("\n4.1 Optimizing model for production...")
    try:
        optimizer = ModelOptimizer(
            model_path="./models/workflow_reranker/best"
        )

        optimized_models = optimizer.optimize_for_inference(
            output_path="./models/workflow_optimized",
            enable_onnx=True,
            enable_quantization=True,
            target_device="cpu"
        )

        logger.info("\nOptimized models:")
        for name, path in optimized_models.items():
            logger.info(f"  {name}: {path}")

        # Compare sizes
        comparison = optimizer.compare_models(
            original_path="./models/workflow_reranker/best",
            optimized_paths=optimized_models
        )

    except Exception as e:
        logger.warning(f"Optimization failed: {e}")

    # ========== PHASE 5: RAG DEPLOYMENT ==========
    logger.info("\n" + "="*60)
    logger.info("PHASE 5: RAG PIPELINE DEPLOYMENT")
    logger.info("="*60)

    # Create sample corpus
    logger.info("\n5.1 Setting up RAG pipeline...")
    documents = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing focuses on human-computer language interaction.",
        "Computer vision enables machines to interpret visual information.",
        "Reinforcement learning involves agents learning through trial and error.",
    ]

    # Initialize pipeline with reranker
    pipeline = RAGPipeline(
        retriever_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="./models/workflow_reranker/best",
        top_k_retrieval=5,
        top_k_rerank=3
    )

    # Index documents
    pipeline.index_documents(documents)

    # Test queries
    logger.info("\n5.2 Testing RAG pipeline...")
    test_queries = [
        "What is machine learning?",
        "Explain deep learning"
    ]

    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results = pipeline.retrieve_and_rerank(
            query,
            return_scores=True
        )

        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. {result['document'][:60]}... (score: {result['score']:.4f})")

    # Benchmark
    logger.info("\n5.3 Benchmarking pipeline...")
    benchmark_stats = pipeline.benchmark(test_queries * 5, num_runs=3)

    logger.info("\nPipeline performance:")
    for key, value in benchmark_stats.items():
        logger.info(f"  {key}: {value:.2f}")

    # ========== SUMMARY ==========
    logger.info("\n" + "="*60)
    logger.info("WORKFLOW COMPLETE - SUMMARY")
    logger.info("="*60)

    logger.info(f"""
    ✓ Data: {len(train_data)} training samples processed
    ✓ Model: Cross-encoder trained and saved
    ✓ Evaluation: {metrics.get('ndcg@10', 0):.4f} NDCG@10
    ✓ Optimization: Models optimized for production
    ✓ Deployment: RAG pipeline ready

    Models saved to:
      - Training: ./models/workflow_reranker/
      - Optimized: ./models/workflow_optimized/

    Next steps:
      1. Scale up with full dataset
      2. Deploy optimized model to production
      3. Monitor performance and retrain as needed
    """)


if __name__ == "__main__":
    main()
