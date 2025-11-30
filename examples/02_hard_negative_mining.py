"""
Example: Train with hard negative mining for improved performance.

This script demonstrates how to use hard negative mining to improve
reranker performance by focusing training on difficult examples.
"""

import logging
import sys
import argparse
from pathlib import Path
from semantic_ranker.data import MSMARCODataLoader, DataPreprocessor
from semantic_ranker.training import CrossEncoderTrainer, HardNegativeMiner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_data_format(data, data_type="training"):
    """Validate data format and content."""
    if not data:
        logger.error(f"No {data_type} data found!")
        return False

    sample = data[0]
    required_keys = ['query']

    # Check if it's the old format (positive/negatives) or new format (documents/labels)
    has_old_format = 'positive' in sample
    has_new_format = 'documents' in sample and 'labels' in sample

    if not (has_old_format or has_new_format):
        logger.error(f"Unsupported data format in {data_type} data")
        logger.error(f"Sample keys: {list(sample.keys())}")
        return False

    logger.info(f"✅ {data_type.capitalize()} data validated: {len(data)} samples")
    logger.info(f"   Format: {'new (documents/labels)' if has_new_format else 'old (positive/negatives)'}")
    return True


def convert_to_miner_format(data):
    """Convert data to format expected by HardNegativeMiner."""
    converted = []

    for item in data:
        query = item.get('query', '').strip()
        if not query:
            continue

        # Handle different data formats
        if 'positive' in item:
            # Old format
            positive = item.get('positive', '').strip()
            if positive:
                converted.append({'query': query, 'positive': positive})
        elif 'documents' in item and 'labels' in item:
            # New format - find first positive document
            documents = item.get('documents', [])
            labels = item.get('labels', [])

            for doc, label in zip(documents, labels):
                if label == 1:  # Positive document
                    converted.append({'query': query, 'positive': doc})
                    break

    logger.info(f"Converted {len(converted)} samples for hard negative mining")
    return converted


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a reranker with hard negative mining for improved performance"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of training samples to use (default: 1000)"
    )

    parser.add_argument(
        "--bi-encoder",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Bi-encoder model for mining hard negatives (default: all-MiniLM-L6-v2)"
    )

    parser.add_argument(
        "--num-negatives",
        type=int,
        default=3,
        help="Number of hard negatives to mine per query (default: 3)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for fine-tuning (default: 2e-5)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/hard_neg_reranker",
        help="Output directory for trained model (default: ./models/hard_neg_reranker)"
    )

    return parser.parse_args()


def main():
    """Train with hard negative mining."""

    args = parse_args()

    logger.info("=" * 60)
    logger.info("HARD NEGATIVE MINING TRAINING EXAMPLE")
    logger.info("=" * 60)
    logger.info("This will improve reranker performance by mining difficult negatives")
    logger.info(f"Training with {args.samples} samples, {args.num_negatives} negatives per query")
    logger.info("=" * 60)

    try:
        # 1. Load and validate data
        logger.info(f"\n📊 Step 1: Loading and validating data ({args.samples} samples)...")
        loader = MSMARCODataLoader()
        train_data, val_data, _ = loader.load_and_split(max_samples=args.samples)

        if not validate_data_format(train_data, "training"):
            sys.exit(1)
        if not validate_data_format(val_data, "validation"):
            sys.exit(1)

        # 2. Initialize hard negative miner
        logger.info("\n🤖 Step 2: Initializing hard negative miner...")

        logger.info(f"   Bi-encoder model: {args.bi_encoder}")
        logger.info(f"   Hard negatives per query: {args.num_negatives}")
        logger.info("   Mining batch size: 32"  # Fixed for stability

        try:
            miner = HardNegativeMiner(bi_encoder_model=args.bi_encoder)
            logger.info("✅ Hard negative miner initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize miner: {e}")
            sys.exit(1)

        # 3. Mine hard negatives
        logger.info(f"\n⛏️ Step 3: Mining hard negatives from {len(dataset)} queries...")

        # Convert to miner format (already done in validation step)
        dataset_for_mining = convert_to_miner_format(train_data)

        if not dataset_for_mining:
            logger.error("❌ No valid data for hard negative mining!")
            sys.exit(1)

        try:
            train_triples = miner.mine_from_dataset(
                dataset_for_mining,
                num_negatives=args.num_negatives,
                batch_size=32  # Fixed batch size for mining
            )

            if not train_triples:
                logger.error("❌ No training triples generated!")
                sys.exit(1)

            logger.info(f"✅ Created {len(train_triples)} training triples with hard negatives")
            logger.info(f"   Average negatives per query: {len(train_triples) / len(dataset):.1f}")

        except Exception as e:
            logger.error(f"❌ Failed to mine hard negatives: {e}")
            sys.exit(1)

        # 4. Train cross-encoder
        logger.info(f"\n🎓 Step 4: Training cross-encoder with {len(train_triples)} hard negative triples...")

        logger.info("Training configuration:")
        logger.info(f"   Model: distilbert-base-uncased")
        logger.info(f"   Epochs: {args.epochs}")
        logger.info(f"   Batch size: {args.batch_size}")
        logger.info(f"   Learning rate: {args.learning_rate}")
        logger.info(f"   Output directory: {args.output_dir}")

        try:
            trainer = CrossEncoderTrainer(
                model_name="distilbert-base-uncased",
                num_labels=1,
                loss_function="bce"
            )
            logger.info("✅ Cross-encoder trainer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize trainer: {e}")
            sys.exit(1)

        # Prepare validation data
        logger.info("\n📊 Preparing validation data...")
        try:
            preprocessor = DataPreprocessor(tokenizer_name=model_name)
            val_triples = preprocessor.create_triples(val_data)

            if not val_triples:
                logger.warning("⚠️ No validation triples created, training without validation")
                val_triples = None
            else:
                logger.info(f"✅ Created {len(val_triples)} validation triples")

        except Exception as e:
            logger.warning(f"⚠️ Failed to create validation triples: {e}")
            logger.warning("Training without validation...")
            val_triples = None

        # Train the model
        try:
            logger.info(f"\n🚀 Starting training for {args.epochs} epochs...")

            history = trainer.train(
                train_samples=train_triples,
                val_samples=val_triples,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                output_dir=args.output_dir,
                save_best_model=True,
                logging_steps=50
            )

            logger.info("\n" + "=" * 60)
            logger.info("🎉 HARD NEGATIVE MINING TRAINING COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"📁 Model saved to: {args.output_dir}")
            logger.info(f"🏆 Best model available at: {args.output_dir}/best")

            # Show final metrics if available
            if history and 'final_metrics' in history:
                metrics = history['final_metrics']
                logger.info("📊 Final training metrics:")
                for key, value in metrics.items():
                    logger.info(f"   {key}: {value:.4f}")

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Success message
        logger.info("\n💡 Next steps:")
        logger.info("• Evaluate: python cli/eval.py")
        logger.info("• Test: python cli/test.py")
        logger.info("• Compare: python scripts/benchmark_comparison.py")


if __name__ == "__main__":
    main()
