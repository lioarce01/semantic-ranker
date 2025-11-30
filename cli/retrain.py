#!/usr/bin/env python3
"""
Retrain the best model with additional data.
Responsibility: Only retraining, finds best model automatically.
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
import random
from semantic_ranker.data import MSMARCODataLoader, CustomDataLoader
from semantic_ranker.training import CrossEncoderTrainer
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
            if best_path.exists() and (best_path / "model.safetensors").exists():
                mtime = best_path.stat().st_mtime
                best_models.append((str(best_path), mtime, model_dir.name))

    if not best_models:
        logger.error("❌ No trained models found. Train a model first.")
        return None

    best_models.sort(key=lambda x: x[1], reverse=True)
    best_path, _, model_name = best_models[0]

    logger.info(f"📍 Found best model: {model_name}")
    return best_path


def main():
    parser = argparse.ArgumentParser(description='Retrain the best model with additional data')
    parser.add_argument('--dataset', choices=['msmarco'] + [f.stem for f in Path('datasets').glob('*.json')],
                       default='msmarco', help='Additional dataset for retraining')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of additional epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for retraining')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                       help='Learning rate for fine-tuning (lower than initial)')
    parser.add_argument('--samples', type=int, default=500,
                       help='Number of additional samples to use')
    parser.add_argument('--suffix', default='_retrained',
                       help='Suffix for retrained model directory')

    args = parser.parse_args()

    logger.info("=== Model Retraining ===")
    logger.info(f"Additional dataset: {args.dataset}")
    logger.info(f"Additional epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    print()

    # 1. Find and load best model
    model_path = find_best_model()
    if not model_path:
        return

    logger.info("Loading existing model...")
    model = CrossEncoderModel.load(model_path)
    logger.info("✅ Model loaded successfully")

    # 2. Load additional data
    logger.info(f"Loading additional data from {args.dataset}...")
    if args.dataset == 'msmarco':
        loader = MSMARCODataLoader()
        additional_data, _, _ = loader.load_and_split(max_samples=args.samples)
    else:
        loader = CustomDataLoader()
        dataset_path = f"datasets/{args.dataset}.json"
        all_data = loader.load_from_json(dataset_path)
        additional_data = all_data[:args.samples]

    logger.info(f"✅ Loaded {len(additional_data)} additional samples")

    # 3. Preprocess additional data
    logger.info("Preparing additional data for retraining...")

    # Convert data to training format (query, document, label)
    # For cross-encoder training, we need individual (query, document, label) pairs
    training_samples = []

    for item in additional_data:
        query = item['query']
        documents = item['documents']
        labels = item['labels']

        # Create individual training samples
        for doc, label in zip(documents, labels):
            training_samples.append((query, doc, float(label)))

    logger.info(f"✅ Created {len(training_samples)} additional training samples")

    # 4. Setup retraining
    logger.info("Setting up retraining...")
    trainer = CrossEncoderTrainer(
        model_name=model.model_name,
        num_labels=model.num_labels,
        max_length=model.max_length,
        loss_function="bce"
    )

    # Use the loaded model
    trainer.model = model

    # 5. Prepare validation data
    logger.info("Preparing validation data...")
    val_samples = None
    if val_data:
        try:
            # Convert validation data to the same format as training
            val_samples = []
            for item in val_data:
                query = item['query']
                documents = item['documents']
                labels = item['labels']

                # Create individual validation samples
                for doc, label in zip(documents, labels):
                    val_samples.append((query, doc, float(label)))

            logger.info(f"✅ Created {len(val_samples)} validation samples")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create validation samples: {e}")
            logger.warning("Training without validation...")
            val_samples = None

    # 6. Retrain
    base_model_dir = Path(model_path).parent.parent
    model_name = Path(model_path).parent.name  # e.g., "basic_reranker"
    output_dir = str(base_model_dir / (model_name + args.suffix))

    logger.info(f"Starting retraining for {args.epochs} epochs...")
    history = trainer.train(
        train_samples=training_samples,
        val_samples=val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=output_dir,
        save_best_model=True,
        eval_steps=50,
        logging_steps=25
    )

    logger.info("\n✅ Retraining completed!")
    logger.info(f"📁 Retrained model saved to: {output_dir}")
    logger.info(".4f")

    logger.info("\n💡 Retrained model available for:")
    logger.info("• Evaluation: python cli/eval.py")
    logger.info("• Testing: python cli/test.py")


if __name__ == "__main__":
    main()
