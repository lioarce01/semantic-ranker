#!/usr/bin/env python3
"""
Train a cross-encoder reranker model.
Responsibility: Only training, no evaluation.
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
from semantic_ranker.data import MSMARCODataLoader, CustomDataLoader, DataPreprocessor
from semantic_ranker.training import CrossEncoderTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train a cross-encoder reranker model')
    parser.add_argument('--dataset', choices=['msmarco'] + [f.stem for f in Path('datasets').glob('*.json')],
                       default='msmarco', help='Dataset to use for training')
    parser.add_argument('--model-name', default='distilbert-base-uncased',
                       help='Pretrained model name')
    parser.add_argument('--output-dir', default='./models/trained_model',
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum samples to load (for quick training)')
    parser.add_argument('--use-lora', action='store_true',
                       help='Use LoRA for efficient training')

    args = parser.parse_args()

    logger.info("=== Cross-Encoder Training ===")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"LoRA: {args.use_lora}")
    print()

    # 1. Load data
    logger.info("Loading data...")
    if args.dataset == 'msmarco':
        loader = MSMARCODataLoader()
        train_data, val_data, test_data = loader.load_and_split(
            max_samples=args.max_samples
        )
        # Use validation data for training validation
        combined_train = train_data + val_data[:len(val_data)//2]
        val_data = val_data[len(val_data)//2:]
    else:
        loader = CustomDataLoader()
        dataset_path = f"datasets/{args.dataset}.json"
        all_data = loader.load_from_json(dataset_path)

        # Split for training
        random.seed(42)
        random.shuffle(all_data)
        split_idx = int(len(all_data) * 0.8)
        combined_train = all_data[:split_idx]
        val_data = all_data[split_idx:]

    logger.info(f"Training samples: {len(combined_train)}")
    logger.info(f"Validation samples: {len(val_data)}")

    # 2. Preprocess
    logger.info("\nPreprocessing data...")
    preprocessor = DataPreprocessor(
        tokenizer_name=args.model_name,
        max_length=256
    )

    train_triples = preprocessor.create_triples(
        combined_train,
        negative_sampling="random",
        num_negatives=1
    )

    val_triples = preprocessor.create_triples(
        val_data,
        negative_sampling="random",
        num_negatives=1
    )

    logger.info(f"Created {len(train_triples)} training triples")

    # 3. Train
    logger.info("\nInitializing trainer...")
    trainer = CrossEncoderTrainer(
        model_name=args.model_name,
        num_labels=1,
        max_length=256,
        loss_function="bce",
        use_lora=args.use_lora
    )

    logger.info("\nStarting training...")
    history = trainer.train(
        train_samples=train_triples,
        val_samples=val_triples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        save_best_model=True,
        eval_steps=100,
        logging_steps=50
    )

    logger.info("\n✅ Training completed!")
    logger.info(f"📁 Model saved to: {args.output_dir}")
    logger.info(".4f")
    if history['val_loss']:
        logger.info(".4f")

    logger.info("\n💡 Next steps:")
    logger.info("• Run evaluation: python cli/eval.py")
    logger.info("• Run testing: python cli/test.py")


if __name__ == "__main__":
    main()
