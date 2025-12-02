#!/usr/bin/env python3
"""
Train a cross-encoder reranker model.
Responsibility: Only training, no evaluation.
"""

import argparse
import random
from pathlib import Path

# Import shared utilities
from cli.utils import (
    setup_project_path,
    setup_logging,
    get_available_datasets,
    load_dataset_unified,
    add_config_args,
    load_config_with_overrides,
    save_config_with_model
)

# Setup project imports
setup_project_path()

# Setup logging
logger = setup_logging()

# Now import semantic_ranker modules
from semantic_ranker.data import DataPreprocessor
from semantic_ranker.training import CrossEncoderTrainer


def main():
    parser = argparse.ArgumentParser(description='Train a cross-encoder reranker model')

    # Add config support
    parser = add_config_args(parser)

    # Add training-specific arguments (can override config)
    parser.add_argument('--dataset', choices=get_available_datasets(),
                       help='Dataset to use for training')
    parser.add_argument('--model-name', help='Pretrained model name')
    parser.add_argument('--output-dir', default='./models/trained_model',
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--max-samples', type=int, help='Maximum samples to load')
    parser.add_argument('--use-lora', action='store_true', help='Use LoRA for efficient training')

    args = parser.parse_args()

    # Load configuration with CLI overrides
    config = load_config_with_overrides(args)

    # Use config values
    dataset = args.dataset or config.data.dataset
    model_name = args.model_name or config.model.model_name
    epochs = args.epochs or config.training.epochs
    batch_size = args.batch_size or config.training.batch_size
    learning_rate = args.learning_rate or config.training.learning_rate
    max_samples = args.max_samples or config.data.max_samples
    use_lora = args.use_lora or config.model.use_lora

    logger.info("=== Cross-Encoder Training ===")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"LoRA: {use_lora}")
    print()

    # 1. Load data using shared utility
    logger.info("Loading data...")
    train_data, val_data, test_data = load_dataset_unified(dataset, max_samples)

    # Use validation data for training validation
    combined_train = train_data + val_data[:len(val_data)//2]
    val_data = val_data[len(val_data)//2:]

    logger.info(f"Training samples: {len(combined_train)}")
    logger.info(f"Validation samples: {len(val_data)}")

    # 2. Preprocess
    logger.info("\nPreprocessing data...")
    preprocessor = DataPreprocessor(
        tokenizer_name=model_name,
        max_length=config.model.max_length
    )

    train_triples = preprocessor.create_triples(
        combined_train,
        negative_sampling=config.data.negative_sampling,
        num_negatives=config.data.num_negatives
    )

    val_triples = preprocessor.create_triples(
        val_data,
        negative_sampling=config.data.negative_sampling,
        num_negatives=config.data.num_negatives
    )

    logger.info(f"Created {len(train_triples)} training triples")

    # 3. Train
    logger.info("\nInitializing trainer...")
    trainer = CrossEncoderTrainer(
        model_name=model_name,
        num_labels=1,
        max_length=config.model.max_length,
        loss_function=config.training.loss_function,
        use_lora=use_lora
    )

    logger.info("\nStarting training...")
    history = trainer.train(
        train_samples=train_triples,
        val_samples=val_triples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=args.output_dir,
        save_best_model=True,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps
    )

    # Save configuration with model for reproducibility
    save_config_with_model(config, args.output_dir)

    logger.info("\n✅ Training completed!")
    logger.info(f"📁 Model saved to: {args.output_dir}")
    logger.info(f"⚙️ Configuration saved alongside model")

    if history.get('train_loss'):
        logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history.get('val_loss'):
        logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")

    logger.info("\n💡 Next steps:")
    logger.info("• Run evaluation: python cli/eval.py")
    logger.info("• Run testing: python cli/test.py")


if __name__ == "__main__":
    main()
