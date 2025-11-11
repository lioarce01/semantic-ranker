"""
Basic example: Train a cross-encoder reranker from scratch.
"""

import logging
from semantic_ranker.data import MSMARCODataLoader, DataPreprocessor
from semantic_ranker.training import CrossEncoderTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train a basic cross-encoder reranker."""

    logger.info("=== Basic Cross-Encoder Training Example ===\n")

    # 1. Load and preprocess data
    logger.info("Step 1: Loading data...")
    loader = MSMARCODataLoader()

    # Load a small subset for demo (remove max_samples for full dataset)
    train_data, val_data, test_data = loader.load_and_split(
        max_samples=1000  # Use 1000 samples for quick demo
    )

    logger.info(f"Loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # 2. Preprocess into triples
    logger.info("\nStep 2: Preprocessing data...")
    preprocessor = DataPreprocessor(
        tokenizer_name="distilbert-base-uncased",  # Smaller model for demo
        max_length=256
    )

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

    logger.info(f"Created {len(train_triples)} training triples")

    # 3. Initialize trainer
    logger.info("\nStep 3: Initializing trainer...")
    trainer = CrossEncoderTrainer(
        model_name="distilbert-base-uncased",
        num_labels=1,
        max_length=256,
        loss_function="bce",
        use_lora=False  # Set to True for LoRA fine-tuning
    )

    # 4. Train
    logger.info("\nStep 4: Training...")
    history = trainer.train(
        train_samples=train_triples,
        val_samples=val_triples,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        output_dir="./models/basic_reranker",
        save_best_model=True,
        eval_steps=100,
        logging_steps=50
    )

    logger.info("\n=== Training Complete ===")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        logger.info(f"Best val loss: {min(history['val_loss']):.4f}")

    logger.info("\nModel saved to: ./models/basic_reranker")


if __name__ == "__main__":
    main()
