"""
Example: Train with hard negative mining for improved performance.
"""

import logging
from semantic_ranker.data import MSMARCODataLoader, DataPreprocessor
from semantic_ranker.training import CrossEncoderTrainer, HardNegativeMiner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train with hard negative mining."""

    logger.info("=== Hard Negative Mining Training Example ===\n")

    # 1. Load data
    logger.info("Step 1: Loading data...")
    loader = MSMARCODataLoader()
    train_data, val_data, _ = loader.load_and_split(max_samples=1000)

    # 2. Initialize hard negative miner
    logger.info("\nStep 2: Initializing hard negative miner...")
    miner = HardNegativeMiner(
        bi_encoder_model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Mine hard negatives
    logger.info("\nStep 3: Mining hard negatives...")

    # Convert dataset format
    dataset = [
        {'query': item['query'], 'positive': item.get('positive', item.get('document', ''))}
        for item in train_data
    ]

    train_triples = miner.mine_from_dataset(
        dataset,
        num_negatives=3,  # Mine 3 hard negatives per query
        batch_size=32
    )

    logger.info(f"Created {len(train_triples)} training triples with hard negatives")

    # 4. Train cross-encoder
    logger.info("\nStep 4: Training cross-encoder with hard negatives...")
    trainer = CrossEncoderTrainer(
        model_name="distilbert-base-uncased",
        num_labels=1,
        loss_function="bce"
    )

    # Prepare validation triples
    preprocessor = DataPreprocessor()
    val_triples = preprocessor.create_triples(val_data)

    history = trainer.train(
        train_samples=train_triples,
        val_samples=val_triples,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        output_dir="./models/hard_neg_reranker"
    )

    logger.info("\n=== Training Complete ===")
    logger.info("Model saved to: ./models/hard_neg_reranker")


if __name__ == "__main__":
    main()
