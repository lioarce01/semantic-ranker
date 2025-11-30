#!/usr/bin/env python3
"""
Create a completely fresh test set for honest model evaluation.
This ensures no data leakage between training/validation/evaluation and testing.
"""

import sys
import os
from pathlib import Path
import json
import random

# Add the parent directory to sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import logging
from semantic_ranker.data import MSMARCODataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_fresh_test_set(output_file="datasets/msmarco_fresh_test.json", num_samples=100, seed=99999):
    """
    Create a completely fresh test set that was never used in training.

    Args:
        output_file: Where to save the fresh test set
        num_samples: Number of samples to extract
        seed: Random seed for reproducibility (different from training seed=42)
    """

    logger.info("🆕 Creating completely fresh test set...")
    logger.info(f"Output: {output_file}")
    logger.info(f"Samples: {num_samples}")
    logger.info(f"Seed: {seed}")

    # Load MS MARCO with different seed to get completely different split
    loader = MSMARCODataLoader()

    # Get a fresh split that was never seen during training
    _, _, fresh_test_data = loader.load_and_split(
        max_samples=num_samples * 3,  # Get more to ensure we have enough after filtering
        seed=seed  # Completely different seed from training (42)
    )

    # Take the first N samples
    fresh_test_set = fresh_test_data[:num_samples]

    # Validate we have queries
    queries = [item.get('query', '') for item in fresh_test_set]
    queries = [q for q in queries if q]

    logger.info(f"✅ Created fresh test set with {len(queries)} queries")

    # Show some examples
    logger.info("📝 Sample queries from fresh test set:")
    for i, query in enumerate(queries[:5], 1):
        logger.info(f"  {i}. {query}")

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fresh_test_set, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Saved fresh test set to: {output_file}")

    # Show statistics
    total_samples = len(fresh_test_set)
    queries_with_positives = sum(1 for item in fresh_test_set if item.get('positive'))
    avg_query_length = sum(len(item.get('query', '')) for item in fresh_test_set) / total_samples

    logger.info("📊 Fresh test set statistics:")
    logger.info(f"  • Total samples: {total_samples}")
    logger.info(f"  • Samples with positives: {queries_with_positives}")
    logger.info(f"  • Avg query length: {avg_query_length:.1f} chars")

    return fresh_test_set


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Create a fresh test set for honest evaluation')
    parser.add_argument('--output', default='datasets/msmarco_fresh_test.json',
                       help='Output file for fresh test set')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of test samples to create')
    parser.add_argument('--seed', type=int, default=99999,
                       help='Random seed (different from training seed=42)')

    args = parser.parse_args()

    try:
        fresh_test_set = create_fresh_test_set(
            output_file=args.output,
            num_samples=args.samples,
            seed=args.seed
        )

        logger.info("\n🎉 Fresh test set created successfully!")
        logger.info("💡 Now you can use this for completely honest model testing:")
        logger.info(f"   python cli/test.py --queries @ {args.output}")

    except Exception as e:
        logger.error(f"❌ Error creating fresh test set: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
