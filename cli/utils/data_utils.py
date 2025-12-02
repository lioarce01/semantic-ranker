"""
Data utilities for CLI scripts.

Provides functions for dataset discovery, unified data loading,
and data format conversion.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
import random

logger = logging.getLogger(__name__)


def get_available_datasets() -> List[str]:
    """Get list of available datasets from datasets directory

    Returns:
        List of dataset names (including 'msmarco' and custom datasets)
    """
    datasets_dir = Path("datasets")

    datasets = ['msmarco']  # Always include msmarco

    if datasets_dir.exists():
        for file in datasets_dir.glob("*.json"):
            datasets.append(file.stem)
        for file in datasets_dir.glob("*.jsonl"):
            datasets.append(file.stem)
        for file in datasets_dir.glob("*.csv"):
            datasets.append(file.stem)

    return sorted(set(datasets))


def load_dataset_unified(dataset_name: str, max_samples: int = None,
                        train_split: float = 0.8, val_split: float = 0.1, test_split: float = 0.1):
    """Unified dataset loading wrapper

    Loads data from MS MARCO or custom datasets with a consistent interface.

    Args:
        dataset_name: Name of dataset ('msmarco' or custom)
        max_samples: Maximum samples to load (None for all)
        train_split: Fraction of data for training (default: 0.8)
        val_split: Fraction of data for validation (default: 0.1)
        test_split: Fraction of data for testing (default: 0.1)

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    from semantic_ranker.data import MSMARCODataLoader, CustomDataLoader, DataPreprocessor

    if dataset_name == 'msmarco':
        # Load MS MARCO using dedicated loader
        loader = MSMARCODataLoader()
        return loader.load_and_split(max_samples=max_samples)
    else:
        # Load custom dataset
        loader = CustomDataLoader()
        preprocessor = DataPreprocessor()

        # Try different file formats
        dataset_path = None
        for ext in ['.json', '.jsonl', '.csv']:
            candidate_path = Path(f"datasets/{dataset_name}{ext}")
            if candidate_path.exists():
                dataset_path = str(candidate_path)
                break

        if dataset_path is None:
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found in datasets/ directory")

        # Load data
        if dataset_path.endswith('.json'):
            all_data = loader.load_from_json(dataset_path)
        elif dataset_path.endswith('.jsonl'):
            all_data = loader.load_from_jsonl(dataset_path)
        elif dataset_path.endswith('.csv'):
            all_data = loader.load_from_csv(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

        # Convert to training format (query, document, label pairs)
        training_samples = convert_to_training_samples(all_data)

        # Limit samples if specified
        if max_samples and len(training_samples) > max_samples:
            random.shuffle(training_samples)
            training_samples = training_samples[:max_samples]
            logger.info(f"Limited dataset to {max_samples} samples")

        all_data = training_samples

        # Split data manually
        random.shuffle(all_data)
        n_total = len(all_data)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        n_test = n_total - n_train - n_val

        train_data = all_data[:n_train]
        val_data = all_data[n_train:n_train + n_val]
        test_data = all_data[n_train + n_val:]

        logger.info(f"Split data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        return train_data, val_data, test_data


def convert_to_training_samples(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert dataset format to training samples (query, document, label)

    Args:
        data: List of items with 'query', 'documents', and 'labels' fields

    Returns:
        List of dictionaries with 'query', 'document', 'label' keys
    """
    samples = []

    for item in data:
        query = item['query']
        documents = item['documents']
        labels = item['labels']

        for doc, label in zip(documents, labels):
            samples.append({
                'query': query,
                'document': doc,
                'label': float(label)
            })

    logger.info(f"Converted {len(data)} items to {len(samples)} training samples")
    return samples
