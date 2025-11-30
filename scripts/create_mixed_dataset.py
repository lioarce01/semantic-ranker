#!/usr/bin/env python3
"""
Create mixed datasets for reranking training.

Combines multiple datasets with configurable proportions.
Supports MS MARCO, Natural Questions, and custom JSON datasets.
"""

import json
import random
from pathlib import Path
import argparse
from typing import List, Dict, Any

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

def load_ms_marco_samples(max_samples: int = 1000, seed: int = 42) -> List[Dict[str, Any]]:
    """Load samples from MS MARCO dataset."""
    if not HAS_DATASETS:
        print("⚠️ datasets library not available, skipping MS MARCO")
        return []

    try:
        from semantic_ranker.data import MSMARCODataLoader

        random.seed(seed)
        loader = MSMARCODataLoader()

        # Load training data (we'll take a subset)
        train_data, _, _ = loader.load_and_split(max_samples=max_samples)

        # Convert to our format
        samples = []
        for item in train_data[:max_samples]:
            query = item['query']
            documents = item['documents']
            labels = item['labels']

            # Convert to our training format
            training_samples = []
            for doc, label in zip(documents, labels):
                training_samples.append({
                    "query": query,
                    "document": doc,
                    "label": float(label)
                })

            samples.extend(training_samples)

        print(f"✅ Loaded {len(samples)} samples from MS MARCO")
        return samples

    except Exception as e:
        print(f"⚠️ Failed to load MS MARCO: {e}")
        return []

def load_json_dataset(file_path: str, max_samples: int = 1000) -> List[Dict[str, Any]]:
    """Load samples from a JSON dataset file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert from our format to training format
        samples = []
        for item in data[:max_samples]:
            query = item['query']
            documents = item['documents']
            labels = item['labels']

            for doc, label in zip(documents, labels):
                samples.append({
                    "query": query,
                    "document": doc,
                    "label": float(label)
                })

        print(f"✅ Loaded {len(samples)} samples from {file_path}")
        return samples

    except Exception as e:
        print(f"⚠️ Failed to load {file_path}: {e}")
        return []

def create_mixed_dataset(
    datasets_config: Dict[str, Any],
    output_path: str = "datasets/mixed_dataset.json",
    total_samples: int = 5000,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42
):
    """
    Create a mixed dataset from multiple sources.

    Args:
        datasets_config: Dict with dataset configurations
        output_path: Where to save the mixed dataset
        total_samples: Total number of samples to create
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed
    """
    random.seed(seed)
    all_samples = []

    print("🔄 Loading datasets...")

    for dataset_name, config in datasets_config.items():
        print(f"📊 Processing {dataset_name}...")

        if config['type'] == 'msmarco':
            samples = load_ms_marco_samples(
                max_samples=config['samples'],
                seed=seed
            )
        elif config['type'] == 'json':
            samples = load_json_dataset(
                config['path'],
                max_samples=config['samples']
            )
        else:
            print(f"⚠️ Unknown dataset type: {config['type']}")
            continue

        # Add dataset identifier
        for sample in samples:
            sample['dataset'] = dataset_name

        all_samples.extend(samples)

    print(f"📊 Total samples collected: {len(all_samples)}")

    # Shuffle and select the requested number of samples
    random.shuffle(all_samples)
    selected_samples = all_samples[:total_samples]

    print(f"📊 Selected {len(selected_samples)} samples for final dataset")

    # Convert back to our format (group by query)
    query_groups = {}
    for sample in selected_samples:
        query = sample['query']
        if query not in query_groups:
            query_groups[query] = {
                'query': query,
                'documents': [],
                'labels': []
            }
        query_groups[query]['documents'].append(sample['document'])
        query_groups[query]['labels'].append(sample['label'])

    # Convert to final format
    final_data = []
    for query_data in query_groups.values():
        # Ensure we have at least 1 positive and some negatives
        if sum(query_data['labels']) >= 1:  # At least one positive
            final_data.append(query_data)

    print(f"📊 Final dataset has {len(final_data)} queries")

    # Split into train/val/test
    random.shuffle(final_data)

    n_val = int(len(final_data) * val_split)
    n_test = int(len(final_data) * test_split)
    n_train = len(final_data) - n_val - n_test

    train_data = final_data[:n_train]
    val_data = final_data[n_train:n_train + n_val]
    test_data = final_data[n_train + n_val:]

    print(f"📊 Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved mixed dataset to {output_path}")

    # Show statistics
    dataset_stats = {}
    for sample in selected_samples[:1000]:  # Sample for stats
        dataset = sample.get('dataset', 'unknown')
        dataset_stats[dataset] = dataset_stats.get(dataset, 0) + 1

    print("\n📈 Dataset composition:")
    for dataset, count in dataset_stats.items():
        percentage = count / sum(dataset_stats.values()) * 100
        print(".1f")

def main():
    parser = argparse.ArgumentParser(description='Create mixed datasets for reranking')
    parser.add_argument('--output', default='datasets/mixed_dataset.json',
                       help='Output path for mixed dataset')
    parser.add_argument('--total-samples', type=int, default=5000,
                       help='Total number of samples in final dataset')
    parser.add_argument('--msmarco-samples', type=int, default=3000,
                       help='Number of samples from MS MARCO')
    parser.add_argument('--nq-path', help='Path to Natural Questions dataset (if available)')
    parser.add_argument('--nq-samples', type=int, default=2000,
                       help='Number of samples from Natural Questions')
    parser.add_argument('--custom-json', action='append',
                       help='Custom JSON datasets: --custom-json path:num_samples')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Build datasets configuration
    datasets_config = {}

    # MS MARCO
    if args.msmarco_samples > 0:
        datasets_config['msmarco'] = {
            'type': 'msmarco',
            'samples': args.msmarco_samples
        }

    # Natural Questions
    if args.nq_path and args.nq_samples > 0:
        datasets_config['natural_questions'] = {
            'type': 'json',
            'path': args.nq_path,
            'samples': args.nq_samples
        }

    # Custom JSON datasets
    if args.custom_json:
        for i, custom_spec in enumerate(args.custom_json):
            try:
                path, samples = custom_spec.split(':')
                datasets_config[f'custom_{i}'] = {
                    'type': 'json',
                    'path': path,
                    'samples': int(samples)
                }
            except ValueError:
                print(f"⚠️ Invalid custom dataset format: {custom_spec}")
                print("   Use: --custom-json path/to/dataset.json:1000")

    if not datasets_config:
        print("❌ No datasets specified. Use --msmarco-samples, --nq-path, or --custom-json")
        return

    print("🚀 Creating Mixed Dataset for Reranking")
    print("=" * 60)
    print("Datasets to mix:")
    for name, config in datasets_config.items():
        if config['type'] == 'msmarco':
            print(f"  • MS MARCO: {config['samples']} samples")
        else:
            print(f"  • {name}: {config['path']} ({config['samples']} samples)")

    create_mixed_dataset(
        datasets_config=datasets_config,
        output_path=args.output,
        total_samples=args.total_samples,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )

    print("\n✅ Mixed dataset creation complete!")
    print("\n💡 Usage examples:")
    print(f"  Training: python cli/train.py --dataset {Path(args.output).name} --use-lora")
    print(f"  Retrain:  python cli/retrain.py --dataset {Path(args.output).name}")
    print(f"  Eval:     python cli/eval.py --dataset {Path(args.output).stem}")

if __name__ == "__main__":
    main()