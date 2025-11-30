#!/usr/bin/env python3
"""
Hard Negative Mining Script

Identifies hard negative examples that the current model ranks poorly.
Creates improved training data for better model performance.
"""

import json
import torch
from pathlib import Path
import argparse
from tqdm import tqdm

def load_model(model_path):
    """Load the trained model"""
    from semantic_ranker.models.cross_encoder import CrossEncoderModel
    print(f"📥 Loading model from {model_path}")
    model = CrossEncoderModel.load(model_path)
    model.model.eval()  # Put the underlying transformers model in eval mode
    return model

def mine_hard_negatives(model, dataset_path, top_k=5, max_samples=1000):
    """
    Mine hard negatives from dataset using current model predictions.

    Args:
        model: Trained model
        dataset_path: Path to dataset JSON
        top_k: Number of hard negatives to mine per query
        max_samples: Maximum samples to process
    """
    print("🔍 Mining hard negatives...")

    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    hard_negative_samples = []

    print(f"📊 Processing {min(len(data), max_samples)} samples...")

    for i, sample in enumerate(tqdm(data[:max_samples])):
        query = sample['query']
        documents = sample['documents']
        labels = sample['labels']

        # Skip if no positive document
        if 1 not in labels:
            continue

        # Get model scores for all documents
        scores = []
        for doc in documents:
            # Prepare input for cross-encoder
            inputs = model.tokenizer(
                query,
                doc,
                truncation=True,
                padding=True,
                max_length=model.max_length,
                return_tensors='pt'
            )

            with torch.no_grad():
                score = model.model(**inputs).logits.item()
                scores.append(score)

        # Find the positive document index
        positive_idx = labels.index(1)

        # Find hard negatives: documents ranked high by model but are actually negative
        negative_indices = [j for j, label in enumerate(labels) if label == 0]

        if not negative_indices:
            continue

        # Sort negative documents by model score (descending - higher scores are harder)
        negative_scores = [(j, scores[j]) for j in negative_indices]
        negative_scores.sort(key=lambda x: x[1], reverse=True)  # Highest scores first

        # Take top-k hard negatives
        hard_negatives = negative_scores[:top_k]

        # Create new training sample with hard negatives
        # Include: 1 positive + k hard negatives
        selected_docs = [documents[positive_idx]]  # Positive first
        selected_labels = [1]

        for neg_idx, _ in hard_negatives:
            selected_docs.append(documents[neg_idx])
            selected_labels.append(0)

        # Only keep if we have at least 1 positive and 2 negatives
        if len(selected_docs) >= 3:
            hard_sample = {
                "query": query,
                "documents": selected_docs,
                "labels": selected_labels
            }
            hard_negative_samples.append(hard_sample)

    print(f"✅ Mined {len(hard_negative_samples)} hard negative samples")
    return hard_negative_samples

def create_hard_negative_dataset(model_path, input_dataset, output_path, top_k=3, max_samples=1000):
    """
    Create a dataset enriched with hard negatives.
    """
    print("🏆 Hard Negative Mining Pipeline")
    print("=" * 50)

    # Load model
    model = load_model(model_path)

    # Mine hard negatives
    hard_samples = mine_hard_negatives(model, input_dataset, top_k=top_k, max_samples=max_samples)

    if not hard_samples:
        print("❌ No hard negative samples found")
        return None

    # Save dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hard_samples, f, indent=2, ensure_ascii=False)

    print("💾 Saved hard negative dataset")
    print(f"📁 Location: {output_path}")
    print(f"📊 Samples: {len(hard_samples)}")
    print(f"📈 Avg docs per query: {sum(len(s['documents']) for s in hard_samples) / len(hard_samples):.1f}")

    return str(output_path)

def combine_datasets(original_path, hard_negative_path, output_path, hard_negative_ratio=0.3):
    """
    Combine original dataset with hard negative samples.
    """
    print("🔀 Combining datasets...")

    # Load datasets
    with open(original_path, 'r', encoding='utf-8') as f:
        original = json.load(f)

    with open(hard_negative_path, 'r', encoding='utf-8') as f:
        hard_negatives = json.load(f)

    # Calculate how many hard negatives to include
    n_hard = int(len(original) * hard_negative_ratio)
    hard_negatives_subset = hard_negatives[:n_hard]

    # Combine
    combined = original + hard_negatives_subset

    # Shuffle
    import random
    random.shuffle(combined)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print("✅ Combined dataset created")
    print(f"📁 Location: {output_path}")
    print(f"📊 Total samples: {len(combined)}")
    print(f"🔍 Original: {len(original)}, Hard negatives: {len(hard_negatives_subset)}")

    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='Hard Negative Mining for improved training')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--input-dataset', required=True, help='Input dataset to mine negatives from')
    parser.add_argument('--output-hard', help='Output path for hard negative dataset')
    parser.add_argument('--output-combined', help='Output path for combined dataset')
    parser.add_argument('--top-k', type=int, default=3, help='Number of hard negatives per query')
    parser.add_argument('--max-samples', type=int, default=500, help='Maximum samples to process')
    parser.add_argument('--combine-ratio', type=float, default=0.3, help='Ratio of hard negatives in combined dataset')

    args = parser.parse_args()

    # Set default output paths
    if not args.output_hard:
        input_name = Path(args.input_dataset).stem
        args.output_hard = f"datasets/{input_name}_hard_negatives.json"

    if not args.output_combined:
        input_name = Path(args.input_dataset).stem
        args.output_combined = f"datasets/{input_name}_with_hard_negatives.json"

    # Mine hard negatives
    hard_negative_path = create_hard_negative_dataset(
        args.model_path,
        args.input_dataset,
        args.output_hard,
        args.top_k,
        args.max_samples
    )

    if hard_negative_path:
        # Combine with original
        combined_path = combine_datasets(
            args.input_dataset,
            hard_negative_path,
            args.output_combined,
            args.combine_ratio
        )

        print("\n🚀 Ready for training:")
        print(f"   python cli/retrain.py --dataset {combined_path} --epochs 3 --learning-rate 2e-5")
        print("\n💡 This should improve performance on difficult examples!")
    else:
        print("❌ Failed to create hard negative dataset")

if __name__ == "__main__":
    main()