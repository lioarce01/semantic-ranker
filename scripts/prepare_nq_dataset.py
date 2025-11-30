#!/usr/bin/env python3
"""
Prepare Natural Questions dataset for reranking training.

Converts NQ to the format expected by the reranker:
- query: the question
- documents: list of documents (1 positive + negatives)
- labels: list of 0s and 1s (1 for relevant, 0 for irrelevant)
"""

import json
import random
from pathlib import Path
import argparse

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("❌ datasets library required. Install with: pip install datasets")

def prepare_nq_dataset(
    output_path: str = "datasets/natural_questions.json",
    max_samples: int = 5000,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42
):
    """
    Prepare Natural Questions dataset for reranking.

    Args:
        output_path: Where to save the prepared dataset
        max_samples: Maximum number of samples to prepare
        val_split: Proportion for validation
        test_split: Proportion for testing
        seed: Random seed
    """
    if not HAS_DATASETS:
        return

    random.seed(seed)

    print("📥 Loading Natural Questions dataset...")
    try:
        # Load NQ dataset
        nq = load_dataset("natural_questions", split="train")
        print(f"✅ Loaded {len(nq)} examples from NQ")
    except Exception as e:
        print(f"❌ Failed to load NQ: {e}")
        print("💡 Make sure you have: pip install datasets")
        return

    # Prepare data
    prepared_data = []
    processed = 0

    print(f"🔄 Preparing {min(max_samples, len(nq))} samples...")

    for item in nq:
        if processed >= max_samples:
            break

        try:
            # Check if question key exists
            if 'question' not in item:
                continue

            # Extract question text - handle different formats
            question_field = item['question']

            if isinstance(question_field, str):
                question = question_field
            elif isinstance(question_field, dict) and 'text' in question_field:
                question = question_field['text']
            else:
                # Try to convert or skip
                try:
                    question = str(question_field)
                except:
                    continue  # Skip if we can't extract question

            # Extract answers from annotations
            answers = []
            has_long_answer = False

            if 'annotations' in item and item['annotations'] and isinstance(item['annotations'], list):
                # Get answers from annotations
                for annotation in item['annotations']:
                    if not isinstance(annotation, dict):
                        continue

                    # Try short answers first
                    if 'short_answers' in annotation and annotation['short_answers'] and isinstance(annotation['short_answers'], list):
                        for short_answer in annotation['short_answers']:
                            if isinstance(short_answer, dict) and 'text' in short_answer and short_answer['text'].strip():
                                answers.append(short_answer['text'].strip())

                    # Try yes/no answers
                    elif 'yes_no_answer' in annotation and annotation['yes_no_answer'] in ['YES', 'NO']:
                        answers.append(annotation['yes_no_answer'])

                    # Check for long answer
                    if 'long_answer' in annotation and isinstance(annotation.get('long_answer'), dict) and 'start_token' in annotation['long_answer']:
                        has_long_answer = True

            # If no clear short answers but has long answer, create a synthetic answer
            if not answers and has_long_answer:
                # For samples with long answers but no short answers, create a synthetic response
                synthetic_answers = [
                    "The answer can be found in the provided context.",
                    "Please refer to the detailed information in the document.",
                    "The specific answer is contained within the text.",
                    "Consult the relevant section of the document for the answer."
                ]
                answers.append(random.choice(synthetic_answers))

            # If still no answers, skip this sample
            if not answers:
                continue

            # Debug: print item structure for first few samples
            if processed < 2:
                print(f"DEBUG Sample {processed}: Found {len(answers)} answers")
                if answers:
                    print(f"DEBUG First answer: {answers[0][:50]}...")

            # Get the correct answer (first one found)
            correct_answer = answers[0]

            # Create positive document using the answer
            positive_doc = f"{correct_answer}. This is the correct answer to the question: {question}"

            # Create negative documents (use other answers or generic text)
            negative_docs = []

            # Use other answers as negatives if available
            if len(answers) > 1:
                other_answers = answers[1:4]  # Up to 3 other answers
                for ans in other_answers:
                    neg_doc = f"{ans}. This is an alternative answer to: {question}"
                    negative_docs.append(neg_doc)

                # Fill remaining negatives with generic text
                while len(negative_docs) < 3:
                    generic_answers = [
                        f"This information is not relevant to the question: {question}",
                        f"General knowledge about various topics unrelated to: {question}",
                        f"Random facts that don't answer: {question}"
                    ]
                    neg_doc = random.choice(generic_answers)
                    if neg_doc not in negative_docs:
                        negative_docs.append(neg_doc)

                # Create the sample
                sample = {
                    "query": question,
                    "documents": [positive_doc] + negative_docs,
                    "labels": [1] + [0] * len(negative_docs)
                }

                prepared_data.append(sample)
                processed += 1

                if processed % 500 == 0:
                    print(f"  📊 Processed {processed}/{min(max_samples, len(nq))} samples")

        except Exception as e:
            print(f"⚠️ Skipping sample {processed}: {e}")
            continue

    print(f"✅ Prepared {len(prepared_data)} samples")

    # Split into train/val/test
    random.shuffle(prepared_data)

    n_val = int(len(prepared_data) * val_split)
    n_test = int(len(prepared_data) * test_split)
    n_train = len(prepared_data) - n_val - n_test

    train_data = prepared_data[:n_train]
    val_data = prepared_data[n_train:n_train + n_val]
    test_data = prepared_data[n_train + n_val:]

    print(f"📊 Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prepared_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved to {output_path}")
    print(f"📋 Total samples: {len(prepared_data)}")

    # Show sample
    if prepared_data:
        print("\n📖 Sample entry:")
        sample = prepared_data[0]
        print(f"Query: {sample['query']}")
        print(f"Documents: {len(sample['documents'])}")
        print(f"Labels: {sample['labels']}")
        print(f"Positive doc: {sample['documents'][0][:100]}...")

def main():
    parser = argparse.ArgumentParser(description='Prepare Natural Questions dataset for reranking')
    parser.add_argument('--output', default='datasets/natural_questions.json',
                       help='Output path for prepared dataset')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Maximum number of samples to prepare')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    print("🚀 Preparing Natural Questions Dataset for Reranking")
    print("=" * 60)

    prepare_nq_dataset(
        output_path=args.output,
        max_samples=args.samples,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )

    print("\n✅ Dataset preparation complete!")
    print("\n💡 Usage examples:")
    print(f"  Training: python cli/train.py --dataset {Path(args.output).name} --use-lora")
    print(f"  Retrain:  python cli/retrain.py --dataset {Path(args.output).name}")
    print(f"  Eval:     python cli/eval.py --dataset {Path(args.output).stem}")

if __name__ == "__main__":
    main()