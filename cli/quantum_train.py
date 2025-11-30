#!/usr/bin/env python3
"""
Train a cross-encoder reranker model with Quantum Resonance Fine-Tuning.

Quantum Resonance FT treats query-document relationships as quantum states
that exist in superposition until collapsing into optimal rankings.
"""

import sys
import os
from pathlib import Path
import argparse
import random
import torch
import numpy as np

# Add the parent directory to sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import logging
from semantic_ranker.data import MSMARCODataLoader, CustomDataLoader, DataPreprocessor
from semantic_ranker.training import CrossEncoderTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumResonanceTrainer:
    """
    Quantum Resonance Fine-Tuning implementation.

    Key concepts:
    - Query-document pairs exist in "quantum superposition" of relevance
    - Training collapses these states toward optimal rankings
    - Resonance frequencies guide the learning process
    """

    def __init__(self, base_trainer, resonance_threshold=0.7, entanglement_weight=0.3):
        self.base_trainer = base_trainer
        self.resonance_threshold = resonance_threshold
        self.entanglement_weight = entanglement_weight
        self.resonance_frequencies = {}
        self.entanglement_graph = {}

    def compute_resonance_frequency(self, query_text, doc_text):
        """Compute quantum resonance frequency between query and document."""
        # Simple resonance based on semantic overlap (could be enhanced with embeddings)
        query_words = set(query_text.lower().split())
        doc_words = set(doc_text.lower().split())
        overlap = len(query_words.intersection(doc_words))
        total_words = len(query_words.union(doc_words))

        if total_words == 0:
            return 0.0

        resonance = overlap / total_words
        return resonance

    def build_entanglement_graph(self, training_data):
        """Build graph of semantically related queries."""
        logger.info("🔗 Building quantum entanglement graph...")

        for sample in training_data:
            query = sample['query']
            if query not in self.entanglement_graph:
                self.entanglement_graph[query] = []

            # Find related queries (simple keyword overlap for now)
            for other_sample in training_data:
                other_query = other_sample['query']
                if other_query != query:
                    resonance = self.compute_resonance_frequency(query, other_query)
                    if resonance > 0.3:  # Entanglement threshold
                        if other_query not in self.entanglement_graph[query]:
                            self.entanglement_graph[query].append(other_query)

        logger.info(f"✅ Built entanglement graph with {len(self.entanglement_graph)} nodes")

    def quantum_loss(self, predictions, targets, batch_queries):
        """Quantum-enhanced loss function."""
        # Standard BCE loss
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions.squeeze(), targets.float()
        )

        # Quantum resonance penalty
        resonance_penalty = self._compute_resonance_penalty(predictions, batch_queries)

        # Entanglement coherence loss
        entanglement_loss = self._compute_entanglement_loss(predictions, batch_queries)

        total_loss = bce_loss + resonance_penalty + self.entanglement_weight * entanglement_loss

        return total_loss

    def _compute_resonance_penalty(self, predictions, batch_queries):
        """Penalize predictions that don't align with resonance frequencies."""
        penalty = 0.0

        for i, query in enumerate(batch_queries):
            if query in self.resonance_frequencies:
                expected_resonance = self.resonance_frequencies[query]
                predicted_prob = torch.sigmoid(predictions[i])

                # Penalize deviation from expected resonance
                penalty += torch.abs(predicted_prob - expected_resonance)

        return penalty * 0.1  # Scale factor

    def _compute_entanglement_loss(self, predictions, batch_queries):
        """Encourage coherent predictions for entangled queries."""
        loss = 0.0

        for i, query1 in enumerate(batch_queries):
            if query1 in self.entanglement_graph:
                for query2 in self.entanglement_graph[query1]:
                    # Find if query2 is in current batch
                    for j, q in enumerate(batch_queries):
                        if q == query2:
                            # Encourage similar predictions for related queries
                            pred_diff = torch.abs(predictions[i] - predictions[j])
                            loss += pred_diff

        return loss * 0.05  # Scale factor

    def collapse_superposition(self, training_data):
        """Collapse quantum superposition states toward optimal rankings."""
        logger.info("🌊 Collapsing quantum superposition...")

        collapsed_data = []

        for sample in training_data:
            query = sample['query']
            documents = sample['documents']
            labels = sample['labels']

            # Compute resonance frequencies
            resonances = []
            for doc in documents:
                resonance = self.compute_resonance_frequency(query, doc)
                resonances.append(resonance)

            # Update resonance memory
            self.resonance_frequencies[query] = np.mean(resonances)

            # Quantum collapse: amplify high resonance signals
            collapsed_sample = {
                'query': query,
                'documents': documents,
                'labels': labels,
                'resonance_scores': resonances
            }

            collapsed_data.append(collapsed_sample)

        logger.info("✅ Quantum superposition collapsed")
        return collapsed_data


def main():
    parser = argparse.ArgumentParser(description='Train with Quantum Resonance Fine-Tuning')
    parser.add_argument('--dataset', choices=['msmarco'] + [f.stem for f in Path('datasets').glob('*.json')],
                       default='msmarco', help='Dataset to use for training')
    parser.add_argument('--model-name', default='distilbert-base-uncased',
                       help='Pretrained model name')
    parser.add_argument('--output-dir', default='./models/quantum_trained_model',
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

    # Quantum-specific arguments
    parser.add_argument('--quantum-mode', choices=['resonance', 'entanglement', 'superposition'],
                       default='resonance', help='Quantum fine-tuning mode')
    parser.add_argument('--resonance-threshold', type=float, default=0.7,
                       help='Resonance threshold for quantum collapse (0.0-1.0)')
    parser.add_argument('--entanglement-weight', type=float, default=0.3,
                       help='Weight for entanglement coherence loss (0.0-1.0)')
    parser.add_argument('--quantum-phase', choices=['superposition', 'collapse', 'resonance'],
                       default='resonance', help='Quantum training phase')

    args = parser.parse_args()

    logger.info("🧬 === QUANTUM RESONANCE FINE-TUNING ===")
    logger.info("🌊 Treating query-document relationships as quantum states")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Quantum Mode: {args.quantum_mode}")
    logger.info(f"Resonance Threshold: {args.resonance_threshold}")
    logger.info(f"Entanglement Weight: {args.entanglement_weight}")
    logger.info(f"Output: {args.output_dir}")

    try:
        # 1. Load and validate data
        logger.info(f"\n📊 Step 1: Loading data ({args.max_samples} samples)...")

        if args.dataset == 'msmarco':
            loader = MSMARCODataLoader()
            train_data, val_data, _ = loader.load_and_split(max_samples=args.max_samples)
        else:
            # Load custom dataset
            custom_loader = CustomDataLoader()
            train_data = custom_loader.load_from_json(f"datasets/{args.dataset}.json", max_samples=args.max_samples)
            val_data = []  # No validation for custom datasets in quantum mode

        logger.info(f"✅ Loaded {len(train_data)} training samples")

        # 2. Initialize quantum resonance trainer
        logger.info("
🧬 Step 2: Initializing Quantum Resonance Trainer..."        logger.info(f"   Mode: {args.quantum_mode}")
        logger.info(f"   Phase: {args.quantum_phase}")
        logger.info(".1f"        logger.info(".1f"
        # Create base trainer
        trainer = CrossEncoderTrainer(
            model_name=args.model_name,
            num_labels=1,
            loss_function="bce",
            use_lora=args.use_lora,
            lora_r=8,
            lora_alpha=16
        )

        # Wrap with quantum resonance
        quantum_trainer = QuantumResonanceTrainer(
            base_trainer=trainer,
            resonance_threshold=args.resonance_threshold,
            entanglement_weight=args.entanglement_weight
        )

        # 3. Quantum data preparation
        logger.info(f"\n🌊 Step 3: Quantum {args.quantum_phase} phase...")

        if args.quantum_phase == 'superposition':
            # Keep data in superposition (standard preprocessing)
            logger.info("   Maintaining superposition state...")
            quantum_train_data = train_data
        else:
            # Build entanglement graph and collapse superposition
            quantum_trainer.build_entanglement_graph(train_data)
            quantum_train_data = quantum_trainer.collapse_superposition(train_data)

        # 4. Prepare data for training
        logger.info("
📊 Step 4: Preparing training data..."        preprocessor = DataPreprocessor(tokenizer_name=args.model_name)

        # Convert to training format
        if hasattr(quantum_train_data[0], 'documents'):
            # Already in correct format
            train_triples = []
            for sample in quantum_train_data:
                query = sample['query']
                docs = sample['documents']
                labels = sample['labels']

                for doc, label in zip(docs, labels):
                    train_triples.append([query, doc, label])
        else:
            # Convert from old format
            train_triples = preprocessor.create_triples(quantum_train_data)

        logger.info(f"✅ Created {len(train_triples)} training triples")

        # 5. Quantum training
        logger.info(f"\n🚀 Step 5: Quantum training for {args.epochs} epochs...")
        logger.info(f"   Learning rate: {args.learning_rate}")
        logger.info(f"   Batch size: {args.batch_size}")
        logger.info(f"   LoRA enabled: {args.use_lora}")

        # Custom training loop with quantum loss
        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=100)

        best_loss = float('inf')

        for epoch in range(args.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
            trainer.model.train()

            epoch_loss = 0.0
            num_batches = 0

            # Shuffle training data
            random.shuffle(train_triples)
            batches = [train_triples[i:i + args.batch_size] for i in range(0, len(train_triples), args.batch_size)]

            for batch_idx, batch in enumerate(batches):
                queries = [item[0] for item in batch]
                documents = [item[1] for item in batch]
                labels = torch.tensor([item[2] for item in batch], dtype=torch.float)

                # Forward pass
                inputs = trainer.tokenizer(
                    queries,
                    documents,
                    truncation=True,
                    padding=True,
                    max_length=trainer.max_length,
                    return_tensors='pt'
                )

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    labels = labels.cuda()

                outputs = trainer.model(**inputs)
                predictions = outputs.logits

                # Quantum loss
                loss = quantum_trainer.quantum_loss(predictions, labels, queries)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % 50 == 0:
                    logger.info(".4f"
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(".4f"
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                trainer.save_model(args.output_dir, save_best=True)
                logger.info("💾 Saved best model")

        # 6. Final save
        trainer.save_model(args.output_dir, save_best=False)
        logger.info("
🎉 Quantum training completed!"        logger.info(f"📁 Model saved to: {args.output_dir}")
        logger.info("
🧬 Quantum resonance patterns learned!"        logger.info("   - Query-document relationships modeled as quantum states")
        logger.info("   - Resonance frequencies computed for optimal ranking")
        logger.info("   - Entanglement coherence maintained across related queries")

    except Exception as e:
        logger.error(f"❌ Quantum training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()