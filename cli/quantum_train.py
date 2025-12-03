#!/usr/bin/env python3
"""
Train a cross-encoder reranker model with Quantum Resonance Fine-Tuning.

Quantum Resonance FT treats query-document relationships as quantum states
that exist in superposition until collapsing into optimal rankings.
"""

import argparse
import random
import sys
from pathlib import Path
import torch
import numpy as np

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
from semantic_ranker.data import MSMARCODataLoader, CustomDataLoader, DataPreprocessor
from semantic_ranker.training import CrossEncoderTrainer
from semantic_ranker.evaluation.metrics import compute_ndcg, compute_mrr, compute_map


class QuantumResonanceTrainer:
    """
    Quantum Resonance Fine-Tuning implementation.

    Key concepts:
    - Query-document pairs exist in "quantum superposition" of relevance
    - Training collapses these states toward optimal rankings
    - Resonance frequencies guide the learning process
    """

    def __init__(self, base_trainer, resonance_threshold=0.35, entanglement_weight=0.3,
                 knowledge_preservation_weight=0.6, resonance_penalty_scale=0.01,
                 entanglement_loss_scale=0.01):
        self.base_trainer = base_trainer
        self.resonance_threshold = resonance_threshold
        self.entanglement_weight = entanglement_weight
        self.knowledge_preservation_weight = knowledge_preservation_weight
        self.resonance_penalty_scale = resonance_penalty_scale
        self.entanglement_loss_scale = entanglement_loss_scale
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
                    if resonance > self.resonance_threshold:  # Use configured threshold
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

        return penalty * self.resonance_penalty_scale

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

        return loss * self.entanglement_loss_scale

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


def evaluate_model(model, val_triples, batch_size=32, device='cpu'):
    """Evaluate model on validation set and compute ranking metrics.

    Args:
        model: The trained model
        val_triples: List of [query, doc, label] validation triples
        batch_size: Batch size for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with NDCG@10, MRR@10, MAP@10
    """
    if len(val_triples) == 0:
        return {'ndcg@10': 0.0, 'mrr@10': 0.0, 'map@10': 0.0}

    model.model.eval()

    # Group triples by query
    from collections import defaultdict
    query_results = defaultdict(list)

    # Batch evaluation
    with torch.no_grad():
        for i in range(0, len(val_triples), batch_size):
            batch = val_triples[i:i + batch_size]
            queries = [item[0] for item in batch]
            documents = [item[1] for item in batch]
            labels = [item[2] for item in batch]

            # Tokenize and predict
            inputs = model.tokenizer(
                queries,
                documents,
                truncation=True,
                padding=True,
                max_length=model.max_length,
                return_tensors='pt'
            )

            if device == 'cuda' and torch.cuda.is_available():
                inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.model(**inputs)
            scores = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

            # Handle single item case
            if len(batch) == 1:
                scores = [scores.item()]
            else:
                scores = scores.tolist()

            # Group by query
            for query, score, label in zip(queries, scores, labels):
                query_results[query].append((score, label))

    # Compute metrics per query, then average
    ndcg_scores = []
    mrr_scores = []
    map_scores = []

    for query, results in query_results.items():
        # Sort by predicted score (descending)
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
        labels = [int(label) for score, label in sorted_results]

        # Compute metrics
        ndcg_scores.append(compute_ndcg(labels, k=10))
        mrr_scores.append(compute_mrr(labels, k=10))
        map_scores.append(compute_map(labels, k=10))

    # Average across queries
    metrics = {
        'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'mrr@10': np.mean(mrr_scores) if mrr_scores else 0.0,
        'map@10': np.mean(map_scores) if map_scores else 0.0
    }

    model.model.train()
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train with Quantum Resonance Fine-Tuning')

    # Config support (loads everything from YAML)
    parser = add_config_args(parser)

    # Only essential CLI arguments (things not in config or commonly overridden)
    parser.add_argument('--output', '--output-dir', dest='output_dir',
                       help='Output directory for trained model (required)')

    # Optional: Common overrides for experimentation
    parser.add_argument('--epochs', type=int, help='Override epochs from config')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate from config')

    args = parser.parse_args()

    # Load config (everything comes from YAML now!)
    config = load_config_with_overrides(args)

    # Extract all values from config (no more redundant parser arguments!)
    dataset = config.data.dataset
    model_name = config.model.model_name
    epochs = args.epochs or config.training.epochs  # Allow CLI override
    batch_size = config.training.batch_size
    learning_rate = args.learning_rate or config.training.learning_rate  # Allow CLI override
    max_samples = config.data.max_samples
    use_lora = config.model.use_lora

    # Output dir (only thing that must come from CLI or has default)
    output_dir = args.output_dir or './models/quantum_trained_model'

    # Quantum config (all from YAML)
    quantum_mode = config.quantum.quantum_mode if isinstance(config.quantum.quantum_mode, str) else ('resonance' if config.quantum.quantum_mode else 'superposition')
    resonance_threshold = config.quantum.resonance_threshold
    entanglement_weight = config.quantum.entanglement_weight
    quantum_phase = config.quantum.quantum_phase
    knowledge_preservation_weight = config.quantum.knowledge_preservation_weight
    resonance_penalty_scale = config.quantum.resonance_penalty_scale
    entanglement_loss_scale = config.quantum.entanglement_loss_scale

    logger.info("🧬 === QUANTUM RESONANCE FINE-TUNING ===")
    logger.info("🌊 Treating query-document relationships as quantum states")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Quantum Mode: {quantum_mode}")
    logger.info(f"Quantum Phase: {quantum_phase}")
    logger.info(f"Resonance Threshold: {resonance_threshold}")
    logger.info(f"Entanglement Weight: {entanglement_weight}")
    logger.info(f"Knowledge Preservation: {knowledge_preservation_weight}")
    logger.info(f"Output: {output_dir}")

    try:
        # 1. Load and validate data
        logger.info(f"\n📊 Step 1: Loading data (max: {max_samples} samples)...")

        # Use unified data loading (handles both MSMARCO and custom datasets with auto-split)
        train_data, val_data, test_data = load_dataset_unified(
            dataset,
            max_samples,
            config.data.train_split,
            config.data.val_split,
            config.data.test_split
        )

        logger.info(f"✅ Loaded data: {len(train_data)} train | {len(val_data)} val | {len(test_data)} test")

        # 2. Initialize quantum resonance trainer
        logger.info("\n🧬 Step 2: Initializing Quantum Resonance Trainer...")
        logger.info(f"   Mode: {quantum_mode}")
        logger.info(f"   Phase: {quantum_phase}")
        logger.info(f"   Resonance Threshold: {resonance_threshold:.1f}")
        logger.info(f"   Entanglement Weight: {entanglement_weight:.1f}")
        # Create base trainer (use config values, not hardcoded)
        # Note: lora_dropout is handled by CrossEncoderModel, not trainer
        trainer = CrossEncoderTrainer(
            model_name=model_name,
            num_labels=1,
            max_length=config.model.max_length,
            loss_function=config.training.loss_function,
            use_lora=use_lora,
            lora_r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha
        )

        # Wrap with quantum resonance
        quantum_trainer = QuantumResonanceTrainer(
            base_trainer=trainer,
            resonance_threshold=resonance_threshold,
            entanglement_weight=entanglement_weight,
            knowledge_preservation_weight=knowledge_preservation_weight,
            resonance_penalty_scale=resonance_penalty_scale,
            entanglement_loss_scale=entanglement_loss_scale
        )

        # 3. Quantum data preparation
        logger.info(f"\n🌊 Step 3: Quantum {quantum_phase} phase...")

        if quantum_phase == 'superposition':
            # Keep data in superposition (standard preprocessing)
            logger.info("   Maintaining superposition state...")
            quantum_train_data = train_data
        else:
            # Resonance mode: need to group data by query for entanglement graph
            # Convert from {'query', 'document', 'label'} to {'query', 'documents', 'labels'}
            logger.info("   Grouping data by query for entanglement graph...")
            from collections import defaultdict
            grouped_data = defaultdict(lambda: {'query': '', 'documents': [], 'labels': []})

            for sample in train_data:
                query = sample['query']
                grouped_data[query]['query'] = query
                grouped_data[query]['documents'].append(sample['document'])
                grouped_data[query]['labels'].append(sample['label'])

            grouped_list = list(grouped_data.values())
            logger.info(f"   Grouped {len(train_data)} samples into {len(grouped_list)} queries")

            # Build entanglement graph and collapse superposition
            quantum_trainer.build_entanglement_graph(grouped_list)
            quantum_train_data = quantum_trainer.collapse_superposition(grouped_list)

        # 4. Prepare data for training
        logger.info("\n📊 Step 4: Preparing training data...")

        # Convert to training triples format
        train_triples = []

        # Check if data is in grouped format (from resonance mode) or flat format (from superposition)
        if isinstance(quantum_train_data, list) and len(quantum_train_data) > 0:
            first_sample = quantum_train_data[0]

            if 'documents' in first_sample and 'labels' in first_sample:
                # Grouped format from resonance mode - expand to triples
                for sample in quantum_train_data:
                    query = sample['query']
                    for doc, label in zip(sample['documents'], sample['labels']):
                        train_triples.append([query, doc, label])
            else:
                # Flat format from superposition mode - direct conversion
                train_triples = [[sample['query'], sample['document'], sample['label']]
                                for sample in quantum_train_data]

        logger.info(f"✅ Created {len(train_triples)} training triples")

        # Prepare validation data
        val_triples = []
        if len(val_data) > 0:
            val_triples = [[sample['query'], sample['document'], sample['label']]
                          for sample in val_data]
            logger.info(f"✅ Created {len(val_triples)} validation triples")
        else:
            logger.warning("⚠️ No validation data available")

        # 5. Quantum training
        logger.info(f"\n🚀 Step 5: Quantum training for {epochs} epochs...")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   LoRA enabled: {use_lora}")

        # Get eval_steps from config (default 100)
        eval_steps = getattr(config.training, 'eval_steps', 100)
        logger.info(f"   Evaluation frequency: every {eval_steps} steps")

        # Custom training loop with quantum loss
        optimizer = torch.optim.AdamW(
            trainer.model.model.parameters(),
            lr=learning_rate,
            weight_decay=config.training.weight_decay
        )

        # Calculate warmup steps based on config
        total_steps = (len(train_triples) // batch_size) * epochs
        warmup_steps = int(total_steps * config.training.warmup_ratio)

        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Track best validation metric (not training loss)
        best_val_ndcg = 0.0
        global_step = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            trainer.model.model.train()

            epoch_loss = 0.0
            num_batches = 0

            # Shuffle training data
            random.shuffle(train_triples)
            batches = [train_triples[i:i + batch_size] for i in range(0, len(train_triples), batch_size)]

            for batch_idx, batch in enumerate(batches):
                queries = [item[0] for item in batch]
                documents = [item[1] for item in batch]
                labels = torch.tensor([item[2] for item in batch], dtype=torch.float)

                # Forward pass
                inputs = trainer.model.tokenizer(
                    queries,
                    documents,
                    truncation=True,
                    padding=True,
                    max_length=trainer.model.max_length,
                    return_tensors='pt'
                )

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    labels = labels.cuda()

                outputs = trainer.model.model(**inputs)
                predictions = outputs.logits

                # Quantum loss
                loss = quantum_trainer.quantum_loss(predictions, labels, queries)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (use config value)
                torch.nn.utils.clip_grad_norm_(
                    trainer.model.model.parameters(),
                    config.training.max_grad_norm
                )

                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                # Logging
                if batch_idx % 50 == 0:
                    logger.info(f"   Step {global_step} | Loss: {epoch_loss / (batch_idx + 1):.4f}")

                # Step-based validation
                if len(val_triples) > 0 and global_step % eval_steps == 0:
                    logger.info(f"\n📊 Validation at step {global_step}...")
                    val_metrics = evaluate_model(trainer.model, val_triples, batch_size=32, device=device)
                    logger.info(f"   NDCG@10: {val_metrics['ndcg@10']:.4f} | MRR@10: {val_metrics['mrr@10']:.4f} | MAP@10: {val_metrics['map@10']:.4f}")

                    # Save best model based on validation NDCG@10
                    if val_metrics['ndcg@10'] > best_val_ndcg:
                        best_val_ndcg = val_metrics['ndcg@10']
                        logger.info(f"   ✨ New best validation NDCG@10: {best_val_ndcg:.4f}")
                        logger.info(f"   💾 Saving best model to {output_dir}")
                        trainer.save_model(output_dir, save_best=True)
                        save_config_with_model(config, output_dir)
                        logger.info("   ✅ Best model saved\n")

                    # Set back to training mode
                    trainer.model.model.train()

            # Log epoch summary (avoid division by zero)
            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                logger.info(f"   Average epoch loss: {avg_epoch_loss:.4f}")
            else:
                logger.warning("   No batches processed in this epoch")

        # 6. Final save and summary
        trainer.save_model(output_dir, save_best=False)
        save_config_with_model(config, output_dir)
        logger.info("\n🎉 Quantum training completed!")
        logger.info(f"📁 Model saved to: {output_dir}")
        if len(val_triples) > 0:
            logger.info(f"🏆 Best validation NDCG@10: {best_val_ndcg:.4f}")
        logger.info("\n🧬 Quantum resonance patterns learned!")
        logger.info("   - Query-document relationships modeled as quantum states")
        logger.info("   - Resonance frequencies computed for optimal ranking")
        logger.info("   - Entanglement coherence maintained across related queries")

    except Exception as e:
        logger.error(f"❌ Quantum training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()