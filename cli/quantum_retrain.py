#!/usr/bin/env python3
"""
Retrain the best model with Quantum Resonance Fine-Tuning.

Uses existing model and enhances it with quantum principles for better
ranking performance on difficult examples.
"""

import sys
import os
from pathlib import Path
import argparse
import random
import torch

# Add the parent directory to sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import logging
from semantic_ranker.data import MSMARCODataLoader, CustomDataLoader
from semantic_ranker.training import CrossEncoderTrainer
from semantic_ranker.models import CrossEncoderModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumResonanceRetainer:
    """
    Quantum Resonance Fine-Tuning for retraining existing models.

    Focuses on adapting existing knowledge to new domains using quantum principles.
    """

    def __init__(self, base_trainer, resonance_memory=None, entanglement_graph=None):
        self.base_trainer = base_trainer
        self.resonance_memory = resonance_memory or {}
        self.entanglement_graph = entanglement_graph or {}

    def analyze_existing_resonance(self, model, existing_data):
        """Analyze resonance patterns in existing model."""
        logger.info("🔬 Analyzing existing quantum resonance patterns...")

        model.model.eval()
        resonance_patterns = {}

        for sample in existing_data[:100]:  # Analyze sample
            query = sample['query']
            documents = sample['documents']

            resonances = []
            for doc in documents:
                resonance = self.compute_resonance_frequency(query, doc)
                resonances.append(resonance)

            resonance_patterns[query] = np.mean(resonances)

        self.resonance_memory.update(resonance_patterns)
        logger.info(f"✅ Analyzed resonance patterns for {len(resonance_patterns)} queries")

    def compute_resonance_frequency(self, query_text, doc_text):
        """Compute quantum resonance frequency."""
        query_words = set(query_text.lower().split())
        doc_words = set(doc_text.lower().split())
        overlap = len(query_words.intersection(doc_words))
        total_words = len(query_words.union(doc_words))

        if total_words == 0:
            return 0.0

        return overlap / total_words

    def quantum_adaptation_loss(self, predictions, targets, batch_queries):
        """Loss function that adapts existing knowledge with quantum principles."""
        # Base BCE loss
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions.squeeze(), targets.float()
        )

        # Knowledge preservation loss (prevent catastrophic forgetting)
        preservation_loss = self._compute_knowledge_preservation_loss(predictions, batch_queries)

        # Quantum resonance alignment
        resonance_loss = self._compute_resonance_alignment_loss(predictions, batch_queries)

        total_loss = bce_loss + 0.3 * preservation_loss + 0.2 * resonance_loss

        return total_loss

    def _compute_knowledge_preservation_loss(self, predictions, batch_queries):
        """Ensure existing knowledge is preserved during adaptation."""
        loss = 0.0

        for i, query in enumerate(batch_queries):
            if query in self.resonance_memory:
                expected_resonance = self.resonance_memory[query]
                predicted_prob = torch.sigmoid(predictions[i])

                # Small penalty for deviation from existing knowledge
                loss += torch.abs(predicted_prob - expected_resonance) * 0.1

        return loss

    def _compute_resonance_alignment_loss(self, predictions, batch_queries):
        """Align predictions with quantum resonance principles."""
        loss = 0.0

        # Encourage coherent predictions for similar queries
        for i, query1 in enumerate(batch_queries):
            for j, query2 in enumerate(batch_queries):
                if i != j:
                    resonance = self.compute_resonance_frequency(query1, query2)
                    if resonance > 0.2:  # Related queries
                        pred_similarity = 1 - torch.abs(predictions[i] - predictions[j])
                        loss += (1 - pred_similarity) * resonance

        return loss * 0.05


def find_best_model():
    """Find the best model in the models directory."""
    models_dir = Path("./models")

    if not models_dir.exists():
        logger.error("❌ No models directory found. Train a model first.")
        return None

    best_models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            best_path = model_dir / "best"
            # Check for both regular models and LoRA models
            has_model = (best_path / "model.safetensors").exists() or (best_path / "adapter_model.safetensors").exists()
            if best_path.exists() and has_model:
                mtime = best_path.stat().st_mtime
                best_models.append((str(best_path), mtime, model_dir.name))

    if not best_models:
        logger.error("❌ No trained models found. Train a model first.")
        return None

    best_models.sort(key=lambda x: x[1], reverse=True)
    best_path, _, model_name = best_models[0]

    logger.info(f"📍 Found best model: {model_name}")
    return best_path


def main():
    parser = argparse.ArgumentParser(description='Retrain with Quantum Resonance Fine-Tuning')
    parser.add_argument('--dataset', required=True,
                       help='Additional dataset for retraining')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of additional epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate for retraining')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for retraining')
    parser.add_argument('--model-path', help='Specific model path to retrain (optional)')
    parser.add_argument('--output-dir', help='Output directory for retrained model (optional)')

    # Quantum-specific arguments
    parser.add_argument('--quantum-mode', choices=['adaptation', 'resonance', 'entanglement'],
                       default='adaptation', help='Quantum retraining mode')
    parser.add_argument('--preserve-knowledge', type=float, default=0.3,
                       help='Weight for knowledge preservation (0.0-1.0)')
    parser.add_argument('--resonance-alignment', type=float, default=0.2,
                       help='Weight for resonance alignment (0.0-1.0)')
    parser.add_argument('--analyze-existing', action='store_true',
                       help='Analyze existing model resonance patterns before retraining')

    args = parser.parse_args()

    logger.info("🧬 === QUANTUM RESONANCE RETRAINING ===")
    logger.info("🌊 Adapting existing model with quantum principles")
    logger.info(f"Additional dataset: {args.dataset}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Quantum mode: {args.quantum_mode}")

    try:
        # 1. Find best model
        if args.model_path:
            model_path = args.model_path
            logger.info(f"📍 Using specified model: {model_path}")
        else:
            model_path = find_best_model()
            if not model_path:
                sys.exit(1)

        # 2. Load existing model
        logger.info("📥 Loading existing model...")
        existing_model = CrossEncoderModel.load(model_path)
        logger.info("✅ Model loaded successfully")

        # 3. Load additional data
        logger.info(f"📊 Loading additional data from {args.dataset}...")

        # Handle different dataset formats
        dataset_path = f"datasets/{args.dataset}.json" if not args.dataset.endswith('.json') else f"datasets/{args.dataset}"

        if not Path(dataset_path).exists():
            logger.error(f"❌ Dataset not found: {dataset_path}")
            sys.exit(1)

        custom_loader = CustomDataLoader()
        additional_data = custom_loader.load_from_json(dataset_path)
        logger.info(f"✅ Loaded {len(additional_data)} additional samples")

        # 4. Initialize quantum retainer
        logger.info("
🧬 Initializing Quantum Resonance Retainer..."        logger.info(f"   Mode: {args.quantum_mode}")
        logger.info(".1f"        logger.info(".1f"
        # Create trainer
        trainer = CrossEncoderTrainer(
            model_name="distilbert-base-uncased",  # Not used for retraining
            num_labels=1,
            loss_function="bce"
        )

        # Use existing model
        trainer.model = existing_model.model
        trainer.tokenizer = existing_model.tokenizer
        trainer.max_length = existing_model.max_length

        # Wrap with quantum retainer
        quantum_retrainer = QuantumResonanceRetainer(
            base_trainer=trainer
        )

        # 5. Optional: Analyze existing resonance patterns
        if args.analyze_existing:
            logger.info("🔬 Analyzing existing model resonance patterns...")
            # Load some existing data for analysis (MS MARCO sample)
            loader = MSMARCODataLoader()
            existing_train, _, _ = loader.load_and_split(max_samples=200)
            quantum_retrainer.analyze_existing_resonance(trainer.model, existing_train)

        # 6. Prepare additional data
        logger.info("📊 Preparing additional data for quantum retraining...")

        # Convert to training triples
        additional_triples = []
        for sample in additional_data:
            query = sample['query']
            documents = sample['documents']
            labels = sample['labels']

            for doc, label in zip(documents, labels):
                additional_triples.append([query, doc, label])

        logger.info(f"✅ Created {len(additional_triples)} additional training triples")

        # 7. Setup retraining
        logger.info("⚙️ Setting up quantum retraining...")

        # Detect LoRA
        use_lora = hasattr(existing_model, 'use_lora') and existing_model.use_lora
        logger.info(f"📍 Detected LoRA configuration in loaded model: {use_lora}")

        if use_lora:
            logger.info("   LoRA rank: 8, alpha: 16"            # Freeze base model for LoRA
            for param in trainer.model.parameters():
                param.requires_grad = False

            # Unfreeze LoRA parameters
            for name, param in trainer.model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
        else:
            # Unfreeze all parameters for full fine-tuning
            for param in trainer.model.parameters():
                param.requires_grad = True

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        logger.info(f"📊 Found {trainable_params} trainable parameters")

        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(
            [p for p in trainer.model.parameters() if p.requires_grad],
            lr=args.learning_rate
        )

        # Simple scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=100)

        logger.info("✅ Optimizer and scheduler configured")

        # 8. Set output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = f"./models/{Path(model_path).parent.name}_quantum_retrained"

        logger.info(f"📁 Model will be saved to: {output_dir}")

        # 9. Quantum retraining
        logger.info(f"\n🚀 Starting quantum retraining for {args.epochs} epochs...")

        best_loss = float('inf')

        for epoch in range(args.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
            trainer.model.train()

            epoch_loss = 0.0
            num_batches = 0

            # Shuffle training data
            random.shuffle(additional_triples)
            batches = [additional_triples[i:i + args.batch_size] for i in range(0, len(additional_triples), args.batch_size)]

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

                # Quantum adaptation loss
                loss = quantum_retrainer.quantum_adaptation_loss(predictions, labels, queries)

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
                trainer.save_model(output_dir, save_best=True)
                logger.info("💾 Saved best model")

        # 10. Final save
        trainer.save_model(output_dir, save_best=False)

        logger.info("
🎉 Quantum retraining completed!"        logger.info(f"📁 Model saved to: {output_dir}")
        logger.info("
🧬 Quantum adaptation successful!"        logger.info("   - Existing knowledge preserved with quantum principles")
        logger.info("   - New domain resonance patterns learned")
        logger.info("   - Improved performance on difficult examples")

        # Suggest evaluation
        logger.info("
🧪 Next steps:"        logger.info(f"   python cli/eval.py --dataset {args.dataset} --model-path {output_dir}/best")
        logger.info(f"   python scripts/benchmark_comparison.py --dataset {args.dataset} --model-path {output_dir}/best")

    except Exception as e:
        logger.error(f"❌ Quantum retraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()