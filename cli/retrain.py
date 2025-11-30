#!/usr/bin/env python3
"""
Retrain the best model with additional data.
Responsibility: Only retraining, finds best model automatically.
"""

import sys
import os
from pathlib import Path
import argparse

# Add the parent directory to sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import logging
import random
from semantic_ranker.data import MSMARCODataLoader, CustomDataLoader
from semantic_ranker.training import CrossEncoderTrainer
from semantic_ranker.models import CrossEncoderModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    parser = argparse.ArgumentParser(description='Retrain the best model with additional data')
    parser.add_argument('--dataset', choices=['msmarco'] + [f.stem for f in Path('datasets').glob('*.json')],
                       default='msmarco', help='Additional dataset for retraining')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of additional epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for retraining')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                       help='Learning rate for fine-tuning (lower than initial)')
    parser.add_argument('--samples', type=int, default=500,
                       help='Number of additional samples to use')
    parser.add_argument('--suffix', default='_retrained',
                       help='Suffix for retrained model directory')

    args = parser.parse_args()

    logger.info("=== Model Retraining ===")
    logger.info(f"Additional dataset: {args.dataset}")
    logger.info(f"Additional epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    print()

    # 1. Find and load best model
    model_path = find_best_model()
    if not model_path:
        return

    logger.info("Loading existing model...")
    try:
        model = CrossEncoderModel.load(model_path)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model from {model_path}: {e}")
        return

    # 2. Load additional data
    logger.info(f"Loading additional data from {args.dataset}...")
    if args.dataset == 'msmarco':
        loader = MSMARCODataLoader()
        # Load more data to have both training and validation
        total_samples = min(args.samples + 200, 5000)  # Add 200 for validation
        additional_data, val_data, _ = loader.load_and_split(max_samples=total_samples)
        # Use a portion for validation if we have enough data
        if len(val_data) < 50 and len(additional_data) > 100:
            # Split additional data for validation
            val_split = min(50, len(additional_data) // 4)
            val_data = additional_data[-val_split:]
            additional_data = additional_data[:-val_split]
    else:
        loader = CustomDataLoader()
        dataset_path = f"datasets/{args.dataset}.json"
        all_data = loader.load_from_json(dataset_path)
        additional_data = all_data[:args.samples]
        val_data = []  # No validation data for custom datasets

    logger.info(f"✅ Loaded {len(additional_data)} additional samples")
    if val_data:
        logger.info(f"✅ Loaded {len(val_data)} validation samples")
    else:
        logger.info("ℹ️ No validation data available for retraining")

    # 3. Preprocess additional data
    logger.info("Preparing additional data for retraining...")

    # Convert data to training format (query, document, label)
    # For cross-encoder training, we need individual (query, document, label) pairs
    training_samples = []

    for item in additional_data:
        query = item['query']
        documents = item['documents']
        labels = item['labels']

        # Create individual training samples
        for doc, label in zip(documents, labels):
            training_samples.append((query, doc, float(label)))

    logger.info(f"✅ Created {len(training_samples)} additional training samples")

    # Validate we have enough training data
    if len(training_samples) < 10:
        logger.error(f"❌ Not enough training samples: {len(training_samples)}. Need at least 10 samples.")
        return

    # 4. Setup retraining
    logger.info("Setting up retraining...")

    # Detect LoRA configuration from loaded model
    # Check if model has LoRA config saved
    use_lora = False
    lora_r = 8
    lora_alpha = 16

    config_path = Path(model_path) / "model_config.json"
    if config_path.exists():
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            lora_config = config.get('lora_config', {})
            use_lora = lora_config.get('use_lora', False)
            lora_r = lora_config.get('lora_r', 8)
            lora_alpha = lora_config.get('lora_alpha', 16)
        except Exception as e:
            logger.warning(f"Could not read LoRA config: {e}")

    # Fallback: check model attributes
    if not use_lora and hasattr(model, 'use_lora'):
        use_lora = model.use_lora

    if use_lora:
        logger.info("📍 Detected LoRA configuration in loaded model")
        logger.info(f"   LoRA rank: {lora_r}, alpha: {lora_alpha}")

    trainer = CrossEncoderTrainer(
        model_name=model.model_name,
        num_labels=model.num_labels,
        max_length=model.max_length,
        loss_function="bce",
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )

    # Use the loaded model (this preserves LoRA weights)
    trainer.model = model

    # Prepare model for training (freeze base, unfreeze LoRA if needed)
    if hasattr(trainer.model, 'freeze_base_model'):
        trainer.model.freeze_base_model()
        logger.info("✅ Model prepared for training (base frozen, LoRA unfrozen)")

    # Recreate optimizer with the loaded model parameters
    # (the original optimizer was configured with different model parameters)
    from torch.optim import AdamW
    try:
        from transformers import get_linear_schedule_with_warmup
    except ImportError:
        from torch.optim.lr_scheduler import get_linear_schedule_with_warmup

    # Get trainable parameters (this handles LoRA correctly)
    # For LoRA models, we need to ensure LoRA parameters are trainable
    trainable_params = []
    for name, param in trainer.model.model.named_parameters():
        # For LoRA models, parameters with 'lora' in the name should be trainable
        if 'lora' in name.lower():
            param.requires_grad_(True)
            trainable_params.append(param)
        elif param.requires_grad:
            trainable_params.append(param)

    # If no trainable parameters found, fall back to all parameters (shouldn't happen)
    if not trainable_params:
        logger.warning("⚠️ No trainable parameters found, using all parameters")
        trainable_params = list(trainer.model.model.parameters())

    # Recreate optimizer
    logger.info(f"📊 Found {len(trainable_params)} trainable parameters")
    trainer.optimizer = AdamW(trainable_params, lr=args.learning_rate)

    # Recreate scheduler (matching original trainer logic)
    # We need to simulate the DataLoader to get the correct step count
    # For simplicity, assume gradient_accumulation_steps = 1 (common default)
    gradient_accumulation_steps = 1
    steps_per_epoch = (len(training_samples) + args.batch_size - 1) // args.batch_size  # Ceiling division
    total_steps = steps_per_epoch * args.epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * 0.1)  # 10% warmup

    trainer.scheduler = get_linear_schedule_with_warmup(
        trainer.optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    logger.info("✅ Optimizer and scheduler recreated for loaded model")

    # 5. Prepare validation data
    logger.info("Preparing validation data...")
    val_samples = None
    if val_data and len(val_data) > 0:
        try:
            # Convert validation data to the same format as training
            val_samples = []
            for item in val_data:
                # Ensure item has required fields
                if not isinstance(item, dict) or 'query' not in item:
                    logger.warning(f"⚠️ Skipping invalid validation item: {item}")
                    continue

                query = item['query']
                documents = item.get('documents', [])
                labels = item.get('labels', [])

                if not documents or not labels or len(documents) != len(labels):
                    logger.warning(f"⚠️ Skipping validation item with mismatched documents/labels: {query}")
                    continue

                # Create individual validation samples
                for doc, label in zip(documents, labels):
                    val_samples.append((query, doc, float(label)))

            if val_samples:
                logger.info(f"✅ Created {len(val_samples)} validation samples")
            else:
                logger.warning("⚠️ No valid validation samples created")
                val_samples = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to create validation samples: {e}")
            logger.warning("Training without validation...")
            val_samples = None

    # 6. Retrain
    try:
        base_model_dir = Path(model_path).parent.parent
        model_name = Path(model_path).parent.name  # e.g., "trained_model"

        # Ensure base directory exists
        base_model_dir.mkdir(parents=True, exist_ok=True)

        output_dir = str(base_model_dir / (model_name + args.suffix))

        logger.info(f"📁 Model will be saved to: {output_dir}")
    except Exception as e:
        logger.error(f"❌ Failed to create output directory: {e}")
        return

    logger.info(f"Starting retraining for {args.epochs} epochs...")
    try:
        history = trainer.train(
            train_samples=training_samples,
            val_samples=val_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=output_dir,
            save_best_model=True,
            eval_steps=50,
            logging_steps=25
        )

        logger.info("\n✅ Retraining completed!")
        logger.info(f"📁 Retrained model saved to: {output_dir}")

        # Show final training metrics if available
        if 'train_loss' in history and history['train_loss']:
            final_loss = history['train_loss'][-1]
            logger.info(".4f")

        if 'val_loss' in history and history['val_loss']:
            final_val_loss = min(history['val_loss'])  # Best validation loss
            logger.info(".4f")

        logger.info("\n💡 Retrained model available for:")
        logger.info("• Evaluation: python cli/eval.py")
        logger.info("• Testing: python cli/test.py")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error("💡 Check your data format and model configuration")
        return


if __name__ == "__main__":
    main()
