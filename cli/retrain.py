#!/usr/bin/env python3
"""
Retrain the best model with additional data.
Responsibility: Only retraining, finds best model automatically.
"""

import argparse
from pathlib import Path

# Import shared utilities
from cli.utils import (
    setup_project_path,
    setup_logging,
    find_best_model,
    detect_lora_config,
    get_available_datasets,
    load_dataset_unified,
    convert_to_training_samples,
    add_config_args,
    load_config_with_overrides,
    save_config_with_model
)

# Setup project imports
setup_project_path()

# Setup logging
logger = setup_logging()

# Now import semantic_ranker modules
from semantic_ranker.data import DataPreprocessor
from semantic_ranker.training import CrossEncoderTrainer
from semantic_ranker.models import CrossEncoderModel


def main():
    parser = argparse.ArgumentParser(description='Retrain the best model with additional data')

    # Add config support
    parser = add_config_args(parser)

    # Add retraining-specific arguments
    parser.add_argument('--dataset', choices=get_available_datasets(),
                       help='Additional dataset for retraining')
    parser.add_argument('--epochs', type=int, help='Number of additional epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size for retraining')
    parser.add_argument('--learning-rate', type=float, help='Learning rate for fine-tuning')
    parser.add_argument('--samples', type=int, help='Number of additional samples to use')
    parser.add_argument('--suffix', default='_retrained',
                       help='Suffix for retrained model directory')
    parser.add_argument('--quantum-mode', action='store_true',
                       help='Enable quantum resonance fine-tuning')

    args = parser.parse_args()

    # Load configuration (use 'retrain' profile defaults)
    config = load_config_with_overrides(args)

    # Use config values with retrain defaults
    dataset = args.dataset or config.data.dataset
    epochs = args.epochs or 2  # Default 2 epochs for retraining
    batch_size = args.batch_size or 8  # Smaller batch for retraining
    learning_rate = args.learning_rate or 1e-5  # Lower LR for retraining
    samples = args.samples or 500
    quantum_mode = args.quantum_mode or config.quantum.quantum_mode

    logger.info("=== Model Retraining ===")
    logger.info(f"Additional dataset: {dataset}")
    logger.info(f"Additional epochs: {epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Quantum mode: {quantum_mode}")
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

    # 2. Load additional data using shared utility
    logger.info(f"Loading additional data from {dataset}...")
    train_data, val_data, _ = load_dataset_unified(dataset, samples + 200)  # Extra for validation

    logger.info(f"✅ Loaded {len(train_data)} additional training samples")
    logger.info(f"✅ Loaded {len(val_data)} validation samples")

    # 3. Convert to training format using shared utility
    logger.info("Preparing data for retraining...")
    training_samples = convert_to_training_samples(train_data)
    logger.info(f"✅ Created {len(training_samples)} training triples")

    # Validate we have enough training data
    if len(training_samples) < 10:
        logger.error(f"❌ Not enough training samples: {len(training_samples)}. Need at least 10 samples.")
        return

    # 4. Setup retraining
    logger.info("Setting up retraining...")

    # Detect LoRA configuration using shared utility
    use_lora, lora_r, lora_alpha, lora_dropout = detect_lora_config(model_path)

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
