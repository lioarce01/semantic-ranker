"""
Model utilities for CLI scripts.

Provides functions for model discovery, LoRA configuration detection,
and parameter management.
"""

from pathlib import Path
import json
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def find_best_model(models_dir: str = "./models") -> Optional[str]:
    """Find the most recent best model in models directory

    Searches for trained models in the models directory and returns
    the path to the most recently modified 'best' model.

    Args:
        models_dir: Path to models directory

    Returns:
        Path to best model or None if not found
    """
    models_path = Path(models_dir)

    if not models_path.exists():
        logger.error(f"❌ Models directory not found: {models_dir}")
        return None

    best_models = []
    for model_dir in models_path.iterdir():
        if not model_dir.is_dir():
            continue

        best_path = model_dir / "best"
        if not best_path.exists():
            continue

        # Check for model files (safetensors or pytorch_model.bin or LoRA adapters)
        has_model = (
            (best_path / "model.safetensors").exists() or
            (best_path / "pytorch_model.bin").exists() or
            (best_path / "adapter_model.safetensors").exists()
        )

        if has_model:
            mtime = best_path.stat().st_mtime
            best_models.append((str(best_path), mtime, model_dir.name))

    if not best_models:
        logger.error("❌ No trained models found in models directory")
        return None

    # Sort by modification time (most recent first)
    best_models.sort(key=lambda x: x[1], reverse=True)
    best_path, _, model_name = best_models[0]

    logger.info(f"📍 Found best model: {model_name}")
    return best_path


def detect_lora_config(model_path: str) -> Tuple[bool, int, int, float]:
    """Detect LoRA configuration from saved model

    Reads model_config.json to determine if the model uses LoRA
    and what the LoRA hyperparameters are.

    Args:
        model_path: Path to model directory

    Returns:
        Tuple of (use_lora, lora_r, lora_alpha, lora_dropout)
    """
    config_path = Path(model_path) / "model_config.json"

    # Defaults
    use_lora = False
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.1

    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            lora_config = config.get('lora_config', {})
            use_lora = lora_config.get('use_lora', False)
            lora_r = lora_config.get('lora_r', 8)
            lora_alpha = lora_config.get('lora_alpha', 16)
            lora_dropout = lora_config.get('lora_dropout', 0.1)
            logger.info(f"✓ Loaded LoRA config: r={lora_r}, alpha={lora_alpha}")
        except Exception as e:
            logger.warning(f"Could not read LoRA config: {e}")

    return use_lora, lora_r, lora_alpha, lora_dropout


def freeze_base_unfreeze_lora(model):
    """Freeze base model parameters, unfreeze LoRA adapters

    Used during retraining to only update LoRA parameters
    while keeping base model frozen.

    Args:
        model: PyTorch model with LoRA adapters
    """
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
