"""
Configuration utilities for CLI scripts.

Provides functions to add config arguments to argparse parsers
and load configurations with CLI override support.
"""

import argparse
from pathlib import Path
from typing import Optional
from semantic_ranker.config import Config


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add config-related arguments to parser

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        Modified parser with config arguments
    """
    config_group = parser.add_argument_group('configuration')

    config_group.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML config file'
    )

    config_group.add_argument(
        '--config-profile',
        type=str,
        choices=['default', 'quick_test', 'full_training', 'lora_training', 'quantum_training', 'retrain'],
        help='Predefined configuration profile from configs/ directory'
    )

    return parser


def load_config_with_overrides(args: argparse.Namespace) -> Config:
    """Load config with CLI overrides following priority order:
    1. CLI arguments (highest priority)
    2. Config file (--config)
    3. Config profile (--config-profile)
    4. Defaults (lowest priority)

    Args:
        args: Parsed command-line arguments

    Returns:
        Config object with all overrides applied
    """
    # Start with defaults
    config = Config()

    # Load from profile if specified
    if hasattr(args, 'config_profile') and args.config_profile:
        profile_path = Path('configs') / f'{args.config_profile}.yaml'
        if profile_path.exists():
            config = Config.from_yaml(str(profile_path))
        else:
            print(f"Warning: Profile '{args.config_profile}' not found at {profile_path}")

    # Load from file if specified (overrides profile)
    if hasattr(args, 'config') and args.config:
        config = Config.from_yaml(args.config)

    # Apply CLI overrides (highest priority)
    _apply_cli_overrides(config, args)

    return config


def _apply_cli_overrides(config: Config, args: argparse.Namespace):
    """Apply CLI argument overrides to config object

    Args:
        config: Config object to modify
        args: Parsed command-line arguments
    """
    # Model overrides
    # Note: --model-name is used for output naming, not model selection
    # The base model is set in the config file (e.g., bert-base-uncased)
    if hasattr(args, 'use_lora') and args.use_lora is not None:
        config.model.use_lora = args.use_lora
    if hasattr(args, 'lora_r') and args.lora_r:
        config.model.lora_r = args.lora_r
    if hasattr(args, 'lora_alpha') and args.lora_alpha:
        config.model.lora_alpha = args.lora_alpha

    # Training overrides
    if hasattr(args, 'epochs') and args.epochs:
        config.training.epochs = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size:
        config.training.batch_size = args.batch_size
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if hasattr(args, 'weight_decay') and args.weight_decay:
        config.training.weight_decay = args.weight_decay
    if hasattr(args, 'warmup_ratio') and args.warmup_ratio:
        config.training.warmup_ratio = args.warmup_ratio
    if hasattr(args, 'loss_function') and args.loss_function:
        config.training.loss_function = args.loss_function
    if hasattr(args, 'fp16') and args.fp16 is not None:
        config.training.fp16 = args.fp16

    # Data overrides
    if hasattr(args, 'dataset') and args.dataset:
        config.data.dataset = args.dataset
    if hasattr(args, 'max_samples') and args.max_samples:
        config.data.max_samples = args.max_samples
    if hasattr(args, 'negative_sampling') and args.negative_sampling:
        config.data.negative_sampling = args.negative_sampling
    if hasattr(args, 'num_negatives') and args.num_negatives:
        config.data.num_negatives = args.num_negatives

    # Quantum overrides
    if hasattr(args, 'quantum_mode') and args.quantum_mode is not None:
        config.quantum.quantum_mode = args.quantum_mode
    if hasattr(args, 'resonance_threshold') and args.resonance_threshold:
        config.quantum.resonance_threshold = args.resonance_threshold
    if hasattr(args, 'entanglement_weight') and args.entanglement_weight:
        config.quantum.entanglement_weight = args.entanglement_weight
    if hasattr(args, 'quantum_phase') and args.quantum_phase:
        config.quantum.quantum_phase = args.quantum_phase

    # Evaluation overrides
    if hasattr(args, 'eval_batch_size') and args.eval_batch_size:
        config.evaluation.batch_size = args.eval_batch_size
    if hasattr(args, 'num_samples') and args.num_samples:
        config.evaluation.num_samples = args.num_samples


def save_config_with_model(config: Config, model_path: str):
    """Save configuration alongside trained model for reproducibility

    Args:
        config: Configuration to save
        model_path: Path to model directory
    """
    model_dir = Path(model_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save as both YAML and JSON for maximum compatibility
    config.to_yaml(str(model_dir / "config.yaml"))
    config.to_json(str(model_dir / "config.json"))
