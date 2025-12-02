"""
CLI utilities package.

Provides shared functionality for all CLI scripts including:
- Common utilities (path setup, logging)
- Model utilities (discovery, LoRA detection)
- Data utilities (dataset loading, format conversion)
- Config utilities (configuration management)
"""

from .common import setup_project_path, setup_logging
from .model_utils import find_best_model, detect_lora_config, freeze_base_unfreeze_lora
from .data_utils import get_available_datasets, load_dataset_unified, convert_to_training_samples
from .config_utils import add_config_args, load_config_with_overrides, save_config_with_model

__all__ = [
    # Common utilities
    'setup_project_path',
    'setup_logging',
    # Model utilities
    'find_best_model',
    'detect_lora_config',
    'freeze_base_unfreeze_lora',
    # Data utilities
    'get_available_datasets',
    'load_dataset_unified',
    'convert_to_training_samples',
    # Config utilities
    'add_config_args',
    'load_config_with_overrides',
    'save_config_with_model',
]
