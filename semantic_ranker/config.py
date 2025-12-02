"""
Centralized configuration management for semantic reranker.

This module provides dataclass-based configuration with YAML support,
enabling reproducible experiments and easy configuration management.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal
from pathlib import Path
import yaml
import json


@dataclass
class ModelConfig:
    """Model-related configuration"""
    model_name: str = "bert-base-uncased"
    max_length: int = 256
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    loss_function: Literal["bce", "mse", "margin_ranking"] = "bce"
    eval_steps: int = 100
    logging_steps: int = 50
    save_strategy: Literal["best", "epoch", "steps"] = "best"
    gradient_accumulation_steps: int = 1
    fp16: bool = False


@dataclass
class DataConfig:
    """Data-related configuration"""
    dataset: str = "msmarco"
    max_samples: Optional[int] = None
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    negative_sampling: Literal["random", "hard", "mixed"] = "random"
    num_negatives: int = 1


@dataclass
class QuantumConfig:
    """Quantum fine-tuning specific parameters"""
    quantum_mode: bool = False
    resonance_threshold: float = 0.7
    entanglement_weight: float = 0.3
    quantum_phase: str = "superposition"
    knowledge_preservation_weight: float = 0.5
    resonance_penalty_scale: float = 0.01
    entanglement_loss_scale: float = 0.01


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: List[str] = field(default_factory=lambda: ["ndcg@10", "mrr@10", "map@10"])
    batch_size: int = 32
    num_samples: Optional[int] = None


@dataclass
class GNNConfig:
    """Query Graph Neural Network configuration"""
    gnn_mode: bool = False
    embedding_model: str = "all-mpnet-base-v2"
    similarity_threshold: float = 0.7
    max_neighbors: int = 10
    gnn_hidden_dim: int = 256
    gnn_output_dim: int = 128
    gnn_dropout: float = 0.1
    lambda_contrastive: float = 0.1
    lambda_rank: float = 0.05
    temperature: float = 0.07


@dataclass
class Config:
    """Complete configuration for semantic reranker"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file

        Args:
            path: Path to YAML configuration file

        Returns:
            Config object
        """
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create Config from dictionary

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            Config object
        """
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            quantum=QuantumConfig(**config_dict.get('quantum', {})),
            gnn=GNNConfig(**config_dict.get('gnn', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {}))
        )

    def to_yaml(self, path: str):
        """Save configuration to YAML file

        Args:
            path: Path to save YAML configuration
        """
        config_dict = asdict(self)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary

        Returns:
            Dictionary representation of config
        """
        return asdict(self)

    def to_json(self, path: str):
        """Save configuration to JSON file

        Args:
            path: Path to save JSON configuration
        """
        config_dict = asdict(self)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """Load configuration from JSON file

        Args:
            path: Path to JSON configuration file

        Returns:
            Config object
        """
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
