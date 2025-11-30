"""
Cross-encoder model for semantic reranking.
"""

import logging
import os
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossEncoderModel:
    """
    Cross-encoder model for document reranking.

    Supports:
    - Multiple pretrained models (BERT, RoBERTa, DeBERTa, etc.)
    - LoRA fine-tuning for memory efficiency
    - Binary classification for relevance scoring
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 1,
        max_length: int = 512,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None
    ):
        """
        Initialize cross-encoder model.

        Args:
            model_name: Pretrained model name or path
            num_labels: Number of labels (1 for binary classification)
            max_length: Maximum sequence length
            use_lora: Whether to use LoRA fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout rate
            lora_target_modules: Target modules for LoRA (None for auto-detect)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.use_lora = use_lora

        logger.info(f"Loading model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model configuration
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="regression" if num_labels == 1 else "single_label_classification"
        )

        # Load base model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )

        # Apply LoRA if requested
        if use_lora:
            logger.info("Applying LoRA fine-tuning")
            self._setup_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)

        # Store parameters for reference
        self.lora_config = {
            'use_lora': use_lora,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'lora_target_modules': lora_target_modules
        }

        # Cache model capabilities
        self._model_supports_token_type_ids = self._check_token_type_ids_support()

    def _setup_lora(
        self,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: Optional[List[str]]
    ):
        """Setup LoRA configuration."""
        if target_modules is None:
            # Auto-detect target modules based on model type
            model_type = self.model.config.model_type
            if model_type in ['bert', 'roberta', 'deberta', 'deberta-v2']:
                target_modules = ['query', 'key', 'value']
            elif model_type == 't5':
                target_modules = ['q', 'k', 'v', 'o']
            else:
                target_modules = ['query', 'key', 'value']  # Default

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="SEQ_CLS"
        )

        self.model = get_peft_model(self.model, lora_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional)
            labels: Labels for loss computation (optional)

        Returns:
            Dictionary with logits and optionally loss
        """
        # Prepare inputs - some models don't support token_type_ids
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        # Only add token_type_ids if the model supports it
        if token_type_ids is not None and self._supports_token_type_ids():
            model_inputs['token_type_ids'] = token_type_ids

        outputs = self.model(**model_inputs)

        result = {
            'logits': outputs.logits
        }

        if labels is not None:
            result['loss'] = outputs.loss

        return result

    def __call__(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Make model callable for training."""
        return self.forward(*args, **kwargs)

    def predict(
        self,
        queries: List[str],
        documents: List[str],
        batch_size: int = 32
    ) -> List[float]:
        """
        Predict relevance scores for query-document pairs.

        Args:
            queries: List of queries
            documents: List of documents
            batch_size: Batch size for inference

        Returns:
            List of relevance scores (higher = more relevant)
        """
        if len(queries) != len(documents):
            raise ValueError("Queries and documents must have same length")

        self.model.eval()
        scores = []

        # Process in batches
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            # Move to device
            batch = {k: v.to(self.model.device) for k, v in encoded.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**batch)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()

                # Ensure scores is 1D
                if batch_scores.ndim == 0:
                    batch_scores = [batch_scores]
                elif batch_scores.ndim > 1:
                    batch_scores = batch_scores.flatten()

                scores.extend(batch_scores.tolist())

        return scores

    def to(self, device: Union[str, torch.device]):
        """Move model to device."""
        self.model.to(device)
        return self

    def print_model_info(self):
        """Print model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Model: {self.model_name}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(".1f")
        logger.info(f"LoRA enabled: {self.use_lora}")

        if self.use_lora:
            logger.info(f"LoRA rank: {self.lora_config['lora_r']}")
            logger.info(f"LoRA alpha: {self.lora_config['lora_alpha']}")

    def save(self, save_path: str):
        """
        Save model and tokenizer.

        Args:
            save_path: Directory to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        if self.use_lora:
            self.model.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

        # Save configuration
        config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'lora_config': self.lora_config
        }

        import json
        with open(save_path / "model_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load(cls, load_path: str) -> 'CrossEncoderModel':
        """
        Load model from disk.

        Args:
            load_path: Path to saved model

        Returns:
            Loaded CrossEncoderModel instance
        """
        load_path = Path(load_path)

        # Load configuration
        config_path = load_path / "model_config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Fallback for models saved without config
            config = {
                'model_name': str(load_path),
                'num_labels': 1,
                'max_length': 512,
                'lora_config': {'use_lora': False}
            }

        # Check if it's a LoRA model
        is_lora = config['lora_config']['use_lora']

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(load_path)

        # Load model
        if is_lora:
            # Load base model first
            base_model_name = config.get('model_name', 'bert-base-uncased')
            base_config = AutoConfig.from_pretrained(
                base_model_name,
                num_labels=config.get('num_labels', 1),
                problem_type="regression" if config.get('num_labels', 1) == 1 else "single_label_classification"
            )

            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                config=base_config
            )

            # Load LoRA weights
            model = PeftModel.from_pretrained(base_model, load_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(load_path)

        # Create instance
        instance = cls.__new__(cls)
        instance.model_name = config.get('model_name', 'loaded_model')
        instance.num_labels = config.get('num_labels', 1)
        instance.max_length = config.get('max_length', 512)
        instance.use_lora = is_lora
        instance.tokenizer = tokenizer
        instance.model = model
        instance.lora_config = config.get('lora_config', {})

        # Initialize cached model capabilities (normally done in __init__)
        instance._model_supports_token_type_ids = instance._check_token_type_ids_support()

        logger.info(f"Model loaded from {load_path}")
        return instance

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def freeze_base_model(self):
        """Freeze base model parameters (useful for LoRA training)."""
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA parameters if using LoRA
        if self.use_lora:
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True

    def _check_token_type_ids_support(self) -> bool:
        """Check if the model supports token_type_ids."""
        # Models that don't support token_type_ids
        unsupported_models = [
            'distilbert',
            'roberta',
            'deberta-v2',
            'albert',
            'electra',
            'camembert',
            'xlm-roberta'
        ]

        model_type = getattr(self.model.config, 'model_type', '').lower()

        # Check if model type is in unsupported list
        if model_type in unsupported_models:
            return False

        # Check if the model has token_type_embeddings
        if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'token_type_embeddings'):
            return self.model.embeddings.token_type_embeddings is not None

        # Default to True for BERT-like models
        return True

    def _supports_token_type_ids(self) -> bool:
        """Return cached result of token_type_ids support check."""
        return self._model_supports_token_type_ids
