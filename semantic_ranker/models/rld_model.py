"""
Resonant Listwise Distillation (RLD) Model

Main model architecture combining:
- Cross-encoder with LoRA
- Set Transformer for listwise ranking
- Adaptive temperature scoring
- Optional MVLI (ColBERT-style token matching)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional, Dict, List, Tuple
import numpy as np

from .set_transformer import SetTransformer


class RLDReranker(nn.Module):
    """
    Resonant Listwise Distillation Reranker.

    Architecture:
    1. Cross-encoder encodes query-document pairs
    2. Set Transformer enables listwise context
    3. Scoring head with learnable temperature
    4. Optional MVLI for token-level matching

    Args:
        model_name: Base transformer model
        use_lora: Whether to use LoRA fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout
        use_set_transformer: Enable listwise Set Transformer
        set_transformer_layers: Number of Set Transformer blocks
        set_transformer_heads: Number of attention heads
        use_mvli: Enable token-level ColBERT matching (optional)
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        use_set_transformer: bool = True,
        set_transformer_layers: int = 2,
        set_transformer_heads: int = 8,
        use_mvli: bool = False,
        max_length: int = 256
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.use_set_transformer = use_set_transformer
        self.use_mvli = use_mvli

        # Load tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # Apply LoRA if enabled
        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "key", "value"],  # BERT attention layers
                inference_mode=False
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            print(f"Applied LoRA: r={lora_r}, alpha={lora_alpha}")
            self.encoder.print_trainable_parameters()

        # Set Transformer for listwise ranking
        if use_set_transformer:
            self.set_transformer = SetTransformer(
                d_model=self.hidden_size,
                num_heads=set_transformer_heads,
                num_layers=set_transformer_layers,
                dropout=0.1
            )
        else:
            self.set_transformer = None

        # Scoring head
        self.scoring_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        # Learnable temperature for adaptive scaling
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Optional MVLI (ColBERT-style token matching)
        if use_mvli:
            self.mvli_projection = nn.Linear(self.hidden_size, 128)
        else:
            self.mvli_projection = None

    def encode_single(
        self,
        query: str,
        document: str,
        return_tokens: bool = False
    ) -> torch.Tensor:
        """
        Encode a single query-document pair.

        Args:
            query: Query text
            document: Document text or pre-computed embedding
            return_tokens: Return token embeddings for MVLI

        Returns:
            [CLS] representation or (cls, token_embeddings) if return_tokens=True
        """
        # Check if document is a pre-computed embedding
        if isinstance(document, np.ndarray) or isinstance(document, torch.Tensor):
            # Use compressed embedding directly
            # TODO: Implement embedding injection into cross-encoder
            raise NotImplementedError("PECC embedding injection not yet implemented")

        # Tokenize query + document
        inputs = self.tokenizer(
            query,
            document,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move to same device as model
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}

        # Encode
        outputs = self.encoder(**inputs)

        # Get [CLS] representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]

        if return_tokens and self.use_mvli:
            token_embeddings = outputs.last_hidden_state  # [1, seq_len, hidden_size]
            return cls_embedding, token_embeddings

        return cls_embedding

    def encode_batch(
        self,
        queries: List[str],
        documents: List[str],
        return_tokens: bool = False
    ) -> torch.Tensor:
        """
        Encode a batch of query-document pairs.

        Args:
            queries: List of query texts
            documents: List of document texts
            return_tokens: Return token embeddings for MVLI

        Returns:
            [CLS] representations [batch_size, hidden_size]
        """
        # Tokenize batch
        inputs = self.tokenizer(
            queries,
            documents,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}

        # Encode
        outputs = self.encoder(**inputs)

        # Get [CLS] representations
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        if return_tokens and self.use_mvli:
            token_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            return cls_embeddings, token_embeddings

        return cls_embeddings

    def forward_listwise(
        self,
        query: str,
        documents: List[str],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for listwise ranking (single query, multiple documents).

        Args:
            query: Query text
            documents: List of document texts
            mask: Optional mask for padded documents

        Returns:
            Scores [num_docs]
        """
        # Encode all query-document pairs
        queries_repeated = [query] * len(documents)
        doc_embeddings = self.encode_batch(queries_repeated, documents)  # [num_docs, hidden_size]

        # Apply Set Transformer if enabled
        if self.set_transformer is not None:
            # Normalize mask shape to match attention expectations
            if mask is not None:
                mask = mask.to(doc_embeddings.device)
                # Ensure batch dimension exists
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                # Trim mask to the number of documents (handles any lingering padding)
                mask = mask[:, : doc_embeddings.size(0)]

            # Add batch dimension
            doc_embeddings = doc_embeddings.unsqueeze(0)  # [1, num_docs, hidden_size]

            # Apply listwise context
            contextualized = self.set_transformer(doc_embeddings, mask)  # [1, num_docs, hidden_size]

            # Remove batch dimension
            doc_embeddings = contextualized.squeeze(0)  # [num_docs, hidden_size]

        # Score each document
        logits = self.scoring_head(doc_embeddings).squeeze(-1)  # [num_docs]

        # Apply temperature scaling
        logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)

        return logits

    def forward_batch_listwise(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        masks: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """
        Forward pass for batched listwise ranking.

        Args:
            queries: List of queries
            documents_list: List of document lists (one per query)
            masks: Optional masks for padded documents

        Returns:
            List of score tensors (one per query)
        """
        all_scores = []

        for i, (query, documents) in enumerate(zip(queries, documents_list)):
            mask = masks[i] if masks is not None else None
            scores = self.forward_listwise(query, documents, mask)
            all_scores.append(scores)

        return all_scores

    def forward(
        self,
        queries: List[str],
        documents: List[str]
    ) -> torch.Tensor:
        """
        Standard forward pass for training (pairwise).

        Args:
            queries: List of queries
            documents: List of documents

        Returns:
            Scores [batch_size]
        """
        # Encode pairs
        embeddings = self.encode_batch(queries, documents)  # [batch_size, hidden_size]

        # Score
        logits = self.scoring_head(embeddings).squeeze(-1)  # [batch_size]

        # Temperature scaling
        logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)

        return logits

    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save encoder (handles LoRA automatically)
        self.encoder.save_pretrained(save_directory)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

        # Save other components
        torch.save({
            'scoring_head': self.scoring_head.state_dict(),
            'set_transformer': self.set_transformer.state_dict() if self.set_transformer else None,
            'temperature': self.temperature.data,
            'mvli_projection': self.mvli_projection.state_dict() if self.mvli_projection else None,
            'config': {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'use_set_transformer': self.use_set_transformer,
                'use_mvli': self.use_mvli,
                'hidden_size': self.hidden_size
            }
        }, os.path.join(save_directory, 'rld_components.pt'))

        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model from directory"""
        import os

        # Load components
        components = torch.load(os.path.join(load_directory, 'rld_components.pt'))
        config = components['config']

        # Create model
        model = cls(
            model_name=config['model_name'],
            max_length=config['max_length'],
            use_set_transformer=config['use_set_transformer'],
            use_mvli=config['use_mvli']
        )

        # Load encoder (handles LoRA automatically)
        from transformers import AutoModel
        model.encoder = AutoModel.from_pretrained(load_directory)

        # Load other components
        model.scoring_head.load_state_dict(components['scoring_head'])
        if model.set_transformer and components['set_transformer']:
            model.set_transformer.load_state_dict(components['set_transformer'])
        model.temperature.data = components['temperature']
        if model.mvli_projection and components['mvli_projection']:
            model.mvli_projection.load_state_dict(components['mvli_projection'])

        print(f"Model loaded from {load_directory}")
        return model

    def get_trainable_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
