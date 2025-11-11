"""
Cross-encoder trainer with support for various training strategies.
"""

import logging
import os
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import json

from ..models.cross_encoder import CrossEncoderModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RankingDataset(Dataset):
    """Dataset for ranking training."""

    def __init__(
        self,
        samples: List[Tuple[str, str, float]],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize dataset.

        Args:
            samples: List of (query, document, label) tuples
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query, document, label = self.samples[idx]

        # Tokenize
        encoded = self.tokenizer(
            query,
            document,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Squeeze batch dimension
        item = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

        # Add token_type_ids if present (e.g., for BERT)
        if 'token_type_ids' in encoded:
            item['token_type_ids'] = encoded['token_type_ids'].squeeze(0)

        return item


class CrossEncoderTrainer:
    """
    Trainer for cross-encoder ranking models.

    Supports:
    - Binary cross-entropy and margin ranking losses
    - Mixed precision training
    - LoRA fine-tuning
    - Hard negative mining
    - Evaluation during training
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 1,
        max_length: int = 512,
        loss_function: str = "bce",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        device: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            model_name: Pretrained model name
            num_labels: Number of labels
            max_length: Maximum sequence length
            loss_function: Loss function ('bce', 'mse', 'margin')
            use_lora: Whether to use LoRA
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            device: Device to use (None for auto-detect)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.loss_function = loss_function
        self.use_lora = use_lora

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Initialize model
        self.model = CrossEncoderModel(
            model_name=model_name,
            num_labels=num_labels,
            max_length=max_length,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        self.model.to(self.device)
        self.model.print_model_info()

        # Setup loss function
        self._setup_loss_function()

    def _setup_loss_function(self):
        """Setup loss function based on configuration."""
        if self.loss_function == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss_function == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_function == "margin":
            self.criterion = nn.MarginRankingLoss(margin=0.5)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")

        logger.info(f"Using loss function: {self.loss_function}")

    def train(
        self,
        train_samples: List[Tuple[str, str, float]],
        val_samples: Optional[List[Tuple[str, str, float]]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        mixed_precision: Optional[str] = None,
        output_dir: str = "./models/reranker",
        save_best_model: bool = True,
        eval_steps: int = 500,
        logging_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_samples: Training samples
            val_samples: Validation samples (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio for scheduler
            gradient_accumulation_steps: Gradient accumulation steps
            mixed_precision: Mixed precision ('fp16', 'bf16', or None)
            output_dir: Directory to save model
            save_best_model: Whether to save best model
            eval_steps: Evaluate every N steps
            logging_steps: Log every N steps

        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Training samples: {len(train_samples)}")
        if val_samples:
            logger.info(f"Validation samples: {len(val_samples)}")

        # Create datasets
        train_dataset = RankingDataset(
            train_samples,
            self.model.tokenizer,
            self.max_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Setup optimizer
        optimizer = AdamW(
            self.model.model.parameters(),
            lr=learning_rate
        )

        # Setup scheduler
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Setup mixed precision
        scaler = None
        if mixed_precision == "fp16":
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Using FP16 mixed precision")

        # Training loop
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        best_val_loss = float('inf')
        global_step = 0

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            self.model.model.train()
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training")

            for step, batch in enumerate(progress_bar):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                if mixed_precision == "fp16" and scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            token_type_ids=batch.get('token_type_ids'),
                            labels=batch['labels'].unsqueeze(1) if self.num_labels == 1 else batch['labels']
                        )
                        loss = outputs['loss']
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch.get('token_type_ids'),
                        labels=batch['labels'].unsqueeze(1) if self.num_labels == 1 else batch['labels']
                    )
                    loss = outputs['loss']

                # Normalize loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * gradient_accumulation_steps

                # Update weights
                if (step + 1) % gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % logging_steps == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        history['learning_rates'].append(current_lr)

                    # Evaluation
                    if val_samples and global_step % eval_steps == 0:
                        val_loss = self._evaluate(val_samples, batch_size)
                        history['val_loss'].append(val_loss)

                        logger.info(f"Step {global_step}: val_loss = {val_loss:.4f}")

                        # Save best model
                        if save_best_model and val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self._save_model(output_dir, "best")
                            logger.info(f"Saved best model (val_loss={val_loss:.4f})")

                        self.model.model.train()

                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

            # Epoch complete
            avg_epoch_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_epoch_loss)

            logger.info(f"Epoch {epoch + 1} completed. Avg loss: {avg_epoch_loss:.4f}")

            # Save checkpoint
            self._save_model(output_dir, f"epoch_{epoch + 1}")

        # Save final model
        self._save_model(output_dir, "final")
        logger.info(f"Training completed. Model saved to {output_dir}")

        # Save training history
        self._save_history(history, output_dir)

        return history

    def _evaluate(
        self,
        val_samples: List[Tuple[str, str, float]],
        batch_size: int
    ) -> float:
        """
        Evaluate on validation set.

        Args:
            val_samples: Validation samples
            batch_size: Batch size

        Returns:
            Average validation loss
        """
        self.model.model.eval()

        val_dataset = RankingDataset(
            val_samples,
            self.model.tokenizer,
            self.max_length
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch.get('token_type_ids'),
                    labels=batch['labels'].unsqueeze(1) if self.num_labels == 1 else batch['labels']
                )

                total_loss += outputs['loss'].item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def _save_model(self, output_dir: str, checkpoint_name: str):
        """
        Save model checkpoint.

        Args:
            output_dir: Output directory
            checkpoint_name: Checkpoint name
        """
        save_path = os.path.join(output_dir, checkpoint_name)
        os.makedirs(save_path, exist_ok=True)
        self.model.save(save_path)

    def _save_history(self, history: Dict[str, Any], output_dir: str):
        """
        Save training history.

        Args:
            history: Training history
            output_dir: Output directory
        """
        history_path = os.path.join(output_dir, "training_history.json")
        os.makedirs(output_dir, exist_ok=True)

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"Training history saved to {history_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        self.model = CrossEncoderModel.load(checkpoint_path)
        self.model.to(self.device)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
