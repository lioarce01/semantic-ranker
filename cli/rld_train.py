#!/usr/bin/env python3
"""
Train with Resonant Listwise Distillation (RLD)

Novel reranking approach combining:
- Set Transformer for listwise context
- Knowledge Distillation from strong teacher
- Adaptive temperature scaling
- Multi-phase training (warmup → distillation → fine-tuning)

Target: NDCG@10 > 0.85 (8-15% improvement over Quantum)
"""

import argparse
import sys
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Import shared utilities
from cli.utils import (
    setup_project_path,
    setup_logging,
    load_config_with_overrides,
    save_config_with_model
)

# Setup project imports
setup_project_path()

# Setup logging
logger = setup_logging()

# Import RLD modules
from semantic_ranker.models.rld_model import RLDReranker
from semantic_ranker.data.listwise_loader import create_listwise_dataloader, split_dataset
from semantic_ranker.training.losses import (
    KnowledgeDistillationLoss,
    AdaptiveTemperatureLoss,
    ListwiseRankingLoss
)
from semantic_ranker.evaluation.metrics import compute_ndcg, compute_mrr, compute_map


class RLDTrainer:
    """
    Multi-phase trainer for RLD.

    Phases:
    1. Pointwise warmup (Set Transformer disabled)
    2. Listwise distillation (full model + KD)
    3. Fine-tuning (lower learning rate)
    """

    def __init__(
        self,
        model: RLDReranker,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        output_dir: str,
        device: str = 'cpu'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device

        self.model.to(device)

        # Setup loss function
        self._setup_loss()

        # Best metrics tracking
        self.best_ndcg = 0.0
        self.global_step = 0

    def _setup_loss(self):
        """Setup loss function based on config"""
        loss_type = self.config.training.loss_function

        if loss_type == 'knowledge_distillation' or loss_type == 'kd':
            self.criterion = KnowledgeDistillationLoss(
                temperature=self.config.training.temperature,
                lambda_kd=self.config.training.lambda_kd,
                lambda_bce=self.config.training.lambda_bce
            )
        elif loss_type == 'adaptive_kd':
            self.criterion = AdaptiveTemperatureLoss(
                lambda_kd=self.config.training.lambda_kd,
                lambda_bce=self.config.training.lambda_bce
            )
        elif loss_type == 'listwise':
            self.criterion = ListwiseRankingLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")

        self.criterion.to(self.device)
        logger.info(f"Using loss function: {loss_type}")

    def train_phase(
        self,
        phase_num: int,
        num_epochs: int,
        learning_rate: float,
        enable_set_transformer: bool = True
    ):
        """
        Train a single phase.

        Args:
            phase_num: Phase number (1, 2, or 3)
            num_epochs: Number of epochs for this phase
            learning_rate: Learning rate for this phase
            enable_set_transformer: Whether to use Set Transformer
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Phase {phase_num}: {num_epochs} epochs, LR={learning_rate}")
        logger.info(f"Set Transformer: {'enabled' if enable_set_transformer else 'disabled'}")
        logger.info(f"{'='*70}\n")

        # Toggle Set Transformer
        if hasattr(self.model, 'use_set_transformer'):
            original_setting = self.model.use_set_transformer
            self.model.use_set_transformer = enable_set_transformer

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.training.weight_decay
        )

        # Learning rate scheduler
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(total_steps * self.config.training.warmup_ratio)

        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            self.model.train()

            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(self.train_loader, desc=f"Phase {phase_num} - Epoch {epoch+1}")

            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                labels = batch['labels'].to(self.device)
                masks = batch['masks'].to(self.device)

                # Forward pass (listwise)
                queries = batch['queries']
                documents_list = batch['documents_list']

                # Process each query in batch
                batch_losses = []

                for i in range(len(queries)):
                    query = queries[i]
                    documents = documents_list[i]
                    query_labels = labels[i]
                    query_mask = masks[i]

                    # Get predictions
                    if enable_set_transformer:
                        # Truncate to valid documents (no padding needed for Set Transformer)
                        num_docs = query_mask.sum().item()
                        valid_documents = documents[:num_docs]
                        logits = self.model.forward_listwise(query, valid_documents, None)
                    else:
                        # Pairwise mode for warmup
                        queries_repeated = [query] * len(documents)
                        logits = self.model(queries_repeated, documents)

                    # Mask padding
                    num_docs = query_mask.sum().item()
                    logits = logits[:num_docs]
                    query_labels = query_labels[:num_docs]

                    # Compute loss based on type
                    if isinstance(self.criterion, (KnowledgeDistillationLoss, AdaptiveTemperatureLoss)):
                        # KD loss requires teacher scores
                        if 'teacher_scores' in batch:
                            teacher_scores = batch['teacher_scores'][i][:num_docs].to(self.device)
                            loss, metrics = self.criterion(logits, teacher_scores, query_labels)
                        else:
                            # Fallback to BCE if no teacher scores available
                            loss = nn.functional.binary_cross_entropy_with_logits(logits, query_labels)
                            logger.warning("No teacher scores found, using BCE loss")
                    elif isinstance(self.criterion, ListwiseRankingLoss):
                        # Listwise loss
                        loss = self.criterion(logits.unsqueeze(0), query_labels.unsqueeze(0))
                    else:
                        # Generic BCE fallback
                        loss = nn.functional.binary_cross_entropy_with_logits(logits, query_labels)

                    batch_losses.append(loss)

                # Average losses in batch
                total_loss = torch.stack(batch_losses).mean()

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                optimizer.step()
                scheduler.step()

                epoch_loss += total_loss.item()
                num_batches += 1
                self.global_step += 1

                # Update progress bar
                progress_bar.set_postfix({'loss': f'{total_loss.item():.4f}'})

                # Logging
                if self.global_step % self.config.training.logging_steps == 0:
                    logger.info(f"  Step {self.global_step} | Loss: {total_loss.item():.4f}")

                # Evaluation
                if self.global_step % self.config.training.eval_steps == 0:
                    metrics = self.evaluate()
                    logger.info(f"\n  Validation at step {self.global_step}...")
                    logger.info(f"    NDCG@10: {metrics['ndcg@10']:.4f} | MRR@10: {metrics['mrr@10']:.4f} | MAP@10: {metrics['map@10']:.4f}")

                    # Save best model
                    if metrics['ndcg@10'] > self.best_ndcg:
                        self.best_ndcg = metrics['ndcg@10']
                        logger.info(f"    New best NDCG@10: {self.best_ndcg:.4f}")
                        self.save_model('best')

                    self.model.train()

            avg_loss = epoch_loss / num_batches
            logger.info(f"  Average epoch loss: {avg_loss:.4f}")

        # Restore original setting
        if hasattr(self.model, 'use_set_transformer'):
            self.model.use_set_transformer = original_setting

    def evaluate(self) -> dict:
        """Evaluate on validation set"""
        self.model.eval()

        all_ndcg = []
        all_mrr = []
        all_map = []

        with torch.no_grad():
            for batch in self.val_loader:
                queries = batch['queries']
                documents_list = batch['documents_list']
                labels = batch['labels'].cpu().numpy()
                masks = batch['masks'].cpu().numpy()

                for i in range(len(queries)):
                    query = queries[i]
                    documents = documents_list[i]
                    query_labels = labels[i]
                    query_mask = masks[i]

                    num_docs = int(query_mask.sum())

                    # Get predictions
                    logits = self.model.forward_listwise(query, documents[:num_docs])
                    scores = logits.cpu().numpy()

                    # Compute metrics
                    query_labels = query_labels[:num_docs]

                    # Rank documents by predicted scores
                    ranked_indices = np.argsort(scores)[::-1]
                    ranked_labels = [query_labels[i] for i in ranked_indices]

                    ndcg = compute_ndcg(ranked_labels, k=10)
                    mrr = compute_mrr(ranked_labels, k=10)
                    map_score = compute_map(ranked_labels, k=10)

                    all_ndcg.append(ndcg)
                    all_mrr.append(mrr)
                    all_map.append(map_score)

        return {
            'ndcg@10': np.mean(all_ndcg),
            'mrr@10': np.mean(all_mrr),
            'map@10': np.mean(all_map)
        }

    def save_model(self, name: str = 'final'):
        """Save model checkpoint"""
        save_path = self.output_dir / name
        self.model.save_pretrained(str(save_path))
        logger.info(f"  Model saved to {save_path}")

    def train(self):
        """Execute full multi-phase training"""
        logger.info("\nStarting RLD multi-phase training...")

        # Phase 1: Pointwise warmup (Set Transformer OFF)
        if self.config.training.phase_1_epochs > 0:
            self.train_phase(
                phase_num=1,
                num_epochs=self.config.training.phase_1_epochs,
                learning_rate=self.config.training.learning_rate,
                enable_set_transformer=False
            )

        # Phase 2: Listwise distillation (Set Transformer ON)
        if self.config.training.phase_2_epochs > 0:
            self.train_phase(
                phase_num=2,
                num_epochs=self.config.training.phase_2_epochs,
                learning_rate=self.config.training.learning_rate,
                enable_set_transformer=True
            )

        # Phase 3: Fine-tuning (lower LR)
        if self.config.training.phase_3_epochs > 0:
            self.train_phase(
                phase_num=3,
                num_epochs=self.config.training.phase_3_epochs,
                learning_rate=self.config.training.phase_3_lr,
                enable_set_transformer=True
            )

        # Save final model
        self.save_model('final')

        logger.info(f"\nTraining complete! Best NDCG@10: {self.best_ndcg:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train with Resonant Listwise Distillation')

    parser.add_argument(
        '--config',
        required=True,
        help='Path to RLD config YAML'
    )
    parser.add_argument(
        '--output',
        '--output-dir',
        dest='output_dir',
        required=True,
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        from types import SimpleNamespace
        config_dict = yaml.safe_load(f)

        def dict_to_namespace(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            return d

        config = dict_to_namespace(config_dict)

    logger.info("=" * 70)
    logger.info("RESONANT LISTWISE DISTILLATION (RLD) TRAINING")
    logger.info("=" * 70)
    logger.info(f"Dataset: {config.data.dataset}")
    logger.info(f"Model: {config.model.model_name}")
    logger.info(f"Set Transformer: {config.model.use_set_transformer}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Device: {args.device}")

    try:
        # 1. Create model
        logger.info("\nStep 1: Creating RLD model...")
        model = RLDReranker(
            model_name=config.model.model_name,
            use_lora=config.model.use_lora,
            lora_r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            use_set_transformer=config.model.use_set_transformer,
            set_transformer_layers=config.model.set_transformer_layers,
            set_transformer_heads=config.model.set_transformer_heads,
            use_mvli=config.model.use_mvli,
            max_length=config.model.max_length
        )

        logger.info(f"Total parameters: {model.get_total_parameters():,}")
        logger.info(f"Trainable parameters: {model.get_trainable_parameters():,}")

        # 2. Load data with proper split
        logger.info(f"\nStep 2: Loading dataset: {config.data.dataset}...")
        dataset_path = f"datasets/{config.data.dataset}.json"

        # Split dataset into train/val/test
        from semantic_ranker.data.listwise_loader import split_dataset
        import tempfile
        import json

        logger.info(f"Splitting dataset (train: {config.data.train_split}, val: {config.data.val_split}, test: {config.data.test_split})...")
        train_data, val_data, test_data = split_dataset(
            data_path=dataset_path,
            train_ratio=config.data.train_split,
            val_ratio=config.data.val_split,
            test_ratio=config.data.test_split
        )

        # Save splits to temp files
        temp_dir = Path(tempfile.gettempdir()) / "rld_splits"
        temp_dir.mkdir(exist_ok=True)

        train_path = temp_dir / "train.json"
        val_path = temp_dir / "val.json"

        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f)
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f)

        # Create data loaders from splits
        train_loader = create_listwise_dataloader(
            data_path=str(train_path),
            batch_size=config.training.batch_size,
            max_queries=config.data.max_queries,
            max_docs_per_query=config.data.max_docs_per_query,
            teacher_scores_path=config.data.teacher_scores_path,
            shuffle=True
        )

        val_loader = create_listwise_dataloader(
            data_path=str(val_path),
            batch_size=config.evaluation.batch_size,
            max_queries=None,
            max_docs_per_query=config.data.max_docs_per_query,
            teacher_scores_path=config.data.teacher_scores_path,
            shuffle=False
        )

        logger.info(f"Train queries: {len(train_loader.dataset)}")
        logger.info(f"Val queries: {len(val_loader.dataset)}")
        logger.info(f"Train samples: {len(train_data)}")
        logger.info(f"Val samples: {len(val_data)}")

        # 3. Create trainer
        logger.info("\nStep 3: Creating RLD trainer...")
        trainer = RLDTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=args.output_dir,
            device=args.device
        )

        # 4. Train
        logger.info("\nStep 4: Starting multi-phase training...")
        trainer.train()

        # Save config
        save_config_with_model(config, args.output_dir)

        logger.info("\nRLD training completed successfully!")

    except Exception as e:
        logger.error(f"RLD training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
