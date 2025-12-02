#!/usr/bin/env python3
"""
Train a cross-encoder reranker with Query Graph Neural Reranking.

QG-Rerank: Novel approach using GNN over query graphs for cross-query learning.
"""

import argparse
import random
import sys
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict

from cli.utils import (
    setup_project_path,
    setup_logging,
    get_available_datasets,
    load_dataset_unified,
    add_config_args,
    load_config_with_overrides,
    save_config_with_model
)

setup_project_path()
logger = setup_logging()

from semantic_ranker.data import MSMARCODataLoader, CustomDataLoader, DataPreprocessor
from semantic_ranker.models import CrossEncoderModel
from semantic_ranker.query_graph import QueryGraphBuilder
from semantic_ranker.qg_reranker import QueryGraphReranker
from semantic_ranker.training import CrossEncoderTrainer


class QGTrainer:
    """Trainer for Query Graph Neural Reranker."""

    def __init__(self, model, config, train_data, val_data):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.relevant_queries = None  # Will be set during training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

        # Scheduler will be initialized in train() method after data filtering

    def train(self):
        """Train the model."""
        logger.info("Starting Query Graph Neural Reranking training")

        # Build query graph from training data
        logger.info("Building query graph from training data...")
        train_queries = [sample['query'] for sample in self.train_data]
        unique_queries = list(set(train_queries))

        # Limit queries for memory efficiency (query graphs scale O(n²))
        max_queries_for_graph = min(200, len(unique_queries))  # Limit to 200 queries max
        if len(unique_queries) > max_queries_for_graph:
            logger.warning(f"Limiting query graph to {max_queries_for_graph} queries (from {len(unique_queries)}) for memory efficiency")
            # Sample diverse queries instead of just taking the first ones
            import random
            unique_queries = random.sample(unique_queries, max_queries_for_graph)

        # Create query-doc relevance mapping for graph construction
        query_doc_relevance = defaultdict(list)
        query_to_idx = {q: i for i, q in enumerate(unique_queries)}

        # Only use training samples that match our selected queries
        relevant_samples = [sample for sample in self.train_data if sample['query'] in unique_queries]

        for sample in relevant_samples:
            query_idx = query_to_idx[sample['query']]
            if sample['label'] == 1:
                query_doc_relevance[query_idx].append(hash(sample['document']) % 10000)

        self.model.build_query_graph(unique_queries, dict(query_doc_relevance))
        logger.info(f"Query graph built with {len(unique_queries)} nodes")

        # Store relevant queries for consistent indexing during training
        self.relevant_queries = unique_queries

        # Filter training and validation data to only use queries in the graph
        filtered_train_data = [sample for sample in self.train_data if sample['query'] in unique_queries]
        filtered_val_data = [sample for sample in self.val_data if sample['query'] in unique_queries]

        logger.info(f"Filtered training data: {len(filtered_train_data)} samples (from {len(self.train_data)})")
        logger.info(f"Filtered validation data: {len(filtered_val_data)} samples (from {len(self.val_data)})")

        # Recalculate warmup steps based on filtered data
        total_steps = (len(filtered_train_data) // self.config.training.batch_size) * self.config.training.epochs
        warmup_steps = int(total_steps * self.config.training.warmup_ratio)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )

        best_val_ndcg = 0.0
        global_step = 0

        for epoch in range(self.config.training.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.training.epochs}")
            self.model.train()

            epoch_loss = 0.0
            epoch_bce = 0.0
            epoch_contrastive = 0.0
            epoch_rank = 0.0

            random.shuffle(filtered_train_data)

            for step in range(0, len(filtered_train_data), self.config.training.batch_size):
                batch = filtered_train_data[step:step + self.config.training.batch_size]

                queries = [s['query'] for s in batch]
                documents = [s['document'] for s in batch]
                labels = torch.tensor([s['label'] for s in batch], dtype=torch.float, device=self.device)
                query_indices = torch.tensor([query_to_idx[q] for q in queries], dtype=torch.long, device=self.device)

                # Tokenize
                encoded = self.model.cross_encoder.tokenizer(
                    queries,
                    documents,
                    padding=True,
                    truncation=True,
                    max_length=self.config.model.max_length,
                    return_tensors='pt'
                )

                batch_inputs = {k: v.to(self.device) for k, v in encoded.items()}
                batch_inputs['query_indices'] = query_indices

                # Forward pass
                outputs = self.model(**batch_inputs)

                # Compute loss
                loss, loss_dict = self.model.compute_loss(outputs, labels)

                # Backward pass
                loss.backward()

                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss += loss_dict['loss']
                epoch_bce += loss_dict['bce_loss']
                epoch_contrastive += loss_dict['contrastive_loss']
                epoch_rank += loss_dict['rank_loss']

                global_step += 1

                if global_step % self.config.training.logging_steps == 0:
                    avg_loss = epoch_loss / (step // self.config.training.batch_size + 1)
                    logger.info(
                        f"Step {global_step} | Loss: {avg_loss:.4f} | "
                        f"BCE: {epoch_bce/(step//self.config.training.batch_size+1):.4f} | "
                        f"Contrastive: {epoch_contrastive/(step//self.config.training.batch_size+1):.4f} | "
                        f"Rank: {epoch_rank/(step//self.config.training.batch_size+1):.4f}"
                    )

                if global_step % self.config.training.eval_steps == 0:
                    val_metrics = self.evaluate(filtered_val_data, query_to_idx)
                    logger.info(f"Validation | NDCG@10: {val_metrics['ndcg@10']:.4f} | MRR@10: {val_metrics['mrr@10']:.4f}")

                    if val_metrics['ndcg@10'] > best_val_ndcg:
                        best_val_ndcg = val_metrics['ndcg@10']
                        logger.info(f"New best validation NDCG@10: {best_val_ndcg:.4f}")

                    self.model.train()

            avg_epoch_loss = epoch_loss / (len(filtered_train_data) // self.config.training.batch_size)
            logger.info(f"Epoch {epoch + 1} completed | Avg Loss: {avg_epoch_loss:.4f}")

        logger.info(f"Training completed | Best NDCG@10: {best_val_ndcg:.4f}")
        logger.info(f"Used {len(unique_queries)} queries for graph construction out of {len(set([sample['query'] for sample in self.train_data]))} total unique queries")

    def evaluate(self, data, query_to_idx):
        """Evaluate model on validation data."""
        self.model.eval()

        predictions = []
        labels_list = []

        with torch.no_grad():
            for step in range(0, len(data), self.config.evaluation.batch_size):
                batch = data[step:step + self.config.evaluation.batch_size]

                queries = [s['query'] for s in batch]
                documents = [s['document'] for s in batch]
                labels = [s['label'] for s in batch]
                query_indices = torch.tensor([query_to_idx.get(q, 0) for q in queries], dtype=torch.long, device=self.device)

                encoded = self.model.cross_encoder.tokenizer(
                    queries,
                    documents,
                    padding=True,
                    truncation=True,
                    max_length=self.config.model.max_length,
                    return_tensors='pt'
                )

                batch_inputs = {k: v.to(self.device) for k, v in encoded.items()}
                batch_inputs['query_indices'] = query_indices

                outputs = self.model(**batch_inputs)
                scores = torch.sigmoid(outputs['logits']).cpu().numpy()

                predictions.extend(scores.tolist())
                labels_list.extend(labels)

        # Compute metrics
        from semantic_ranker.evaluation import RankingMetrics
        metrics_calculator = RankingMetrics()

        # Group by query for ranking metrics
        query_predictions = defaultdict(list)
        query_labels = defaultdict(list)

        for i, sample in enumerate(data):
            query = sample['query']
            query_predictions[query].append(predictions[i])
            query_labels[query].append(labels_list[i])

        ndcg_scores = []
        mrr_scores = []

        for query in query_predictions:
            preds = np.array(query_predictions[query])
            labs = np.array(query_labels[query])

            ndcg = metrics_calculator.ndcg_at_k(labs, preds, k=10)
            mrr = metrics_calculator.mrr_at_k(labs, preds, k=10)

            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)

        return {
            'ndcg@10': np.mean(ndcg_scores),
            'mrr@10': np.mean(mrr_scores)
        }


def main():
    parser = argparse.ArgumentParser(description='Train Query Graph Neural Reranker')
    add_config_args(parser)
    parser.add_argument('--experiment-name', type=str, help='Experiment/model save name')
    args = parser.parse_args()

    # Load config
    config = load_config_with_overrides(args)

    if not config.gnn.gnn_mode:
        logger.error("GNN mode must be enabled in config. Set gnn.gnn_mode: true")
        sys.exit(1)

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    logger.info(f"Loading dataset: {config.data.dataset}")
    train_data, val_data, test_data = load_dataset_unified(
        config.data.dataset,
        config.data.max_samples,
        config.data.train_split,
        config.data.val_split,
        config.data.test_split
    )

    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # Initialize base cross-encoder
    logger.info("Initializing base cross-encoder...")
    cross_encoder = CrossEncoderModel(
        model_name=config.model.model_name,
        max_length=config.model.max_length,
        use_lora=config.model.use_lora,
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout
    )

    # Initialize query graph builder
    logger.info("Initializing query graph builder...")
    query_graph_builder = QueryGraphBuilder(
        embedding_model=config.gnn.embedding_model,
        similarity_threshold=config.gnn.similarity_threshold,
        max_neighbors=config.gnn.max_neighbors
    )

    # Initialize QG-Reranker
    logger.info("Initializing Query Graph Reranker...")
    model = QueryGraphReranker(
        cross_encoder=cross_encoder,
        query_graph_builder=query_graph_builder,
        gnn_hidden_dim=config.gnn.gnn_hidden_dim,
        gnn_output_dim=config.gnn.gnn_output_dim,
        gnn_dropout=config.gnn.gnn_dropout,
        lambda_contrastive=config.gnn.lambda_contrastive,
        lambda_rank=config.gnn.lambda_rank,
        temperature=config.gnn.temperature
    )

    # Train
    trainer = QGTrainer(model, config, train_data, val_data)
    trainer.train()

    # Save model
    experiment_name = args.experiment_name or f"qg_rerank_{config.data.dataset}"
    output_dir = Path('models') / experiment_name / 'best'
    output_dir = Path('models') / model_name / 'best'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {output_dir}")
    save_config_with_model(config, output_dir.parent)
    torch.save(model.state_dict(), output_dir / 'model.pt')
    save_config_with_model(config, output_dir.parent)

    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
