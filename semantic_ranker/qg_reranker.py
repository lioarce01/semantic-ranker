"""
Query Graph Neural Reranker - Novel research implementation.

Combines cross-encoder relevance scoring with query graph neural networks
for cross-query knowledge transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from semantic_ranker.models.cross_encoder import CrossEncoderModel
from semantic_ranker.query_graph import QueryGraphBuilder
from semantic_ranker.query_gnn import QueryGNN, QueryGraphAttention


class QueryGraphReranker(nn.Module):
    """
    Query Graph Neural Reranker.

    Extends cross-encoder with query graph reasoning for improved generalization.
    """

    def __init__(
        self,
        cross_encoder: CrossEncoderModel,
        query_graph_builder: QueryGraphBuilder,
        gnn_hidden_dim: int = 256,
        gnn_output_dim: int = 128,
        gnn_dropout: float = 0.1,
        lambda_contrastive: float = 0.1,
        lambda_rank: float = 0.05,
        temperature: float = 0.07
    ):
        """
        Args:
            cross_encoder: Base cross-encoder model
            query_graph_builder: Query graph builder
            gnn_hidden_dim: GNN hidden dimension
            gnn_output_dim: GNN output dimension
            gnn_dropout: GNN dropout rate
            lambda_contrastive: Weight for contrastive loss
            lambda_rank: Weight for GNN ranking loss
            temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.cross_encoder = cross_encoder
        self.query_graph_builder = query_graph_builder

        # GNN for query embeddings
        query_emb_dim = 768  # all-mpnet-base-v2 dimension
        self.query_gnn = QueryGNN(
            input_dim=query_emb_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            dropout=gnn_dropout
        )

        # Attention for combining GNN and cross-encoder
        ce_hidden_dim = cross_encoder.model.config.hidden_size
        self.attention = QueryGraphAttention(
            query_dim=gnn_output_dim,
            cross_encoder_dim=ce_hidden_dim,
            hidden_dim=128
        )

        # Final prediction head
        self.predictor = nn.Linear(ce_hidden_dim, 1)

        # Loss weights
        self.lambda_contrastive = lambda_contrastive
        self.lambda_rank = lambda_rank
        self.temperature = temperature

        # Cache for query graph
        self.edge_index = None
        self.edge_weights = None
        self.query_embeddings = None
        self.gnn_query_embeddings = None

    def build_query_graph(
        self,
        queries: List[str],
        query_doc_relevance: Optional[Dict[int, List[int]]] = None
    ):
        """Build query graph from training queries."""
        edge_index, edge_weights, query_embeddings = self.query_graph_builder.build_graph(
            queries, query_doc_relevance
        )

        self.edge_index = edge_index.to(self.cross_encoder.model.device)
        self.edge_weights = edge_weights.to(self.cross_encoder.model.device)
        self.query_embeddings = query_embeddings.to(self.cross_encoder.model.device)

        # Run GNN
        with torch.no_grad():
            self.gnn_query_embeddings = self.query_gnn(
                self.query_embeddings,
                self.edge_index,
                self.edge_weights
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        query_indices: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with query graph reasoning.

        Args:
            input_ids: Tokenized query-doc pairs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            query_indices: Query index for each sample [batch_size]
            token_type_ids: Token type IDs [batch_size, seq_len]

        Returns:
            Dict with logits and intermediate outputs
        """
        # Get cross-encoder outputs
        ce_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_hidden_states': True
        }
        if token_type_ids is not None and hasattr(self.cross_encoder.model.config, 'type_vocab_size'):
            ce_inputs['token_type_ids'] = token_type_ids

        ce_outputs = self.cross_encoder.model(**ce_inputs)
        hidden_states = ce_outputs.hidden_states[-1]  # Last layer [batch_size, seq_len, hidden_dim]

        # Pool to get sequence representation (use CLS token)
        pooled = hidden_states[:, 0, :]  # [batch_size, hidden_dim]

        # Apply query graph attention if graph is available
        if self.gnn_query_embeddings is not None:
            attended = self.attention(
                self.gnn_query_embeddings,
                pooled,
                query_indices
            )
        else:
            attended = pooled

        # Final prediction
        logits = self.predictor(attended).squeeze(-1)  # [batch_size]

        return {
            'logits': logits,
            'pooled': pooled,
            'query_indices': query_indices,
            'gnn_embeddings': self.gnn_query_embeddings
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss: BCE + contrastive + ranking.

        Args:
            outputs: Forward pass outputs
            labels: Ground truth labels [batch_size]

        Returns:
            Total loss and loss components dict
        """
        logits = outputs['logits']
        pooled = outputs['pooled']
        query_indices = outputs['query_indices']
        gnn_embeddings = outputs['gnn_embeddings']

        # 1. BCE loss (main task)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)

        # 2. Contrastive loss in query space (if GNN available)
        contrastive_loss = torch.tensor(0.0, device=logits.device)
        if gnn_embeddings is not None and len(query_indices) > 1:
            contrastive_loss = self._compute_contrastive_loss(
                pooled, query_indices, labels
            )

        # 3. GNN ranking loss (if GNN available)
        rank_loss = torch.tensor(0.0, device=logits.device)
        if gnn_embeddings is not None:
            rank_loss = self._compute_rank_loss(gnn_embeddings, query_indices, labels)

        # Total loss
        total_loss = (
            bce_loss +
            self.lambda_contrastive * contrastive_loss +
            self.lambda_rank * rank_loss
        )

        loss_dict = {
            'loss': total_loss.item(),
            'bce_loss': bce_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'rank_loss': rank_loss.item()
        }

        return total_loss, loss_dict

    def _compute_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        query_indices: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss in query space.

        Queries with shared relevant docs should have similar embeddings.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive mask: same query or queries with shared relevant docs
        batch_size = embeddings.size(0)
        positive_mask = torch.zeros(batch_size, batch_size, device=embeddings.device)

        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and query_indices[i] == query_indices[j] and labels[i] == labels[j] == 1:
                    positive_mask[i, j] = 1

        # If no positives, return zero loss
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Mask out self-similarity
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        sim_matrix.masked_fill_(mask, -float('inf'))

        # Compute InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Average over positives
        loss = -(positive_mask * log_prob).sum() / (positive_mask.sum() + 1e-8)

        return loss

    def _compute_rank_loss(
        self,
        gnn_embeddings: torch.Tensor,
        query_indices: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Ranking loss using GNN embeddings.

        Encourages queries with more relevant docs to have higher-norm embeddings.
        """
        # Group by query and compute avg relevance
        unique_queries = torch.unique(query_indices)
        loss = torch.tensor(0.0, device=gnn_embeddings.device)

        if len(unique_queries) < 2:
            return loss

        query_scores = []
        query_relevance = []

        for q_idx in unique_queries:
            mask = query_indices == q_idx
            if mask.sum() == 0:
                continue

            # Average relevance for this query
            avg_rel = labels[mask].float().mean()
            query_relevance.append(avg_rel)

            # Query embedding norm (proxy for "quality")
            q_emb = gnn_embeddings[q_idx.item()]
            q_score = torch.norm(q_emb)
            query_scores.append(q_score)

        if len(query_scores) < 2:
            return loss

        query_scores = torch.stack(query_scores)
        query_relevance = torch.tensor(query_relevance, device=gnn_embeddings.device)

        # Pairwise ranking loss
        for i in range(len(query_scores)):
            for j in range(i + 1, len(query_scores)):
                if query_relevance[i] > query_relevance[j]:
                    margin_loss = F.relu(query_scores[j] - query_scores[i] + 0.1)
                    loss = loss + margin_loss
                elif query_relevance[j] > query_relevance[i]:
                    margin_loss = F.relu(query_scores[i] - query_scores[j] + 0.1)
                    loss = loss + margin_loss

        # Normalize by number of pairs
        num_pairs = len(query_scores) * (len(query_scores) - 1) / 2
        if num_pairs > 0:
            loss = loss / num_pairs

        return loss

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
            List of relevance scores
        """
        if len(queries) != len(documents):
            raise ValueError("Queries and documents must have same length")

        self.eval()
        scores = []

        # Get unique queries and build mapping
        unique_queries = list(set(queries))
        query_to_idx = {q: i for i, q in enumerate(unique_queries)}

        # Build temporary query graph if needed
        if self.gnn_query_embeddings is None or len(unique_queries) != len(self.query_embeddings):
            self.build_query_graph(unique_queries)

        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i+batch_size]
                batch_docs = documents[i:i+batch_size]

                # Tokenize
                encoded = self.cross_encoder.tokenizer(
                    batch_queries,
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=self.cross_encoder.max_length,
                    return_tensors="pt"
                )

                # Get query indices
                query_indices = torch.tensor(
                    [query_to_idx[q] for q in batch_queries],
                    dtype=torch.long,
                    device=self.cross_encoder.model.device
                )

                # Move to device
                batch = {k: v.to(self.cross_encoder.model.device) for k, v in encoded.items()}
                batch['query_indices'] = query_indices

                # Forward pass
                outputs = self.forward(**batch)
                batch_scores = torch.sigmoid(outputs['logits']).cpu().numpy()

                # Handle shape
                if batch_scores.ndim == 0:
                    batch_scores = [batch_scores]
                elif batch_scores.ndim > 1:
                    batch_scores = batch_scores.flatten()

                scores.extend(batch_scores.tolist())

        return scores

    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.cross_encoder.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
            self.edge_weights = self.edge_weights.to(device)
            self.query_embeddings = self.query_embeddings.to(device)
            if self.gnn_query_embeddings is not None:
                self.gnn_query_embeddings = self.gnn_query_embeddings.to(device)
        return self
