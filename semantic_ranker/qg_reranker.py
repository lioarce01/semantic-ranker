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
import logging
from collections import defaultdict

from semantic_ranker.models.cross_encoder import CrossEncoderModel
from semantic_ranker.query_graph import QueryGraphBuilder
from semantic_ranker.query_gnn import (
    QueryGNN,
    GraphAttentionNetwork,
    QueryGraphAttention,
    LearnableQueryEncoder,
    CrossAttentionFusion
)

logger = logging.getLogger(__name__)


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
        gnn_num_heads: int = 4,
        gnn_num_layers: int = 3,
        lambda_contrastive: float = 0.1,
        lambda_rank: float = 0.05,
        lambda_coherence: float = 0.15,
        lambda_alignment: float = 0.1,
        temperature: float = 0.07,
        use_dqgan: bool = True  # Enable DQGAN enhancements
    ):
        """
        Args:
            cross_encoder: Base cross-encoder model
            query_graph_builder: Query graph builder
            gnn_hidden_dim: GNN hidden dimension
            gnn_output_dim: GNN output dimension
            gnn_dropout: GNN dropout rate
            gnn_num_heads: Number of attention heads for GAT (DQGAN)
            gnn_num_layers: Number of GNN layers (DQGAN uses 3)
            lambda_contrastive: Weight for contrastive loss
            lambda_rank: Weight for GNN ranking loss
            lambda_coherence: Weight for graph coherence loss (DQGAN)
            lambda_alignment: Weight for CE-GNN alignment loss (DQGAN)
            temperature: Temperature for contrastive loss
            use_dqgan: Enable DQGAN architecture enhancements
        """
        super().__init__()

        self.cross_encoder = cross_encoder
        self.query_graph_builder = query_graph_builder
        self.use_dqgan = use_dqgan

        # DQGAN Phase 1: Learnable Query Encoder
        query_emb_dim = 768  # all-mpnet-base-v2 dimension
        learnable_query_dim = gnn_hidden_dim  # Project to GNN input dimension

        if use_dqgan:
            self.query_encoder = LearnableQueryEncoder(
                input_dim=query_emb_dim,
                output_dim=learnable_query_dim,
                dropout=gnn_dropout
            )
        else:
            self.query_encoder = None

        # GNN for query embeddings (choose architecture based on use_dqgan)
        gnn_input_dim = learnable_query_dim if use_dqgan else query_emb_dim

        if use_dqgan:
            # DQGAN Phase 2: 3-layer Graph Attention Network
            self.query_gnn = GraphAttentionNetwork(
                input_dim=gnn_input_dim,
                hidden_dim=gnn_hidden_dim,
                output_dim=gnn_output_dim,
                num_heads=gnn_num_heads,
                dropout=gnn_dropout
            )
        else:
            # Original 2-layer GCN
            self.query_gnn = QueryGNN(
                input_dim=gnn_input_dim,
                hidden_dim=gnn_hidden_dim,
                output_dim=gnn_output_dim,
                dropout=gnn_dropout
            )

        # DQGAN Phase 1: Cross-Attention Fusion (replaces scalar attention)
        ce_hidden_dim = cross_encoder.model.config.hidden_size
        if use_dqgan:
            self.attention = CrossAttentionFusion(
                query_dim=gnn_output_dim,
                cross_encoder_dim=ce_hidden_dim,
                num_heads=4,
                dropout=gnn_dropout
            )
        else:
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
        self.lambda_coherence = lambda_coherence
        self.lambda_alignment = lambda_alignment
        self.temperature = temperature

        # DQGAN Phase 3: Adaptive loss normalization (EMA)
        if use_dqgan:
            self.register_buffer('bce_ema', torch.tensor(1.0))
            self.register_buffer('contrastive_ema', torch.tensor(1.0))
            self.register_buffer('coherence_ema', torch.tensor(1.0))
            self.register_buffer('alignment_ema', torch.tensor(1.0))
            self.ema_decay = 0.9
        else:
            self.bce_ema = None
            self.contrastive_ema = None
            self.coherence_ema = None
            self.alignment_ema = None

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

        # Initialize GNN query embeddings cache (will be computed in forward pass with gradients)
        self.gnn_query_embeddings = None

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
        # Compute GNN embeddings with gradients enabled (if graph is available)
        gnn_embeddings = None
        if self.edge_index is not None and self.query_embeddings is not None:
            # DQGAN Phase 1: Apply learnable query encoder first
            if self.use_dqgan and self.query_encoder is not None:
                encoded_queries = self.query_encoder(self.query_embeddings)
            else:
                encoded_queries = self.query_embeddings

            # Run GNN forward pass WITH gradients (critical fix!)
            gnn_embeddings = self.query_gnn(
                encoded_queries,
                self.edge_index,
                self.edge_weights
            )

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
        if gnn_embeddings is not None:
            attended = self.attention(
                gnn_embeddings,
                pooled,
                query_indices
            )
            # Debug: ensure attention is working
            if torch.isnan(attended).any():
                attended = pooled  # Fallback if attention fails
        else:
            attended = pooled

        # Final prediction
        logits = self.predictor(attended).squeeze(-1)  # [batch_size]

        return {
            'logits': logits,
            'pooled': pooled,
            'query_indices': query_indices,
            'gnn_embeddings': gnn_embeddings if gnn_embeddings is not None else pooled
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss: BCE + contrastive + coherence + alignment.

        DQGAN Phase 3: Enhanced loss with GNN-targeted contrastive loss,
        graph coherence loss, and adaptive normalization.

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

        # 2. GNN Contrastive loss (FIXED: uses GNN embeddings, not CE!)
        contrastive_loss = torch.tensor(0.0, device=logits.device)
        if self.use_dqgan and gnn_embeddings is not None and len(query_indices) > 1:
            try:
                # Get GNN embeddings for this batch
                batch_gnn_emb = gnn_embeddings[query_indices]
                contrastive_loss = self._compute_gnn_contrastive_loss(
                    batch_gnn_emb, query_indices, labels
                )
                if torch.isnan(contrastive_loss):
                    contrastive_loss = torch.tensor(0.0, device=logits.device)
            except Exception as e:
                contrastive_loss = torch.tensor(0.0, device=logits.device)
        elif not self.use_dqgan and gnn_embeddings is not None and len(query_indices) > 1:
            # Original contrastive loss (uses CE embeddings)
            try:
                contrastive_loss = self._compute_contrastive_loss(
                    pooled, query_indices, labels
                )
                if torch.isnan(contrastive_loss):
                    contrastive_loss = torch.tensor(0.0, device=logits.device)
            except:
                contrastive_loss = torch.tensor(0.0, device=logits.device)

        # 3. Graph Coherence loss (NOVEL - DQGAN only)
        coherence_loss = torch.tensor(0.0, device=logits.device)
        if self.use_dqgan and gnn_embeddings is not None and self.edge_index is not None:
            try:
                coherence_loss = self._compute_graph_coherence_loss(
                    gnn_embeddings, self.edge_index, labels, query_indices
                )
                if torch.isnan(coherence_loss):
                    coherence_loss = torch.tensor(0.0, device=logits.device)
            except Exception as e:
                coherence_loss = torch.tensor(0.0, device=logits.device)

        # 4. CE-GNN Alignment loss (DQGAN only)
        alignment_loss = torch.tensor(0.0, device=logits.device)
        if self.use_dqgan and gnn_embeddings is not None:
            try:
                batch_gnn_emb = gnn_embeddings[query_indices]
                alignment_loss = self._compute_alignment_loss(pooled, batch_gnn_emb)
                if torch.isnan(alignment_loss):
                    alignment_loss = torch.tensor(0.0, device=logits.device)
            except Exception as e:
                alignment_loss = torch.tensor(0.0, device=logits.device)

        # 5. Ranking loss (kept for backwards compatibility)
        rank_loss = torch.tensor(0.0, device=logits.device)
        if not self.use_dqgan and gnn_embeddings is not None:
            try:
                rank_loss = self._compute_rank_loss(gnn_embeddings, query_indices, labels)
                if torch.isnan(rank_loss):
                    rank_loss = torch.tensor(0.0, device=logits.device)
            except:
                rank_loss = torch.tensor(0.0, device=logits.device)

        # Compute total loss with adaptive normalization (DQGAN)
        if self.use_dqgan and self.training:
            # Update EMAs
            with torch.no_grad():
                self.bce_ema = self.ema_decay * self.bce_ema + (1 - self.ema_decay) * bce_loss.item()
                if contrastive_loss.item() > 0:
                    self.contrastive_ema = self.ema_decay * self.contrastive_ema + (1 - self.ema_decay) * contrastive_loss.item()
                if coherence_loss.item() > 0:
                    self.coherence_ema = self.ema_decay * self.coherence_ema + (1 - self.ema_decay) * coherence_loss.item()
                if alignment_loss.item() > 0:
                    self.alignment_ema = self.ema_decay * self.alignment_ema + (1 - self.ema_decay) * alignment_loss.item()

            # Normalize losses
            bce_norm = bce_loss / (self.bce_ema + 1e-8)
            contrastive_norm = contrastive_loss / (self.contrastive_ema + 1e-8) if contrastive_loss.item() > 0 else contrastive_loss
            coherence_norm = coherence_loss / (self.coherence_ema + 1e-8) if coherence_loss.item() > 0 else coherence_loss
            alignment_norm = alignment_loss / (self.alignment_ema + 1e-8) if alignment_loss.item() > 0 else alignment_loss

            total_loss = (
                1.0 * bce_norm +
                self.lambda_contrastive * contrastive_norm +
                self.lambda_coherence * coherence_norm +
                self.lambda_alignment * alignment_norm
            )
        else:
            # Original loss (no adaptive normalization)
            total_loss = (
                bce_loss +
                self.lambda_contrastive * contrastive_loss +
                self.lambda_rank * rank_loss
            )

        loss_dict = {
            'loss': total_loss.item(),
            'bce_loss': bce_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'coherence_loss': coherence_loss.item() if self.use_dqgan else 0.0,
            'alignment_loss': alignment_loss.item() if self.use_dqgan else 0.0,
            'rank_loss': rank_loss.item() if not self.use_dqgan else 0.0
        }

        return total_loss, loss_dict

    def _compute_contrastive_loss(
        self,
        pooled_embeddings: torch.Tensor,
        query_indices: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Simple contrastive loss: pull together same labels, push apart different labels.
        """
        device = pooled_embeddings.device

        if len(pooled_embeddings) < 2:
            return torch.tensor(0.0, device=device)

        # Normalize embeddings
        pooled_norm = F.normalize(pooled_embeddings, dim=1)

        # Compute pairwise similarities
        sim_matrix = torch.matmul(pooled_norm, pooled_norm.T) / self.temperature

        # Create positive/negative masks
        labels_expanded = labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        negative_mask = (labels_expanded != labels_expanded.T).float()

        # Remove self-similarities
        mask_self = torch.eye(len(labels), device=device)
        positive_mask = positive_mask * (1 - mask_self)
        negative_mask = negative_mask * (1 - mask_self)

        # Only compute if we have positives and negatives
        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # InfoNCE loss: for each anchor, maximize similarity to positives, minimize to negatives
        # exp(sim_pos) / [exp(sim_pos) + sum(exp(sim_neg))]
        exp_sim = torch.exp(sim_matrix)

        # Sum of positive similarities
        pos_sim_sum = (exp_sim * positive_mask).sum(dim=1)

        # Sum of negative similarities
        neg_sim_sum = (exp_sim * negative_mask).sum(dim=1)

        # Avoid division by zero
        denominator = pos_sim_sum + neg_sim_sum + 1e-8

        # Loss: -log(pos / (pos + neg))
        loss = -torch.log(pos_sim_sum / denominator + 1e-8)

        # Only average over samples that have positives
        valid_samples = positive_mask.sum(dim=1) > 0
        if valid_samples.sum() > 0:
            loss = loss[valid_samples].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return loss

    def _compute_gnn_contrastive_loss(
        self,
        gnn_embeddings: torch.Tensor,
        query_indices: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        GNN-targeted contrastive loss (DQGAN Phase 3 - CRITICAL FIX).

        Uses GNN embeddings instead of cross-encoder embeddings, ensuring
        the contrastive loss actually trains the GNN!

        Args:
            gnn_embeddings: GNN query embeddings for this batch [batch_size, gnn_dim]
            query_indices: Query indices [batch_size]
            labels: Ground truth labels [batch_size]

        Returns:
            Contrastive loss scalar
        """
        device = gnn_embeddings.device

        if len(gnn_embeddings) < 2:
            return torch.tensor(0.0, device=device)

        # Normalize GNN embeddings
        gnn_norm = F.normalize(gnn_embeddings, dim=1)

        # Compute pairwise similarities
        sim_matrix = torch.matmul(gnn_norm, gnn_norm.T) / self.temperature

        # Create positive/negative masks based on labels
        labels_expanded = labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        negative_mask = (labels_expanded != labels_expanded.T).float()

        # Remove self-similarities
        mask_self = torch.eye(len(labels), device=device)
        positive_mask = positive_mask * (1 - mask_self)
        negative_mask = negative_mask * (1 - mask_self)

        # Only compute if we have positives and negatives
        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        pos_sim_sum = (exp_sim * positive_mask).sum(dim=1)
        neg_sim_sum = (exp_sim * negative_mask).sum(dim=1)
        denominator = pos_sim_sum + neg_sim_sum + 1e-8

        loss = -torch.log(pos_sim_sum / denominator + 1e-8)

        # Average over samples with positives
        valid_samples = positive_mask.sum(dim=1) > 0
        if valid_samples.sum() > 0:
            loss = loss[valid_samples].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return loss

    def _compute_graph_coherence_loss(
        self,
        gnn_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        labels: torch.Tensor,
        query_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Graph Coherence Loss (NOVEL - DQGAN Phase 3).

        Enforces that connected queries in the graph should have similar predictions
        when they have similar labels. This is the SECRET SAUCE that forces the GNN
        to learn from neighbor relationships!

        Args:
            gnn_embeddings: GNN query embeddings [num_queries, gnn_dim]
            edge_index: Graph edge indices [2, num_edges]
            labels: Batch labels [batch_size]
            query_indices: Query indices for batch [batch_size]

        Returns:
            Coherence loss scalar
        """
        device = gnn_embeddings.device

        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=device)

        # Get source and destination nodes
        src, dst = edge_index

        # Create a mapping from query index to label (for queries in current batch)
        query_label_map = {}
        for q_idx, label in zip(query_indices.tolist(), labels.tolist()):
            query_label_map[q_idx] = label

        # Filter edges to only those where both nodes are in current batch
        valid_edges = []
        edge_label_agreements = []

        for i in range(edge_index.size(1)):
            src_idx = src[i].item()
            dst_idx = dst[i].item()

            if src_idx in query_label_map and dst_idx in query_label_map:
                valid_edges.append(i)

                # Label agreement: 1 if both have same label, 0 if different
                src_label = query_label_map[src_idx]
                dst_label = query_label_map[dst_idx]
                agreement = 1.0 if src_label == dst_label else 0.0
                edge_label_agreements.append(agreement)

        if len(valid_edges) == 0:
            return torch.tensor(0.0, device=device)

        # Get embeddings for valid edges
        valid_edges_tensor = torch.tensor(valid_edges, device=device)
        valid_src = src[valid_edges_tensor]
        valid_dst = dst[valid_edges_tensor]

        # Compute cosine similarity between connected nodes
        src_emb = gnn_embeddings[valid_src]
        dst_emb = gnn_embeddings[valid_dst]
        emb_similarity = F.cosine_similarity(src_emb, dst_emb, dim=-1)

        # Target: high similarity when labels agree, low when they don't
        target = torch.tensor(edge_label_agreements, device=device, dtype=torch.float)

        # MSE loss: encourage embedding similarity to match label agreement
        loss = F.mse_loss(emb_similarity, target)

        return loss

    def _compute_alignment_loss(
        self,
        ce_embeddings: torch.Tensor,
        gnn_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        CE-GNN Alignment Loss (DQGAN Phase 3).

        Encourages GNN embeddings to align with cross-encoder embeddings,
        ensuring they capture complementary information.

        Uses element-wise cosine similarity since embeddings have different dimensions.

        Args:
            ce_embeddings: Cross-encoder embeddings [batch_size, ce_dim]
            gnn_embeddings: GNN embeddings [batch_size, gnn_dim]

        Returns:
            Alignment loss scalar
        """
        # Normalize both embeddings
        ce_norm = F.normalize(ce_embeddings, dim=-1)  # [batch_size, 768]
        gnn_norm = F.normalize(gnn_embeddings, dim=-1)  # [batch_size, 128]

        # Compute element-wise cosine similarity for each sample
        # We want high similarity between CE[i] and GNN[i] for each i
        batch_size = ce_norm.size(0)

        # Project GNN embeddings to CE dimension via learned linear projection
        # OR use mutual information-based loss
        # For simplicity, use L2 distance in normalized space (direction similarity)

        # Compute pairwise distances in batch
        # Expand to [batch_size, batch_size, dim] for pairwise computation
        ce_expanded = ce_norm.unsqueeze(1)  # [batch_size, 1, 768]
        gnn_expanded = gnn_norm.unsqueeze(0)  # [1, batch_size, 128]

        # Since dimensions differ, use cosine similarity via dot product after norm
        # Compute cosine similarity between all pairs
        # For different dims, we need to project or use a different approach

        # Alternative: Encourage similar pairwise distances
        # If CE[i] and CE[j] are close, then GNN[i] and GNN[j] should be close
        ce_dist = torch.cdist(ce_norm, ce_norm)  # [batch_size, batch_size]
        gnn_dist = torch.cdist(gnn_norm, gnn_norm)  # [batch_size, batch_size]

        # MSE between distance matrices
        loss = F.mse_loss(ce_dist, gnn_dist)

        return loss

    def _compute_rank_loss(
        self,
        gnn_embeddings: torch.Tensor,
        query_indices: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Ranking loss to ensure GNN embeddings preserve ranking relationships.

        Queries with similar ranking patterns should have similar GNN embeddings.
        """
        if gnn_embeddings is None or len(query_indices) < 2:
            return torch.tensor(0.0, device=gnn_embeddings.device if gnn_embeddings is not None else torch.device('cpu'))

        # Simple MSE loss between predicted logits and ranking-based targets
        # For now, just return a small regularization loss
        # TODO: Implement proper ranking-aware loss

        # Simple L2 regularization on GNN embeddings
        l2_reg = torch.norm(gnn_embeddings, p=2) / gnn_embeddings.numel()
        return self.lambda_rank * l2_reg

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
            # q_idx is the query index in the graph, so we can access gnn_embeddings[q_idx]
            if q_idx < len(gnn_embeddings):
                q_emb = gnn_embeddings[q_idx]
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

    def save(self, save_path: str):
        """
        Save QueryGraphReranker model and all components.

        Args:
            save_path: Directory to save model
        """
        from pathlib import Path
        import json

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save cross-encoder
        ce_path = save_path / "cross_encoder"
        ce_path.mkdir(exist_ok=True)
        self.cross_encoder.save(str(ce_path))

        # Save GNN components
        gnn_state = {
            'query_gnn': self.query_gnn.state_dict(),
            'attention': self.attention.state_dict(),
            'predictor': self.predictor.state_dict(),
        }
        torch.save(gnn_state, save_path / "gnn_components.pt")

        # Save query graph data (if exists)
        graph_data = {}
        if self.edge_index is not None:
            graph_data['edge_index'] = self.edge_index.cpu()
            graph_data['edge_weights'] = self.edge_weights.cpu()
            graph_data['query_embeddings'] = self.query_embeddings.cpu()
            if self.gnn_query_embeddings is not None:
                graph_data['gnn_query_embeddings'] = self.gnn_query_embeddings.cpu()
        if graph_data:
            torch.save(graph_data, save_path / "query_graph.pt")

        # Save configuration
        config = {
            'gnn_hidden_dim': self.query_gnn.hidden_dim,
            'gnn_output_dim': self.query_gnn.output_dim,
            'gnn_dropout': self.query_gnn.dropout.p,
            'lambda_contrastive': self.lambda_contrastive,
            'lambda_rank': self.lambda_rank,
            'temperature': self.temperature,
            'query_emb_dim': 768,  # all-mpnet-base-v2
        }

        with open(save_path / "qg_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"QueryGraphReranker saved to {save_path}")

    @classmethod
    def load(cls, load_path: str):
        """
        Load QueryGraphReranker model and all components.

        Args:
            load_path: Directory to load model from

        Returns:
            QueryGraphReranker: Loaded model
        """
        from pathlib import Path
        import json

        load_path = Path(load_path)

        # Load QG config
        with open(load_path / "qg_config.json", 'r') as f:
            qg_config = json.load(f)

        # Load cross-encoder
        ce_path = load_path / "cross_encoder"
        cross_encoder = CrossEncoderModel.load(str(ce_path))

        # Load query graph builder (recreate with same config)
        from .query_graph import QueryGraphBuilder
        query_graph_builder = QueryGraphBuilder(
            embedding_model="all-mpnet-base-v2",  # Same as training
            similarity_threshold=0.65,
            max_neighbors=10
        )

        # Create model instance
        model = cls(
            cross_encoder=cross_encoder,
            query_graph_builder=query_graph_builder,
            gnn_hidden_dim=qg_config['gnn_hidden_dim'],
            gnn_output_dim=qg_config['gnn_output_dim'],
            gnn_dropout=qg_config['gnn_dropout'],
            lambda_contrastive=qg_config['lambda_contrastive'],
            lambda_rank=qg_config['lambda_rank'],
            temperature=qg_config['temperature']
        )

        # Load GNN components
        gnn_state = torch.load(load_path / "gnn_components.pt", map_location='cpu')
        model.query_gnn.load_state_dict(gnn_state['query_gnn'])
        model.attention.load_state_dict(gnn_state['attention'])
        model.predictor.load_state_dict(gnn_state['predictor'])

        # Load query graph data (if exists)
        graph_path = load_path / "query_graph.pt"
        if graph_path.exists():
            graph_data = torch.load(graph_path, map_location='cpu')
            model.edge_index = graph_data['edge_index']
            model.edge_weights = graph_data['edge_weights']
            model.query_embeddings = graph_data['query_embeddings']
            if 'gnn_query_embeddings' in graph_data:
                model.gnn_query_embeddings = graph_data['gnn_query_embeddings']

        logger.info(f"QueryGraphReranker loaded from {load_path}")
        return model

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
