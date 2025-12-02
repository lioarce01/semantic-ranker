"""
Graph Neural Network for query graph processing.

Implements message passing over query graphs to enable cross-query learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class QueryGNN(nn.Module):
    """2-layer Graph Convolutional Network for query embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
        use_edge_weights: bool = True
    ):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout: Dropout rate
            use_edge_weights: Whether to use edge weights in message passing
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_edge_weights = use_edge_weights

        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges]

        Returns:
            Updated node embeddings [num_nodes, output_dim]
        """
        # Layer 1
        x = self.conv1(x, edge_index, edge_weight if self.use_edge_weights else None)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index, edge_weight if self.use_edge_weights else None)
        x = self.norm2(x)

        return x


class GCNConv(nn.Module):
    """Graph Convolutional Layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Linear transformation
        x = self.linear(x)

        # Handle empty graph
        if edge_index.size(1) == 0:
            return x

        # Message passing
        row, col = edge_index
        num_nodes = x.size(0)

        # Compute normalization
        deg = torch.zeros(num_nodes, dtype=torch.float, device=x.device)
        if edge_weight is not None:
            deg.index_add_(0, row, edge_weight)
        else:
            deg.index_add_(0, row, torch.ones(edge_index.size(1), device=x.device))

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Normalize edge weights
        if edge_weight is not None:
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        else:
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Aggregate messages
        out = torch.zeros_like(x)
        out.index_add_(0, row, norm.view(-1, 1) * x[col])

        return out


class QueryGraphAttention(nn.Module):
    """
    Attention mechanism for aggregating query graph information
    into cross-encoder predictions.
    """

    def __init__(self, query_dim: int, cross_encoder_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.ce_proj = nn.Linear(cross_encoder_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        cross_encoder_outputs: torch.Tensor,
        query_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_embeddings: GNN query embeddings [num_queries, query_dim]
            cross_encoder_outputs: Cross-encoder hidden states [batch_size, ce_dim]
            query_indices: Query index for each batch item [batch_size]

        Returns:
            Attended outputs [batch_size, ce_dim]
        """
        # Get query embeddings for batch
        batch_query_emb = query_embeddings[query_indices]  # [batch_size, query_dim]

        # Project
        query_proj = torch.tanh(self.query_proj(batch_query_emb))  # [batch_size, hidden]
        ce_proj = torch.tanh(self.ce_proj(cross_encoder_outputs))  # [batch_size, hidden]

        # Compute attention weights
        combined = query_proj + ce_proj
        attn_weights = torch.sigmoid(self.attention(combined))  # [batch_size, 1]

        # Apply attention
        attended = attn_weights * cross_encoder_outputs

        return attended
