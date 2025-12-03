"""
Graph Neural Network for query graph processing.

Implements message passing over query graphs to enable cross-query learning.

DQGAN Phase 1: Enhanced architecture with learnable query encoder and cross-attention fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LearnableQueryEncoder(nn.Module):
    """
    Learnable Query Encoder (DQGAN Phase 1).

    Transforms pre-computed query embeddings into a learnable representation
    that can adapt during training.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of input query embeddings (e.g., 768 for all-mpnet-base-v2)
            output_dim: Dimension of learnable query embeddings
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Two-layer MLP for query encoding
        hidden_dim = (input_dim + output_dim) // 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_embeddings: Pre-computed query embeddings [num_queries, input_dim]

        Returns:
            Learnable query embeddings [num_queries, output_dim]
        """
        return self.encoder(query_embeddings)


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Module (DQGAN Phase 1).

    Replaces weak scalar attention with full cross-attention between GNN query embeddings
    and cross-encoder hidden states.
    """

    def __init__(
        self,
        query_dim: int,
        cross_encoder_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            query_dim: Dimension of GNN query embeddings
            cross_encoder_dim: Dimension of cross-encoder hidden states
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.query_dim = query_dim
        self.cross_encoder_dim = cross_encoder_dim
        self.num_heads = num_heads

        # Project query embeddings to cross-encoder dimension
        self.query_proj = nn.Linear(query_dim, cross_encoder_dim)

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=cross_encoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network for final fusion
        self.ffn = nn.Sequential(
            nn.Linear(cross_encoder_dim * 2, cross_encoder_dim),
            nn.LayerNorm(cross_encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cross_encoder_dim, cross_encoder_dim)
        )

        # Layer normalization
        self.norm = nn.LayerNorm(cross_encoder_dim)

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
            Fused representations [batch_size, ce_dim]
        """
        batch_size = cross_encoder_outputs.size(0)

        # Get query embeddings for this batch
        batch_query_emb = query_embeddings[query_indices]  # [batch_size, query_dim]

        # Project query embeddings to cross-encoder dimension
        batch_query_proj = self.query_proj(batch_query_emb)  # [batch_size, ce_dim]

        # Reshape for multi-head attention (add sequence dimension)
        # Query: use batch_query_proj as queries
        # Key/Value: use cross_encoder_outputs as keys and values
        query = batch_query_proj.unsqueeze(1)  # [batch_size, 1, ce_dim]
        key_value = cross_encoder_outputs.unsqueeze(1)  # [batch_size, 1, ce_dim]

        # Apply cross-attention: query embeddings attend to cross-encoder outputs
        attn_output, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value
        )  # [batch_size, 1, ce_dim]

        attn_output = attn_output.squeeze(1)  # [batch_size, ce_dim]

        # Concatenate attention output with original cross-encoder output
        fused = torch.cat([attn_output, cross_encoder_outputs], dim=-1)  # [batch_size, 2 * ce_dim]

        # Feed-forward fusion
        fused = self.ffn(fused)  # [batch_size, ce_dim]

        # Residual connection + normalization
        output = self.norm(fused + cross_encoder_outputs)

        return output


class GraphAttentionNetwork(nn.Module):
    """
    3-layer Graph Attention Network (GAT) for query embeddings.

    DQGAN Phase 2: Replaces QueryGNN with deeper GAT architecture
    for better receptive field and attention-based message passing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        # 3-layer GAT architecture
        # Layer 1: input_dim -> hidden_dim * num_heads
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)

        # Layer 2: hidden_dim * num_heads -> hidden_dim * num_heads
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim * num_heads)

        # Layer 3: hidden_dim * num_heads -> output_dim (single head)
        self.gat3 = GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=dropout, concat=False)
        self.norm3 = nn.LayerNorm(output_dim)

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
            edge_weight: Edge weights [num_edges] (optional)

        Returns:
            Updated node embeddings [num_nodes, output_dim]
        """
        # Layer 1: 1-hop neighbors
        x = self.gat1(x, edge_index, edge_attr=edge_weight)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 2: 2-hop clusters
        x = self.gat2(x, edge_index, edge_attr=edge_weight)
        x = self.norm2(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 3: global context
        x = self.gat3(x, edge_index, edge_attr=edge_weight)
        x = self.norm3(x)

        return x


class GATConv(nn.Module):
    """
    Graph Attention Convolutional Layer.

    Implements multi-head attention mechanism for graph neural networks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2
    ):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension (per head)
            heads: Number of attention heads
            concat: If True, concatenate head outputs; otherwise average
            dropout: Dropout rate for attention coefficients
            negative_slope: LeakyReLU negative slope
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        # Linear transformation for each head
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention mechanism parameters (per head)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # Bias
        if concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge features [num_edges]

        Returns:
            Updated node features [num_nodes, heads * out_channels] if concat
            else [num_nodes, out_channels]
        """
        num_nodes = x.size(0)

        # Handle empty graph
        if edge_index.size(1) == 0:
            x_transformed = self.lin(x)
            if self.concat:
                return x_transformed + self.bias
            else:
                x_transformed = x_transformed.view(-1, self.heads, self.out_channels)
                return x_transformed.mean(dim=1) + self.bias

        # Linear transformation: [num_nodes, heads * out_channels]
        x_transformed = self.lin(x)

        # Reshape to [num_nodes, heads, out_channels]
        x_transformed = x_transformed.view(-1, self.heads, self.out_channels)

        # Compute attention scores
        row, col = edge_index

        # Source and target attention: [num_edges, heads]
        alpha_src = (x_transformed[row] * self.att_src).sum(dim=-1)
        alpha_dst = (x_transformed[col] * self.att_dst).sum(dim=-1)
        alpha = alpha_src + alpha_dst

        # Apply LeakyReLU
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Add edge attributes if provided
        if edge_attr is not None:
            # Broadcast edge_attr to match attention heads
            edge_attr_expanded = edge_attr.view(-1, 1).expand(-1, self.heads)
            alpha = alpha * edge_attr_expanded

        # Softmax normalization per target node
        alpha = self._softmax(alpha, col, num_nodes)

        # Apply dropout to attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Message passing: aggregate neighbor features
        # alpha: [num_edges, heads]
        # x_transformed[col]: [num_edges, heads, out_channels]
        messages = alpha.unsqueeze(-1) * x_transformed[col]  # [num_edges, heads, out_channels]

        # Aggregate messages per node
        out = torch.zeros(num_nodes, self.heads, self.out_channels, device=x.device)
        out.index_add_(0, row, messages)

        # Concatenate or average heads
        if self.concat:
            out = out.view(num_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # Add bias
        out = out + self.bias

        return out

    def _softmax(self, alpha: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute softmax over attention scores per target node.

        Args:
            alpha: Attention scores [num_edges, heads]
            index: Target node indices [num_edges]
            num_nodes: Total number of nodes

        Returns:
            Normalized attention scores [num_edges, heads]
        """
        # Compute exp
        alpha_exp = torch.exp(alpha)

        # Sum exp per target node
        alpha_sum = torch.zeros(num_nodes, self.heads, device=alpha.device)
        alpha_sum.index_add_(0, index, alpha_exp)

        # Normalize
        alpha_norm = alpha_exp / (alpha_sum[index] + 1e-16)

        return alpha_norm


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
