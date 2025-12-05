"""
Set Transformer for Listwise Ranking

Enables documents to "resonate" with each other through self-attention,
solving the lost-in-middle problem. Based on "Set Transformer" paper
adapted for ranking tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for list position awareness.

    Adds sinusoidal position embeddings to help the model
    understand document ranking positions.
    """

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Tensor with positional encodings added
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention for set elements.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Attention output [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)

        # Linear projections and split into heads
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # Accept masks shaped [batch, seq] or already broadcastable;
            # expand to [batch, 1, 1, seq] for scores shape [batch, heads, seq, seq]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

            mask = mask.to(dtype=torch.bool)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        output = torch.matmul(attention, V)

        # Concatenate heads and apply final linear
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            Transformed tensor
        """
        x = self.linear1(x)
        x = F.gelu(x)  # Using GELU like BERT
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class SetTransformerBlock(nn.Module):
    """
    Single Set Transformer block with self-attention and feed-forward.
    """

    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, num_docs, d_model]
            mask: Optional attention mask

        Returns:
            Transformed tensor [batch_size, num_docs, d_model]
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class SetTransformer(nn.Module):
    """
    Set Transformer for listwise document ranking.

    Processes a set of document representations with self-attention,
    allowing documents to interact and influence each other's scores.

    Args:
        d_model: Hidden dimension (must match cross-encoder output)
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
        use_positional_encoding: Whether to add position embeddings
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        use_positional_encoding: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.use_positional_encoding = use_positional_encoding

        # Positional encoding for rank-awareness
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, max_len=100)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            SetTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Process a set of document representations.

        Args:
            x: Document representations [batch_size, num_docs, d_model]
            mask: Optional padding mask [batch_size, num_docs]

        Returns:
            Contextualized representations [batch_size, num_docs, d_model]
        """
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.positional_encoding(x)

        x = self.dropout(x)

        # Prepare attention mask if provided
        if mask is not None:
            # Convert mask to attention mask format [batch, 1, 1, num_docs]
            attention_mask = mask.unsqueeze(1).unsqueeze(2)
        else:
            attention_mask = None

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        return x

    def get_attention_weights(self, x, layer_idx=0):
        """
        Get attention weights from a specific layer for visualization.

        Args:
            x: Input [batch_size, num_docs, d_model]
            layer_idx: Which layer to extract attention from

        Returns:
            Attention weights [batch_size, num_heads, num_docs, num_docs]
        """
        if layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} >= num_layers {self.num_layers}")

        # Apply positional encoding
        if self.use_positional_encoding:
            x = self.positional_encoding(x)

        # Forward through blocks until desired layer
        for i, block in enumerate(self.blocks):
            if i == layer_idx:
                # Extract attention from this layer
                batch_size = x.size(0)
                Q = block.attention.q_linear(x).view(batch_size, -1, block.attention.num_heads, block.attention.d_k).transpose(1, 2)
                K = block.attention.k_linear(x).view(batch_size, -1, block.attention.num_heads, block.attention.d_k).transpose(1, 2)

                scores = torch.matmul(Q, K.transpose(-2, -1)) / block.attention.scale
                attention = F.softmax(scores, dim=-1)

                return attention

            x = block(x)

        return None
