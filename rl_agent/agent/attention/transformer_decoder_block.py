import torch
import torch.nn as nn


class TransformerDecoderBlock(nn.Module):
    """
    Standard Transformer Decoder block.

    Includes multi-head self-attention, feed-forward network, layer normalization,
    and residual connections. Uses causal masking for the self-attention layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        activation_fn: nn.Module = nn.GELU(),
    ):
        """
        Initialize the Transformer Decoder block.

        Args:
            hidden_dim: Size of the hidden dimension (embedding dimension)
            n_heads: Number of attention heads
            dropout: Dropout probability
            dim_feedforward: Dimension of the feed-forward network
            activation_fn: Activation function for the feed-forward network
        """
        super(TransformerDecoderBlock, self).__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # Expect (batch, seq, feature)
        )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generate a causal mask for self-attention.
        Ensures that attention is only paid to previous positions.
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # For nn.MultiheadAttention, the mask should be additive.
        # A value of -inf indicates that the position should not be attended to.
        # A value of 0.0 indicates that the position can be attended to.
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer Decoder block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Generate causal mask for self-attention
        # MHA expects mask shape (L, S) or (N*num_heads, L, S)
        # where L is target sequence length, S is source sequence length
        # For self-attention L=S=seq_len
        causal_mask = self._generate_causal_mask(seq_len, device)

        # Multi-head self-attention with residual connection and layer norm
        # Note: MHA expects query, key, value. For self-attention, they are the same.
        # Need mask to prevent attending to future positions.
        # is_causal=True can be used for PyTorch >= 1.9 for efficiency
        # but explicit mask is more general.
        attn_output, _ = self.self_attn(
            query=x, key=x, value=x, attn_mask=causal_mask, is_causal=False # Set is_causal=False if using explicit mask
        )
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x
