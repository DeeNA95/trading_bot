import torch
import torch.nn as nn

from .pyramidal_attention import PyramidalAttention


class PyraFormerBlock(nn.Module):
    """
    PyraFormer block with feed-forward network and layer normalization.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int=8,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        """
        Initialize the PyraFormer block.

        Args:
            hidden_dim: Size of the hidden dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_len: Maximum sequence length supported
        """
        super(PyraFormerBlock, self).__init__()

        # Pyramidal attention
        self.attention = PyramidalAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PyraFormer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Multi-head attention with residual connection and layer norm
        attended = self.attention(self.norm1(x))
        x = x + self.dropout(attended)

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x
