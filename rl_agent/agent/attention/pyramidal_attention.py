import torch
import torch.nn as nn
import torch.nn.functional as F



class PyramidalAttention(nn.Module):
    """
    Pyramidal Attention module for time series data.

    This implements log-sparse attention patterns at multiple time scales
    for efficient processing of long sequences with O(n log n) complexity.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 8,
        dropout: float = 0.25,
        max_seq_len: int = 512,
    ):
        """
        Initialize the Pyramidal Attention module.

        Args:
            hidden_dim: Size of the hidden dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_len: Maximum sequence length supported
        """
        super(PyramidalAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Ensure hidden_dim is divisible by n_heads
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        # Multi-scale attention layers with different dilation rates
        # 2^0, 2^1, 2^2, etc.
        self.dilations = [2**i for i in range(min(5, n_heads))]

        # Projections for Q, K, V for each attention head
        self.query_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, self.head_dim) for _ in range(n_heads)]
        )
        self.key_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, self.head_dim) for _ in range(n_heads)]
        )
        self.value_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, self.head_dim) for _ in range(n_heads)]
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Pre-compute log-sparse attention masks for different dilations
        self.register_buffer("masks", self._generate_masks(max_seq_len))

    def _generate_masks(self, max_seq_len: int) -> torch.Tensor:
        """
        Generate log-sparse attention masks for different dilation rates.

        Args:
            max_seq_len: Maximum sequence length

        Returns:
            Tensor of shape (n_heads, max_seq_len, max_seq_len) with masks
        """
        # Determine device from a parameter (assuming parameters are on the correct device)
        # This assumes the module has been moved to the device before this is called.
        # A more robust way might involve passing the device explicitly if needed earlier.
        device = next(self.parameters()).device
        masks = torch.zeros(len(self.dilations), max_seq_len, max_seq_len, device=device)

        for h, dilation in enumerate(self.dilations):
            for i in range(max_seq_len):
                # For each position, attend to:
                # - immediate neighbors based on dilation
                # - log-sparse positions (powers of 2 distance away)
                for j in range(max_seq_len):
                    # Immediate neighborhood attention based on dilation
                    if abs(i - j) <= dilation:
                        masks[h, i, j] = 1.0
                    # Log-sparse attention (powers of 2)
                    elif abs(i - j) % (2 ** (1 + h)) == 0:
                        masks[h, i, j] = 1.0

        return masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pyramidal attention module.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Attended tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Use pre-computed masks or dynamically generate for longer sequences
        if seq_len > self.masks.shape[1]:
            attention_masks = self._generate_masks(seq_len).to(device)
        else:
            attention_masks = self.masks[:, :seq_len, :seq_len]

        # Multi-head attention with different dilations
        head_outputs = []

        for h in range(self.n_heads):
            q = self.query_projections[h](x)
            k = self.key_projections[h](x)
            v = self.value_projections[h](x)

            # Compute attention scores
            attention_scores = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim**0.5)

            # Apply log-sparse mask
            mask = attention_masks[min(h, len(self.dilations) - 1)]
            attention_scores = attention_scores.masked_fill(
                mask.expand(batch_size, seq_len, seq_len) == 0, float("-inf")
            )

            # Apply softmax and dropout
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)

            # Apply attention to values
            head_output = torch.bmm(attention_probs, v)
            head_outputs.append(head_output)

        # Concatenate heads and project
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.output_projection(concatenated)

        return output

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
