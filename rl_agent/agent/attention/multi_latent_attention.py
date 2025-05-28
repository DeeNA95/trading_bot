import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# Assuming MultiHeadAttention is in the same directory or accessible via path
from .multi_head_attention import MultiHeadAttention

class MultiLatentAttention(nn.Module):
    """
    Multi-Latent Attention (MLA) mechanism.

    Uses a set of learnable latent vectors as queries to attend to the input
    sequence (key/value). This allows aggregating information from the input
    into a fixed number of latent representations. Inspired by Set Transformer
    and Perceiver IO concepts.
    """
    def __init__(self, embedding_dim: int, n_heads: int, n_latents: int, dropout: float = 0.1, bias: bool = False): # Renamed n_embd
        """
        Initialize the MultiLatentAttention module.

        Args:
            embedding_dim: Total dimension of the model (embedding size).
            n_heads: Number of attention heads for the underlying MHA.
            n_latents: The number of learnable latent vectors to use as queries.
            dropout: Dropout probability for the underlying MHA.
            bias: Whether to use bias in linear layers for the underlying MHA.
        """
        super().__init__()
        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads" # Renamed n_embd

        self.embedding_dim = embedding_dim # Renamed n_embd
        self.n_heads = n_heads
        self.n_latents = n_latents

        # Learnable latent vectors, initialized typically with standard normal distribution
        self.latents = nn.Parameter(torch.randn(1, n_latents, embedding_dim)) # Renamed n_embd

        # Use standard MHA for the core attention calculation
        # Latents will be the query, input sequence will be key/value
        # Assuming MHA has also been updated to use embedding_dim
        self.mha = MultiHeadAttention(embedding_dim, n_heads, dropout, bias) # Renamed n_embd

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for Multi-Latent Attention.

        Args:
            x: Input tensor (used as key and value) of shape (batch_size, seq_len, embedding_dim).
            mask: Optional mask tensor for the input sequence `x`.
                  Typically a padding mask of shape (batch_size, 1, 1, seq_len)
                  or (batch_size, seq_len).

        Returns:
            Output tensor corresponding to the updated latent vectors,
            shape (batch_size, n_latents, embedding_dim).
        """
        batch_size = x.shape[0]

        # Expand latents to match batch size: (1, n_latents, C) -> (B, n_latents, C)
        latent_queries = self.latents.expand(batch_size, -1, -1)

        # Apply attention: Latents attend to the input sequence x
        # Query: latents (B, n_latents, C)
        # Key:   x       (B, seq_len, C)
        # Value: x       (B, seq_len, C)
        # Mask:  Applied to key/value sequence (seq_len dimension)
        output_latents = self.mha(query=latent_queries, key=x, value=x, mask=mask)

        # Output has shape (batch_size, n_latents, embedding_dim)
        return output_latents
