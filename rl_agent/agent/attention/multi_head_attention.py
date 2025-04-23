import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention mechanism.

    This implementation projects Q, K, V once for all heads,
    then reshapes and computes attention in parallel across heads.
    """
    def __init__(self, embedding_dim: int, n_heads: int, dropout: float = 0.1, bias: bool = False): # Renamed n_embd
        """
        Initialize the MultiHeadAttention module.

        Args:
            embedding_dim: Total dimension of the model (embedding size).
            n_heads: Number of attention heads.
            dropout: Dropout probability.
            bias: Whether to use bias in linear layers. Defaults to False.
        """
        super().__init__()
        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads" # Renamed n_embd

        self.embedding_dim = embedding_dim # Renamed n_embd
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads # Renamed n_embd
        self.dropout = dropout

        # Single linear layers for Q, K, V projections for all heads
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias) # Renamed n_embd
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias) # Renamed n_embd
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias) # Renamed n_embd

        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias) # Renamed n_embd

        # Dropout layer
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for Multi-Head Attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, embedding_dim).
            key: Key tensor of shape (batch_size, seq_len_kv, embedding_dim).
            value: Value tensor of shape (batch_size, seq_len_kv, embedding_dim).
            mask: Optional mask tensor. Its shape depends on the type of mask:
                  - For padding mask: (batch_size, 1, 1, seq_len_kv) or (batch_size, seq_len_q, seq_len_kv)
                  - For causal mask: (1, 1, seq_len_q, seq_len_q) or (seq_len_q, seq_len_q)
                  Mask values should be 0 for positions to be masked and 1 otherwise,
                  or boolean (True to keep, False to mask). The code handles conversion.

        Returns:
            Output tensor of shape (batch_size, seq_len_q, embedding_dim).
        """
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_kv, _ = key.shape

        # 1. Project Q, K, V for all heads
        q = self.q_proj(query)  # (B, T_q, C)
        k = self.k_proj(key)    # (B, T_kv, C)
        v = self.v_proj(value)  # (B, T_kv, C)

        # 2. Reshape Q, K, V to separate heads
        # (B, T, C) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Compute attention scores (Scaled Dot-Product Attention)
        # (B, nh, T_q, hs) @ (B, nh, hs, T_kv) -> (B, nh, T_q, T_kv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. Apply mask (if provided)
        if mask is not None:
            # Ensure mask has compatible dimensions for broadcasting: (B, nh, T_q, T_kv)
            if mask.dim() == 2: # (T_q, T_kv) -> (1, 1, T_q, T_kv) - Typical for causal mask
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3: # (B, T_q, T_kv) -> (B, 1, T_q, T_kv)
                mask = mask.unsqueeze(1)
            # Mask values should be False or 0 where attention is NOT allowed.
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5. Apply softmax and dropout to attention probabilities
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 6. Apply attention to values
        # (B, nh, T_q, T_kv) @ (B, nh, T_kv, hs) -> (B, nh, T_q, hs)
        context = torch.matmul(attn_probs, v)

        # 7. Concatenate heads and project output
        # (B, nh, T_q, hs) -> (B, T_q, nh, hs) -> (B, T_q, C)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embedding_dim) # Renamed n_embd
        output = self.resid_dropout(self.out_proj(context))

        return output
