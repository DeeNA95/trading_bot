import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) mechanism.

    Query heads are divided into groups, and heads within the same group
    share Key and Value projections/heads. Interpolates between MHA and MQA.
    """
    def __init__(self, embedding_dim: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1, bias: bool = False): # Renamed n_embd
        """
        Initialize the GroupedQueryAttention module.

        Args:
            embedding_dim: Total dimension of the model (embedding size).
            n_heads: Total number of query heads.
            n_kv_heads: Number of key/value heads. Must be a divisor of n_heads.
            dropout: Dropout probability.
            bias: Whether to use bias in linear layers. Defaults to False.
        """
        super().__init__()
        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads" # Renamed n_embd
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.embedding_dim = embedding_dim # Renamed n_embd
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.num_q_per_kv = n_heads // n_kv_heads # Number of query heads per key/value head
        self.head_dim = embedding_dim // n_heads # Renamed n_embd
        self.dropout = dropout

        # Query projection for all heads
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias) # Renamed n_embd
        # Key/Value projections for the reduced number of KV heads
        self.k_proj = nn.Linear(embedding_dim, self.n_kv_heads * self.head_dim, bias=bias) # Renamed n_embd
        self.v_proj = nn.Linear(embedding_dim, self.n_kv_heads * self.head_dim, bias=bias) # Renamed n_embd

        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias) # Renamed n_embd

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """ Repeats K/V heads to match the number of query heads. """
        batch_size, seq_len, _ = x.shape
        # Reshape to (B, T_kv, n_kv_heads, hs)
        x = x.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # Repeat heads: (B, T_kv, n_kv_heads, hs) -> (B, T_kv, n_kv_heads, num_q_per_kv, hs)
        x = x.unsqueeze(3).repeat(1, 1, 1, self.num_q_per_kv, 1)
        # Reshape to match query heads: (B, T_kv, n_heads, hs)
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return x

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for Grouped-Query Attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, embedding_dim).
            key: Key tensor of shape (batch_size, seq_len_kv, embedding_dim).
            value: Value tensor of shape (batch_size, seq_len_kv, embedding_dim).
            mask: Optional mask tensor. Shape requirements are the same as for MHA.

        Returns:
            Output tensor of shape (batch_size, seq_len_q, embedding_dim).
        """
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_kv, _ = key.shape

        # 1. Project Q, K, V
        q = self.q_proj(query)      # (B, T_q, C)
        k = self.k_proj(key)        # (B, T_kv, n_kv_heads * hs)
        v = self.v_proj(value)      # (B, T_kv, n_kv_heads * hs)

        # 2. Reshape Q to separate heads
        # (B, T_q, C) -> (B, T_q, n_heads, head_dim) -> (B, n_heads, T_q, head_dim)
        q = q.view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Repeat K and V heads to match query heads and reshape
        # (B, T_kv, n_kv_heads * hs) -> (B, n_heads, T_kv, hs)
        k = self._repeat_kv(k).transpose(1, 2)
        v = self._repeat_kv(v).transpose(1, 2)

        # 4. Compute attention scores (Scaled Dot-Product Attention)
        # (B, nh, T_q, hs) @ (B, nh, hs, T_kv) -> (B, nh, T_q, T_kv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 5. Apply mask (if provided)
        if mask is not None:
            if mask.dim() == 2: # (T_q, T_kv) -> (1, 1, T_q, T_kv)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3: # (B, T_q, T_kv) -> (B, 1, T_q, T_kv)
                mask = mask.unsqueeze(1)
            # Mask is broadcast across the n_heads dimension
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 6. Apply softmax and dropout
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 7. Apply attention to values
        # (B, nh, T_q, T_kv) @ (B, nh, T_kv, hs) -> (B, nh, T_q, hs)
        context = torch.matmul(attn_probs, v)

        # 8. Concatenate heads and project output
        # (B, nh, T_q, hs) -> (B, T_q, nh, hs) -> (B, T_q, C)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embedding_dim) # Renamed n_embd
        output = self.resid_dropout(self.out_proj(context))

        return output
