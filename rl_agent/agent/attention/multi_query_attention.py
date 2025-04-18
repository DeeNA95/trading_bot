import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) mechanism.

    All query heads share a single Key and Value projection/head,
    reducing computation compared to standard Multi-Head Attention.
    """
    def __init__(self, n_embd: int, n_heads: int, dropout: float = 0.1, bias: bool = False):
        """
        Initialize the MultiQueryAttention module.

        Args:
            n_embd: Total dimension of the model (embedding size).
            n_heads: Number of query heads.
            dropout: Dropout probability.
            bias: Whether to use bias in linear layers. Defaults to False.
        """
        super().__init__()
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.dropout = dropout

        # Query projection for all heads
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # Single Key projection (shared across heads)
        self.k_proj = nn.Linear(n_embd, self.head_dim, bias=bias)
        # Single Value projection (shared across heads)
        self.v_proj = nn.Linear(n_embd, self.head_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for Multi-Query Attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, n_embd).
            key: Key tensor of shape (batch_size, seq_len_kv, n_embd).
            value: Value tensor of shape (batch_size, seq_len_kv, n_embd).
            mask: Optional mask tensor. Shape requirements are the same as for MHA.

        Returns:
            Output tensor of shape (batch_size, seq_len_q, n_embd).
        """
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_kv, _ = key.shape

        # 1. Project Q, K, V
        q = self.q_proj(query)      # (B, T_q, C)
        k = self.k_proj(key)        # (B, T_kv, hs) - Single head dim
        v = self.v_proj(value)      # (B, T_kv, hs) - Single head dim

        # 2. Reshape Q to separate heads
        # (B, T_q, C) -> (B, T_q, n_heads, head_dim) -> (B, n_heads, T_q, head_dim)
        q = q.view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Reshape K and V for broadcasting across query heads
        # (B, T_kv, hs) -> (B, 1, T_kv, hs) -> transpose -> (B, 1, hs, T_kv) for K
        # (B, T_kv, hs) -> (B, 1, T_kv, hs) for V
        k = k.unsqueeze(1) # Add head dimension for broadcasting
        v = v.unsqueeze(1) # Add head dimension for broadcasting

        # 4. Compute attention scores (Scaled Dot-Product Attention)
        # (B, nh, T_q, hs) @ (B, 1, hs, T_kv) -> (B, nh, T_q, T_kv)
        # K is broadcast across the n_heads dimension
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
        # (B, nh, T_q, T_kv) @ (B, 1, T_kv, hs) -> (B, nh, T_q, hs)
        # V is broadcast across the n_heads dimension
        context = torch.matmul(attn_probs, v)

        # 8. Concatenate heads and project output
        # (B, nh, T_q, hs) -> (B, T_q, nh, hs) -> (B, T_q, C)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.n_embd)
        output = self.resid_dropout(self.out_proj(context))

        return output
