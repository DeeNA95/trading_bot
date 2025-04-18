import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Dict, Any, Optional

# Import default FeedForward and the new default Attention
from rl_agent.agent.feedforward import FeedForward
from rl_agent.agent.attention.multi_head_attention import MultiHeadAttention # Import MHA

class EncoderBlock(nn.Module):
    """
    A flexible Transformer Encoder Block.
    Uses pre-normalization (LayerNorm before attention/FFN).
    Allows specifying attention, feed-forward, and normalization types.
    Defaults to MultiHeadAttention.
    """
    def __init__(self,
                 n_embd: int,
                 # attention_class: Type[nn.Module], # Removed required type hint
                 attention_args: Dict[str, Any], # Keep attention_args required
                 attention_class: Type[nn.Module] = MultiHeadAttention, # Set MHA as default
                 ffn_class: Type[nn.Module] = FeedForward,
                 ffn_args: Optional[Dict[str, Any]] = None,
                 norm_class: Type[nn.Module] = nn.LayerNorm,
                 norm_args: Optional[Dict[str, Any]] = None,
                 dropout: float = 0.1):
        """
        Initializes the EncoderBlock.

        Args:
            n_embd: Embedding dimension.
            attention_args: Arguments for attention_class constructor (e.g., {'n_heads': 8}).
            attention_class: Class for the self-attention mechanism. Defaults to MultiHeadAttention.
            ffn_class: Class for the feed-forward network. Defaults to FeedForward.
            ffn_args: Arguments for ffn_class constructor.
            norm_class: Class for the normalization layer. Defaults to nn.LayerNorm.
            norm_args: Arguments for norm_class constructor.
            dropout: Dropout rate for residual connections.
        """
        super().__init__()

        if norm_args is None:
            norm_args = {}
        if ffn_args is None:
            ffn_args = {'dropout': dropout}

        # Ensure n_embd is passed if needed by components
        if 'n_embd' not in attention_args:
             attention_args['n_embd'] = n_embd
        if 'n_embd' not in ffn_args:
             ffn_args['n_embd'] = n_embd

        # Instantiate components
        self.norm1 = norm_class(n_embd, **norm_args)
        self.self_attn = attention_class(**attention_args) # Instantiate using provided or default class
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = norm_class(n_embd, **norm_args)
        self.ffn = ffn_class(**ffn_args)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Encoder Block (Pre-Norm).

        Args:
            src: Input tensor (batch_size, seq_len, n_embd).
            src_mask: Optional mask for self-attention.

        Returns:
            Output tensor (batch_size, seq_len, n_embd).
        """
        # 1. Self-Attention block (LayerNorm -> Attention -> Dropout -> Residual)
        norm_src = self.norm1(src)
        try:
            attn_output = self.self_attn(query=norm_src, key=norm_src, value=norm_src, mask=src_mask)
        except TypeError:
             try:
                 attn_output = self.self_attn(norm_src, mask=src_mask)
             except TypeError:
                 attn_output = self.self_attn(norm_src)

        src = src + self.dropout1(attn_output)

        # 2. Feed-Forward block (LayerNorm -> FFN -> Dropout -> Residual)
        norm_src = self.norm2(src)
        ffn_output = self.ffn(norm_src)
        src = src + self.dropout2(ffn_output)

        return src
