import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Dict, Any, Optional

# Import default FeedForward and the new default Attention
from rl_agent.agent.feedforward import FeedForward
from rl_agent.agent.attention.multi_head_attention import MultiHeadAttention # Import MHA

class EncoderBlock(nn.Module):
    """
    A flexible Transformer Encoder Block with enhanced residual connections.
    Uses pre-normalization (LayerNorm before attention/FFN).
    Allows specifying attention, feed-forward, and normalization types.
    Defaults to MultiHeadAttention.

    Features:
    - Standard residual connections
    - Optional residual scaling
    - Optional gated residual connections
    - Optional final layer normalization
    """
    def __init__(self,
                 embedding_dim: int, # Renamed from n_embd
                 # attention_class: Type[nn.Module], # Removed required type hint
                 attention_args: Dict[str, Any], # Keep attention_args required
                 attention_class: Type[nn.Module] = MultiHeadAttention, # Set MHA as default
                 ffn_class: Type[nn.Module] = FeedForward,
                 ffn_args: Optional[Dict[str, Any]] = None,
                 norm_class: Type[nn.Module] = nn.LayerNorm,
                 norm_args: Optional[Dict[str, Any]] = None,
                 dropout: float = 0.1,
                 residual_scale: float = 1.0,
                 use_gated_residual: bool = False,
                 use_final_norm: bool = False):
        """
        Initializes the EncoderBlock.

        Args:
            embedding_dim: Embedding dimension.
            attention_args: Arguments for attention_class constructor (e.g., {'embedding_dim': 256, 'n_heads': 8}).
            attention_class: Class for the self-attention mechanism. Defaults to MultiHeadAttention.
            ffn_class: Class for the feed-forward network. Defaults to FeedForward.
            ffn_args: Arguments for ffn_class constructor.
            norm_class: Class for the normalization layer. Defaults to nn.LayerNorm.
            norm_args: Arguments for norm_class constructor.
            dropout: Dropout rate for residual connections.
            residual_scale: Scaling factor for residual connections (1.0 = standard residual).
            use_gated_residual: Whether to use learnable gates for residual connections.
            use_final_norm: Whether to apply a final layer normalization after all residual connections.
        """
        super().__init__()

        if norm_args is None:
            norm_args = {}
        if ffn_args is None:
            ffn_args = {'dropout': dropout}

        # Ensure embedding_dim is passed if needed by components, using the new standard name
        # Note: The factory should now be primarily responsible for ensuring args are correct before passing them.
        # This block now assumes 'embedding_dim' is the standard.
        if 'embedding_dim' not in attention_args:
             attention_args['embedding_dim'] = embedding_dim
        if 'embedding_dim' not in ffn_args: # FFN classes might expect embedding_dim or similar
             ffn_args['embedding_dim'] = embedding_dim # Assuming FFN also standardizes

        # Store parameters
        self.residual_scale = residual_scale
        self.use_gated_residual = use_gated_residual
        self.use_final_norm = use_final_norm

        # Instantiate components
        self.norm1 = norm_class(embedding_dim, **norm_args) # Use embedding_dim
        self.self_attn = attention_class(**attention_args) # Instantiate using provided or default class
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = norm_class(embedding_dim, **norm_args) # Use embedding_dim
        self.ffn = ffn_class(**ffn_args)
        self.dropout2 = nn.Dropout(dropout)

        # Optional gated residual connections
        if self.use_gated_residual:
            self.gate1 = nn.Parameter(torch.ones(1))
            self.gate2 = nn.Parameter(torch.ones(1))

        # Optional final layer normalization
        if self.use_final_norm:
            self.final_norm = norm_class(embedding_dim, **norm_args)

    def _apply_residual(self, x: torch.Tensor, residual: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply residual connection with optional gating and scaling."""
        if self.use_gated_residual and gate is not None:
            return x + gate * self.residual_scale * residual
        else:
            return x + self.residual_scale * residual

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Encoder Block (Pre-Norm).

        Args:
            src: Input tensor (batch_size, seq_len, embedding_dim).
            src_mask: Optional mask for self-attention.

        Returns:
            Output tensor (batch_size, seq_len, embedding_dim).
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

        # Apply residual connection with optional gating
        gate1 = self.gate1 if self.use_gated_residual else None
        src = self._apply_residual(src, self.dropout1(attn_output), gate1)

        # 2. Feed-Forward block (LayerNorm -> FFN -> Dropout -> Residual)
        norm_src = self.norm2(src)
        ffn_output = self.ffn(norm_src)

        # Apply residual connection with optional gating
        gate2 = self.gate2 if self.use_gated_residual else None
        src = self._apply_residual(src, self.dropout2(ffn_output), gate2)

        # Apply final normalization if enabled
        if self.use_final_norm:
            src = self.final_norm(src)

        return src
