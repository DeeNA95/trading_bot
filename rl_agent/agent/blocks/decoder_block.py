import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Dict, Any, Optional

# Import default components
from rl_agent.agent.feedforward import FeedForward
from rl_agent.agent.attention.multi_head_attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    """
    A flexible Transformer Decoder Block. Can operate with or without cross-attention.
    Uses pre-normalization. Allows specifying attention, FFN, and norm types.
    Defaults to MultiHeadAttention for attention layers.
    """
    def __init__(self,
                 embedding_dim: int, # Renamed from n_embd
                 self_attention_args: Dict[str, Any],
                 cross_attention_args: Optional[Dict[str, Any]] = None, # Made optional
                 self_attention_class: Type[nn.Module] = MultiHeadAttention,
                 cross_attention_class: Type[nn.Module] = MultiHeadAttention,
                 ffn_class: Type[nn.Module] = FeedForward,
                 ffn_args: Optional[Dict[str, Any]] = None,
                 norm_class: Type[nn.Module] = nn.LayerNorm,
                 norm_args: Optional[Dict[str, Any]] = None,
                 dropout: float = 0.1):
        """
        Initializes the DecoderBlock.

        Args:
            embedding_dim: Embedding dimension.
            self_attention_args: Args for self_attention_class constructor (e.g., {'embedding_dim': 256, ...}).
            cross_attention_args: Args for cross_attention_class constructor. If None,
                                  cross-attention is disabled (decoder-only mode).
            self_attention_class: Class for masked self-attention.
            cross_attention_class: Class for encoder-decoder cross-attention.
            ffn_class: Class for the feed-forward network.
            ffn_args: Arguments for ffn_class constructor.
            norm_class: Class for the normalization layer.
            norm_args: Arguments for norm_class constructor.
            dropout: Dropout rate for residual connections.
        """
        super().__init__()

        if norm_args is None: norm_args = {}
        if ffn_args is None: ffn_args = {'dropout': dropout}

        # Ensure embedding_dim is passed if needed, using the new standard name
        # Assumes the factory provides args with 'embedding_dim'
        if 'embedding_dim' not in self_attention_args: self_attention_args['embedding_dim'] = embedding_dim
        if cross_attention_args and 'embedding_dim' not in cross_attention_args: cross_attention_args['embedding_dim'] = embedding_dim
        # Ensure FFN args also use the unified name if they expect it
        # Assuming FFN classes might also expect 'embedding_dim' or similar
        if 'embedding_dim' not in ffn_args: ffn_args['embedding_dim'] = embedding_dim

        # --- Self-Attention Components ---
        self.norm_self = norm_class(embedding_dim, **norm_args) # Use embedding_dim
        self.self_attn = self_attention_class(**self_attention_args)
        self.dropout_self = nn.Dropout(dropout)

        # --- Cross-Attention Components (Conditional) ---
        self.has_cross_attention = cross_attention_args is not None
        if self.has_cross_attention:
            self.norm_cross = norm_class(embedding_dim, **norm_args) # Use embedding_dim
            self.cross_attn = cross_attention_class(**cross_attention_args)
            self.dropout_cross = nn.Dropout(dropout)
        else:
            self.norm_cross = None
            self.cross_attn = None
            self.dropout_cross = None

        # --- Feed-Forward Components ---
        # Use a different norm instance before FFN
        self.norm_ffn = norm_class(embedding_dim, **norm_args) # Use embedding_dim
        self.ffn = ffn_class(**ffn_args)
        self.dropout_ffn = nn.Dropout(dropout)


    def forward(self,
                tgt: torch.Tensor,
                memory: Optional[torch.Tensor] = None, # Made optional
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Decoder Block (Pre-Norm).

        Args:
            tgt: Target sequence tensor (batch_size, seq_len_tgt, embedding_dim).
            memory: Encoder output tensor (batch_size, seq_len_mem, embedding_dim).
                    Required if cross-attention is enabled, otherwise ignored.
            tgt_mask: Mask for self-attention (e.g., causal mask).
            memory_mask: Mask for cross-attention (e.g., padding mask for memory).

        Returns:
            Output tensor (batch_size, seq_len_tgt, embedding_dim).
        """
        # 1. Masked Self-Attention block
        norm_tgt_self = self.norm_self(tgt)
        try:
            attn_output_self = self.self_attn(query=norm_tgt_self, key=norm_tgt_self, value=norm_tgt_self, mask=tgt_mask)
        except TypeError: # Fallback for simpler signatures
             try: attn_output_self = self.self_attn(norm_tgt_self, mask=tgt_mask)
             except TypeError: attn_output_self = self.self_attn(norm_tgt_self)
        # Residual connection for self-attention
        x = tgt + self.dropout_self(attn_output_self)

        # 2. Cross-Attention block (Optional)
        if self.has_cross_attention:
            if memory is None:
                raise ValueError("Memory tensor must be provided when cross-attention is enabled.")
            if self.norm_cross is None or self.cross_attn is None or self.dropout_cross is None:
                 raise RuntimeError("Cross-attention components not initialized properly.") # Should not happen

            norm_x_cross = self.norm_cross(x) # Norm applied to output of self-attn residual
            try:
                attn_output_cross = self.cross_attn(query=norm_x_cross, key=memory, value=memory, mask=memory_mask)
            except TypeError: # Fallback for simpler signatures
                 try: attn_output_cross = self.cross_attn(norm_x_cross, memory, mask=memory_mask)
                 except TypeError: attn_output_cross = self.cross_attn(norm_x_cross, memory)
            # Residual connection for cross-attention
            x = x + self.dropout_cross(attn_output_cross)

        # 3. Feed-Forward block
        norm_x_ffn = self.norm_ffn(x) # Norm applied to output of previous stage (self-attn or cross-attn)
        ffn_output = self.ffn(norm_x_ffn)
        # Residual connection for FFN
        x = x + self.dropout_ffn(ffn_output)

        return x
