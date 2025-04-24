import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Dict, Any, Optional

# Import default components
from rl_agent.agent.feedforward import FeedForward
from rl_agent.agent.attention.multi_head_attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    """
    A flexible Transformer Decoder Block with enhanced residual connections.
    Can operate with or without cross-attention.
    Uses pre-normalization. Allows specifying attention, FFN, and norm types.
    Defaults to MultiHeadAttention for attention layers.

    Features:
    - Standard residual connections
    - Optional residual scaling
    - Optional gated residual connections
    - Optional final layer normalization
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
                 dropout: float = 0.1,
                 residual_scale: float = 1.0,
                 use_gated_residual: bool = False,
                 use_final_norm: bool = False):
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
            residual_scale: Scaling factor for residual connections (1.0 = standard residual).
            use_gated_residual: Whether to use learnable gates for residual connections.
            use_final_norm: Whether to apply a final layer normalization after all residual connections.
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

        # Store parameters
        self.residual_scale = residual_scale
        self.use_gated_residual = use_gated_residual
        self.use_final_norm = use_final_norm

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

        # Optional gated residual connections
        if self.use_gated_residual:
            self.gate_self = nn.Parameter(torch.ones(1))
            if self.has_cross_attention:
                self.gate_cross = nn.Parameter(torch.ones(1))
            self.gate_ffn = nn.Parameter(torch.ones(1))

        # Optional final layer normalization
        if self.use_final_norm:
            self.final_norm = norm_class(embedding_dim, **norm_args)


    def _apply_residual(self, x: torch.Tensor, residual: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply residual connection with optional gating and scaling."""
        if self.use_gated_residual and gate is not None:
            return x + gate * self.residual_scale * residual
        else:
            return x + self.residual_scale * residual

    def forward(self,
                tgt: Optional[torch.Tensor] = None,
                memory: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                input_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Decoder Block (Pre-Norm).

        Args:
            tgt: Target sequence tensor (batch_size, seq_len_tgt, embedding_dim).
                 Can be None if input_tensor is provided.
            memory: Encoder output tensor (batch_size, seq_len_mem, embedding_dim).
                    Required if cross-attention is enabled, otherwise ignored.
            tgt_mask: Mask for self-attention (e.g., causal mask).
            memory_mask: Mask for cross-attention (e.g., padding mask for memory).
            input_tensor: Alternative input tensor that can be used instead of tgt.
                         Useful for decoder-only models where the input is the output
                         of a previous layer.

        Returns:
            Output tensor (batch_size, seq_len, embedding_dim).
        """
        # Determine the input tensor
        if tgt is None and input_tensor is None:
            raise ValueError("Either tgt or input_tensor must be provided")

        # Use tgt if provided, otherwise use input_tensor
        input_x = tgt if tgt is not None else input_tensor

        # 1. Masked Self-Attention block
        norm_input_self = self.norm_self(input_x)
        try:
            attn_output_self = self.self_attn(query=norm_input_self, key=norm_input_self, value=norm_input_self, mask=tgt_mask)
        except TypeError: # Fallback for simpler signatures
             try: attn_output_self = self.self_attn(norm_input_self, mask=tgt_mask)
             except TypeError: attn_output_self = self.self_attn(norm_input_self)

        # Apply residual connection with optional gating
        gate_self = self.gate_self if self.use_gated_residual else None
        x = self._apply_residual(input_x, self.dropout_self(attn_output_self), gate_self)

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

            # Apply residual connection with optional gating
            gate_cross = self.gate_cross if self.use_gated_residual and self.has_cross_attention else None
            x = self._apply_residual(x, self.dropout_cross(attn_output_cross), gate_cross)

        # 3. Feed-Forward block
        norm_x_ffn = self.norm_ffn(x) # Norm applied to output of previous stage (self-attn or cross-attn)
        ffn_output = self.ffn(norm_x_ffn)

        # Apply residual connection with optional gating
        gate_ffn = self.gate_ffn if self.use_gated_residual else None
        x = self._apply_residual(x, self.dropout_ffn(ffn_output), gate_ffn)

        # Apply final normalization if enabled
        if self.use_final_norm:
            x = self.final_norm(x)

        return x
