import torch
import torch.nn as nn
from typing import Type, Dict, Any, Optional

# Import necessary components from your project structure
from ..blocks.encoder_block import EncoderBlock
from ..blocks.decoder_block import DecoderBlock
from ..attention.multi_head_attention import MultiHeadAttention
from ..feedforward import FeedForward
from ..embeddings import TimeEmbedding

class EncoderDecoderTransformer(nn.Module):
    """
    Standard Encoder-Decoder Transformer architecture using MultiHeadAttention.
    """
    def __init__(self,
                 n_embd: int,
                 n_encoder_layers: int,
                 n_decoder_layers: int,
                 window_size: int, # Needed for TimeEmbedding max_len
                 attention_args: Dict[str, Any], # Moved before attention_class
                 attention_class: Type[nn.Module] = MultiHeadAttention, # Moved after attention_args
                 ffn_class: Type[nn.Module] = FeedForward,
                 ffn_args: Optional[Dict[str, Any]] = None,
                 norm_class: Type[nn.Module] = nn.LayerNorm,
                 norm_args: Optional[Dict[str, Any]] = None,
                 dropout: float = 0.1):
        """
        Initializes the Encoder-Decoder Transformer.

        Args:
            n_embd: Embedding dimension.
            n_encoder_layers: Number of layers in the encoder stack.
            n_decoder_layers: Number of layers in the decoder stack.
            window_size: Max sequence length for time embedding.
            attention_args: Arguments for attention_class constructor. # Moved doc
            attention_class: The class to use for attention mechanisms. # Moved doc
            ffn_class: Class for the feed-forward network.
            ffn_args: Arguments for ffn_class constructor.
            norm_class: Class for the normalization layer.
            norm_args: Arguments for norm_class constructor.
            dropout: Dropout rate.
        """
        super().__init__()
        self.window_size = window_size

        # --- Embeddings (Assuming input embedding happens outside) ---
        # Include TimeEmbedding for both encoder and decoder inputs if needed
        self.encoder_time_embedding = TimeEmbedding(n_embd, max_len=window_size)
        self.decoder_time_embedding = TimeEmbedding(n_embd, max_len=window_size) # Separate instance if needed
        self.dropout_emb = nn.Dropout(dropout)

        # --- Encoder Stack ---
        self.encoder_blocks = nn.ModuleList()
        for _ in range(n_encoder_layers):
            block = EncoderBlock(
                n_embd=n_embd,
                attention_class=attention_class, # Pass attention_class
                attention_args=attention_args.copy(),
                ffn_class=ffn_class,
                ffn_args=ffn_args,
                norm_class=norm_class,
                norm_args=norm_args,
                dropout=dropout
            )
            self.encoder_blocks.append(block)
        self.encoder_norm = norm_class(n_embd) if norm_class else nn.LayerNorm(n_embd)

        # --- Decoder Stack ---
        self.decoder_blocks = nn.ModuleList()
        for _ in range(n_decoder_layers):
            block = DecoderBlock(
                n_embd=n_embd,
                self_attention_class=attention_class, # Pass attention_class
                self_attention_args=attention_args.copy(),
                cross_attention_class=attention_class, # Pass attention_class
                cross_attention_args=attention_args.copy(), # Enable cross-attn
                ffn_class=ffn_class,
                ffn_args=ffn_args,
                norm_class=norm_class,
                norm_args=norm_args,
                dropout=dropout
            )
            self.decoder_blocks.append(block)
        self.decoder_norm = norm_class(n_embd) if norm_class else nn.LayerNorm(n_embd)

    def _generate_causal_mask(self, size: int, device) -> torch.Tensor:
         mask = torch.tril(torch.ones(size, size, device=device)).bool()
         return mask

    def forward(self,
                src: torch.Tensor, # Encoder input (B, T_src, C)
                tgt: torch.Tensor, # Decoder input (B, T_tgt, C)
                src_mask: Optional[torch.Tensor] = None, # Padding mask for src
                tgt_mask: Optional[torch.Tensor] = None  # Causal mask for tgt self-attn
               ) -> torch.Tensor:
        """
        Forward pass through the Encoder-Decoder Transformer.

        Args:
            src: Source sequence tensor (encoder input).
            tgt: Target sequence tensor (decoder input).
            src_mask: Padding mask for the source sequence.
            tgt_mask: Causal mask for the target sequence self-attention.

        Returns:
            Output tensor from the decoder (B, T_tgt, C).
        """
        # --- Encoder ---
        batch_size_src, seq_len_src, _ = src.shape
        time_indices_src = torch.arange(seq_len_src, dtype=torch.float32, device=src.device)
        time_indices_src = time_indices_src.view(1, -1, 1).expand(batch_size_src, -1, -1)
        memory = self.encoder_time_embedding(src, time_indices_src)
        memory = self.dropout_emb(memory)

        for block in self.encoder_blocks:
            memory = block(memory, src_mask=src_mask)
        memory = self.encoder_norm(memory) # Apply norm after last encoder block

        # --- Decoder ---
        batch_size_tgt, seq_len_tgt, _ = tgt.shape
        time_indices_tgt = torch.arange(seq_len_tgt, dtype=torch.float32, device=tgt.device)
        time_indices_tgt = time_indices_tgt.view(1, -1, 1).expand(batch_size_tgt, -1, -1)
        output = self.decoder_time_embedding(tgt, time_indices_tgt)
        output = self.dropout_emb(output)

        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(seq_len_tgt, tgt.device)

        for block in self.decoder_blocks:
            output = block(
                tgt=output,
                memory=memory, # Pass encoder output
                tgt_mask=tgt_mask, # Causal mask for self-attention
                memory_mask=src_mask # Padding mask for cross-attention (masking encoder output)
            )
        output = self.decoder_norm(output) # Apply norm after last decoder block

        return output
