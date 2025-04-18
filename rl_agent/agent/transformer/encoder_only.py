import torch.nn as nn
from typing import List, Type, Dict, Any, Optional
from ..blocks.encoder_block import EncoderBlock
from ..embeddings import TimeEmbedding
# Import default components if needed for defaults
from ..attention.multi_head_attention import MultiHeadAttention
from ..feedforward import FeedForward
import torch

class EncoderOnlyTransformer(nn.Module):
    def __init__(self, n_embd: int, n_layers: int, window_size: int,
                 block_class: Type[nn.Module] = EncoderBlock,
                 attention_class: Type[nn.Module] = MultiHeadAttention,
                 attention_args: Dict[str, Any] = None,
                 ffn_class: Type[nn.Module] = FeedForward,
                 ffn_args: Optional[Dict[str, Any]] = None,
                 norm_class: Type[nn.Module] = nn.LayerNorm,
                 norm_args: Optional[Dict[str, Any]] = None,
                 dropout: float = 0.1,
                 use_causal_mask: bool = False): # Control masking
        super().__init__()
        self.use_causal_mask = use_causal_mask
        self.window_size = window_size

        # Embeddings (Input embedding handled outside, TimeEmbedding included)
        self.time_embedding = TimeEmbedding(n_embd, max_len=window_size)
        self.dropout_emb = nn.Dropout(dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
             # Simplified args passing, assuming EncoderBlock structure
             block = block_class(
                 n_embd=n_embd,
                 attention_class=attention_class,
                 attention_args=attention_args.copy(),
                 ffn_class=ffn_class,
                 ffn_args=ffn_args,
                 norm_class=norm_class,
                 norm_args=norm_args,
                 dropout=dropout
             )
             self.blocks.append(block)

        self.final_norm = norm_class(n_embd) if norm_class else nn.LayerNorm(n_embd)

    def _generate_causal_mask(self, size: int, device) -> Optional[torch.Tensor]:
         if not self.use_causal_mask:
             return None
         mask = torch.tril(torch.ones(size, size, device=device)).bool()
         return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
         # Assume x is already embedded: (B, T, C)
         batch_size, seq_len, _ = x.shape
         assert seq_len <= self.window_size, "Input sequence longer than max_len"

         # Add time embedding
         time_indices = torch.arange(seq_len, dtype=torch.float32, device=x.device)
         time_indices = time_indices.view(1, -1, 1).expand(batch_size, -1, -1)
         x = self.time_embedding(x, time_indices)
         x = self.dropout_emb(x)

         # Generate mask
         mask = self._generate_causal_mask(seq_len, x.device)

         # Apply blocks
         for block in self.blocks:
             x = block(x, src_mask=mask) # EncoderBlock uses src_mask

         x = self.final_norm(x)
         return x # Return final hidden states (B, T, C)
