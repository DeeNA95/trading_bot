import torch
import torch.nn as nn
from typing import Optional

class TimeEmbedding(nn.Module):
    """
    Time-aware embedding for capturing temporal information.
    Combines learnable position embeddings with a temporal encoding network.
    """
    def __init__(self, hidden_dim: int, max_len: int = 512):
        """
        Initializes the time embedding layer.

        Args:
            hidden_dim: Size of the hidden dimension.
            max_len: Maximum sequence length for position embedding.
        """
        super(TimeEmbedding, self).__init__()

        # Learnable position embedding
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_len, hidden_dim), requires_grad=True
        )

        # Temporal encoding network (optional based on time_values input)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, time_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Adds time-aware embedding to the input tensor.

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim).
            time_values: Optional tensor with time values (batch_size, seq_len, 1).

        Returns:
            Time-embedded tensor (batch_size, seq_len, hidden_dim).
        """
        seq_len = x.shape[1]

        # Add learnable position embedding (sliced to current seq_len)
        x = x + self.position_embedding[:, :seq_len, :]

        # Add temporal encoding if time values are provided
        if time_values is not None:
            temporal_code = self.temporal_encoder(time_values)
            x = x + temporal_code

        return x
