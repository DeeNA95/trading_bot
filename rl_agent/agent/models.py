"""
Neural network models for reinforcement learning agents.

This module contains the actor and critic network architectures used by RL agents.
"""

from typing import Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention.pyraformer_block import PyraFormerBlock
from .attention.transformer_decoder_block import TransformerDecoderBlock


def initialize_weights(layer: nn.Module, std: float = 1.0) -> None:
    """
    Initialize the weights of a neural network layer with orthogonal initialization.

    Args:
        layer: Neural network layer to initialize
        std: Standard deviation for scaling the weights
    """
    if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, (nn.LSTM, nn.GRU)):
        for name, param in layer.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param, std)
            elif "bias" in name:
                nn.init.zeros_(param)

# Removed PyramidalAttention and PyraFormerBlock definitions as they are now imported


class TimeEmbedding(nn.Module):
    """
    Time-aware embedding for capturing temporal information.
    """

    def __init__(self, hidden_dim: int, max_len: int = 512):
        """
        Initialize the time embedding layer.

        Args:
            hidden_dim: Size of the hidden dimension
            max_len: Maximum sequence length
        """
        super(TimeEmbedding, self).__init__()

        # Learnable position embedding
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_len, hidden_dim), requires_grad=True
        )

        # Temporal encoding network
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, time_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add time-aware embedding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            time_values: Optional tensor with time values of shape (batch_size, seq_len, 1)

        Returns:
            Time-embedded tensor of shape (batch_size, seq_len, hidden_dim)
        """
        seq_len = x.shape[1]

        # Add learnable position embedding
        x = x + self.position_embedding[:, :seq_len, :]

        # Add temporal encoding if time values are provided
        if time_values is not None:
            temporal_code = self.temporal_encoder(time_values)
            x = x + temporal_code

        return x


class ActorCriticTransformer(nn.Module):
    """
    Actor-Critic model using a Transformer architecture for processing financial time series.

    This model can be configured to use either PyraFormer blocks (with log-sparse
    pyramidal attention) or standard Transformer Decoder blocks (with causal self-attention).
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, features)
        action_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        dropout: float = 0.25,
        activation_fn: nn.Module = nn.GELU(),
        transformer_type: Literal["pyraformer", "decoder"] = "pyraformer",
        device: str = "auto",
    ):
        """
        Initialize the Actor-Critic Transformer model.

        Args:
            input_shape: Shape of input data (window_size, features)
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layers (must be divisible by n_heads)
            n_heads: Number of attention heads
            n_layers: Number of Transformer blocks (PyraFormer or Decoder)
            dropout: Dropout probability
            activation_fn: Activation function to use
            transformer_type: Type of transformer block ('pyraformer' or 'decoder')
            device: Device to run the model on ('cpu', 'cuda', 'mps', or 'auto')
        """
        super(ActorCriticTransformer, self).__init__()

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        window_size, n_features = input_shape

        # Ensure hidden_dim is divisible by n_heads
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        # Initial feature extraction with convolution
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Input embedding
        self.input_embedding = nn.Linear(hidden_dim, hidden_dim)

        # Time embedding
        self.time_embedding = TimeEmbedding(hidden_dim, max_len=window_size)

        # Transformer blocks (configurable type)
        self.transformer_blocks = nn.ModuleList()
        if transformer_type == "pyraformer":
            for _ in range(n_layers):
                self.transformer_blocks.append(
                    PyraFormerBlock(
                        hidden_dim=hidden_dim,
                        n_heads=n_heads,
                        dropout=dropout,
                        max_seq_len=window_size, # Pyraformer needs max_seq_len
                    )
                )
        elif transformer_type == "decoder":
            for _ in range(n_layers):
                self.transformer_blocks.append(
                    TransformerDecoderBlock(
                        hidden_dim=hidden_dim,
                        n_heads=n_heads,
                        dropout=dropout,
                        dim_feedforward=hidden_dim * 4, # Standard practice
                        activation_fn=activation_fn,
                    )
                )
        else:
            raise ValueError(f"Unknown transformer_type: {transformer_type}")

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Actor (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            activation_fn,
            nn.Linear(128, action_dim),
        )

        # Critic (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            activation_fn,
            nn.Linear(128, 1),
        )

        # Initialize weights
        self.apply(initialize_weights)

        # Move model to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, window_size, n_features)

        Returns:
            Tuple of (action_logits, values)
        """
        batch_size, window_size, n_features = x.shape

        # Apply feature extraction with convolution (need to transpose for Conv1d)
        x_conv = x.transpose(1, 2)  # (batch_size, n_features, window_size)
        x_conv = self.feature_extractor(x_conv)
        x = x_conv.transpose(1, 2)  # Back to (batch_size, window_size, hidden_dim)

        # Input embedding
        x = self.input_embedding(x)

        # Add time embedding
        time_indices = torch.arange(
            window_size, dtype=torch.float32, device=self.device
        )
        time_indices = time_indices.view(1, -1, 1).expand(batch_size, -1, -1)
        x = self.time_embedding(x, time_indices)

        # Apply Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Layer normalization
        x = self.layer_norm(x)

        # Get representations from the last time step
        last_output = x[:, -1]

        # Actor output (action logits)
        action_logits = self.actor(last_output)

        # Clip action logits to prevent extreme values
        # action_logits = torch.clamp(action_logits, min=-50.0, max=50.0)

        # Critic output (state value)
        values = self.critic(last_output)

        # Clip value outputs to prevent extreme values
        # values = torch.clamp(values, min=-100.0, max=100.0)

        return action_logits, values
