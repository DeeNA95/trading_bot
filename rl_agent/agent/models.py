"""
Neural network models for reinforcement learning agents.

This module contains the actor and critic network architectures used by RL agents.
"""

from typing import Dict, List, Optional, Tuple, Union, Literal, Type, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_xla.core.xla_model as xm
    _HAS_XLA = True
except ImportError:
    _HAS_XLA = False

from .attention.multi_head_attention import MultiHeadAttention
from .feedforward import FeedForward
from .embeddings import TimeEmbedding
from .blocks.encoder_block import EncoderBlock
from .blocks.decoder_block import DecoderBlock


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


class ActorCriticWrapper(nn.Module):
    """
    Wraps a core Transformer model (encoder-only, decoder-only, etc.)
    with input feature extraction and Actor/Critic heads.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int],
        action_dim: int,
        transformer_core: nn.Module, # Accepts an instantiated core model
        feature_extractor_hidden_dim: int = 128, # Dim for initial CNN
        embedding_dim: int = 256, # Dim expected by transformer_core
        activation_fn: nn.Module = nn.GELU(),
        dropout: float = 0.1, # Dropout for heads
        temperature: float = 1.0, # Temperature parameter
        device: str = "auto",
    ):
        super(ActorCriticWrapper, self).__init__()
        self.temperature = temperature

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif _HAS_XLA:
                self.device = xm.xla_device()
                print("Using TPU (XLA) device.")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)


        window_size, n_features = input_shape

        # --- Feature Extractor and Input Embedding ---
        # Adjust hidden dim if needed, ensure output matches embedding_dim
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_features, feature_extractor_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(feature_extractor_hidden_dim, embedding_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # No separate input_embedding linear layer needed if CNN outputs embedding_dim

        # --- Core Transformer Model ---
        self.transformer_core = transformer_core

        # --- Actor and Critic Heads ---
        # Input dimension for heads should match the output dim of transformer_core
        # Assuming transformer_core outputs embedding_dim
        self.actor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(128, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.apply(initialize_weights)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        batch_size, window_size, n_features = x.shape

        # Feature extraction
        x_conv = x.transpose(1, 2)
        x_conv = self.feature_extractor(x_conv)
        embedded_input = x_conv.transpose(1, 2) # Shape: (B, T, C=embedding_dim)

        # Pass through the core transformer model
        # The core model should handle time embedding internally
        # Masking logic might need to be handled based on core model type
        # For now, assume core handles its own masking or doesn't need it here
        core_output = self.transformer_core(embedded_input) # Shape: (B, T, C)

        # Use last time step output for actor/critic
        last_output = core_output[:, -1]

        action_logits = self.actor(last_output)
        values = self.critic(last_output)

        return action_logits, values
