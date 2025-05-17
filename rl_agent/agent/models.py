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
from .feature_extractors import create_feature_extractor


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
        core_model: nn.Module, # Accepts an instantiated core model (Transformer or LSTM)
        feature_extractor_hidden_dim: int = 128, # Dim for initial CNN
        embedding_dim: int = 256, # Dim expected as output from core_model and input to heads
        activation_fn: nn.Module = nn.GELU(),
        dropout: float = 0.1, # Dropout for heads
        temperature: float = 1.0, # Temperature parameter
        device: str = "auto",
        # Feature extractor parameters
        feature_extractor_type: str = "basic",
        feature_extractor_layers: int = 2,
        use_skip_connections: bool = False,
        use_layer_norm: bool = False,
        use_instance_norm: bool = False,
        feature_dropout: float = 0.0,
        # Actor-Critic head parameters
        head_hidden_dim: int = 128,
        head_n_layers: int = 2,
        head_use_layer_norm: bool = False,
        head_use_residual: bool = False,
        head_activation_fn: Optional[nn.Module] = None,
        head_dropout: Optional[float] = None,
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
        # Create feature extractor using the factory function
        self.feature_extractor = create_feature_extractor(
            extractor_type=feature_extractor_type,
            in_channels=n_features,
            hidden_dim=feature_extractor_hidden_dim,
            out_channels=embedding_dim,
            num_layers=feature_extractor_layers,
            use_skip_connections=use_skip_connections,
            use_layer_norm=use_layer_norm,
            use_instance_norm=use_instance_norm,
            dropout=feature_dropout,
        )

        # --- Core Model (Transformer or LSTM) ---
        self.core_model = core_model

        # --- Actor and Critic Heads ---
        # Use provided head parameters or defaults
        head_dropout_val = head_dropout if head_dropout is not None else dropout
        head_activation = head_activation_fn if head_activation_fn is not None else activation_fn

        # Build enhanced actor and critic networks
        self.actor = self._build_mlp_head(
            input_dim=embedding_dim,
            hidden_dim=head_hidden_dim,
            output_dim=action_dim,
            n_layers=head_n_layers,
            activation_fn=head_activation,
            dropout=head_dropout_val,
            use_layer_norm=head_use_layer_norm,
            use_residual=head_use_residual
        )

        self.critic = self._build_mlp_head(
            input_dim=embedding_dim,
            hidden_dim=head_hidden_dim,
            output_dim=1,
            n_layers=head_n_layers,
            activation_fn=head_activation,
            dropout=head_dropout_val,
            use_layer_norm=head_use_layer_norm,
            use_residual=head_use_residual
        )

        self.apply(initialize_weights)
        self.to(self.device)

    def _build_mlp_head(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        activation_fn: nn.Module,
        dropout: float,
        use_layer_norm: bool,
        use_residual: bool
    ) -> nn.Module:
        """
        Build an enhanced MLP head with optional layer normalization and residual connections.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            n_layers: Number of hidden layers
            activation_fn: Activation function
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections

        Returns:
            nn.Module: The constructed MLP head
        """
        if n_layers < 1:
            # If no hidden layers, just return a single linear layer
            return nn.Linear(input_dim, output_dim)

        layers = []

        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation_fn)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Middle layers: hidden_dim -> hidden_dim with optional residual connections
        for i in range(n_layers - 1):
            if use_residual:
                # Create a residual block
                res_block = []
                res_block.append(nn.Linear(hidden_dim, hidden_dim))
                if use_layer_norm:
                    res_block.append(nn.LayerNorm(hidden_dim))
                res_block.append(activation_fn)
                if dropout > 0:
                    res_block.append(nn.Dropout(dropout))
                res_block.append(nn.Linear(hidden_dim, hidden_dim))
                if use_layer_norm:
                    res_block.append(nn.LayerNorm(hidden_dim))

                # Wrap the residual block in a nn.Sequential
                res_sequential = nn.Sequential(*res_block)

                # Create a residual wrapper that adds the input to the output
                class ResidualWrapper(nn.Module):
                    def __init__(self, module):
                        super().__init__()
                        self.module = module
                        self.activation = activation_fn

                    def forward(self, x):
                        return self.activation(x + self.module(x))

                # Add the wrapped residual block
                layers.append(ResidualWrapper(res_sequential))
            else:
                # Standard feedforward layer
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(activation_fn)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        # Output layer: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        # Extract shape but use _ for unused variables to avoid IDE warnings
        _, _, _ = x.shape

        # Feature extraction
        x_conv = x.transpose(1, 2)
        x_conv = self.feature_extractor(x_conv)
        embedded_input = x_conv.transpose(1, 2) # Shape: (B, T, C=embedding_dim)

        # Pass through the core transformer model
        # The core model should handle time embedding internally
        # Masking logic might need to be handled based on core model type
        # For now, assume core handles its own masking or doesn't need it here
        # The core_model could be a Transformer or an LSTM.
        # Both are expected to take (B, T, Features) and output (B, T, HiddenDim)
        core_output = self.core_model(embedded_input) # Shape: (B, T, C)

        # Use last time step output for actor/critic
        last_output = core_output[:, -1]

        action_logits = self.actor(last_output)
        values = self.critic(last_output)

        return action_logits, values
