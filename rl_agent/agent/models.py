"""
Neural network models for reinforcement learning agents.

This module contains the actor and critic network architectures used by RL agents.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List, Union


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
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param, std)
            elif 'bias' in name:
                nn.init.zeros_(param)



class ActorCriticCNN(nn.Module):
    """
    Actor-Critic model with CNN layers for processing time series data like OHLCV.

    The model uses convolutional layers to extract features from the input, followed
    by separate fully connected layers for the actor (policy) and critic (value).
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, features)
        action_dim: int,
        hidden_dim: int = 128,
        n_filters: List[int] = [64, 128],
        kernel_sizes: List[int] = [3, 2],
        activation_fn: nn.Module = nn.GELU(),
        device: str = 'auto'
    ):
        """
        Initialize the Actor-Critic CNN model.

        Args:
            input_shape: Shape of input data (window_size, features)
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layers
            n_filters: Number of filters in each CNN layer
            kernel_sizes: Kernel sizes for CNN layers
            activation_fn: Activation function to use
            device: Device to run the model on ('cpu', 'cuda', 'mps', or 'auto')
        """
        super(ActorCriticCNN, self).__init__()

        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        window_size, n_features = input_shape

        # Feature extraction layers - 1D CNNs
        self.cnn_layers = nn.ModuleList()

        # First CNN layer
        self.cnn_layers.append(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=n_filters[0],
                kernel_size=kernel_sizes[0],
                stride=1
            )
        )
        self.cnn_layers.append(activation_fn)

        # Calculate output size after first convolution
        conv_output_size = window_size - kernel_sizes[0] + 1

        # Additional CNN layers
        for i in range(1, len(n_filters)):
            self.cnn_layers.append(
                nn.Conv1d(
                    in_channels=n_filters[i-1],
                    out_channels=n_filters[i],
                    kernel_size=kernel_sizes[i] if i < len(kernel_sizes) else kernel_sizes[-1],
                    stride=1
                )
            )
            self.cnn_layers.append(activation_fn)

            # Update output size
            conv_output_size = conv_output_size - (kernel_sizes[i] if i < len(kernel_sizes) else kernel_sizes[-1]) + 1

        # Flatten layer
        self.flatten = nn.Flatten()

        # Calculate flattened size
        flattened_size = n_filters[-1] * conv_output_size

        # Actor (policy network)
        self.actor = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic (value network)
        self.critic = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, 1)
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
        # Transpose to shape (batch_size, n_features, window_size) for CNN
        x = x.permute(0, 2, 1)

        # Pass through CNN layers
        for layer in self.cnn_layers:
            x = layer(x)

        # Flatten
        x = self.flatten(x)

        # Actor output (action logits)
        action_logits = self.actor(x)

        # Critic output (state value)
        values = self.critic(x)

        return action_logits, values


class ActorCriticLSTM(nn.Module):
    """
    Actor-Critic model with LSTM layers for processing sequential financial data.

    The model uses LSTM layers to process time series data, followed by separate fully
    connected layers for the actor (policy) and critic (value).
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, features)
        action_dim: int,
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        activation_fn: nn.Module = nn.GELU(),
        device: str = 'auto'
    ):
        """
        Initialize the Actor-Critic LSTM model.

        Args:
            input_shape: Shape of input data (window_size, features)
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layers
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            activation_fn: Activation function to use
            device: Device to run the model on ('cpu', 'cuda', 'mps', or 'auto')
        """
        super(ActorCriticLSTM, self).__init__()

        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        window_size, n_features = input_shape

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True  # (batch, seq, feature)
        )

        # Actor (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, 1)
        )

        # Hidden state for LSTM
        self.hidden = None

        # Initialize weights
        self.apply(initialize_weights)

        # Move model to device
        self.to(self.device)

    def reset_hidden_state(self, batch_size: int = 1) -> None:
        """
        Reset the hidden state of the LSTM.

        Args:
            batch_size: Batch size for the hidden state
        """
        # (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(
            self.lstm.num_layers,
            batch_size,
            self.lstm.hidden_size,
            device=self.device
        )
        c0 = torch.zeros(
            self.lstm.num_layers,
            batch_size,
            self.lstm.hidden_size,
            device=self.device
        )

        self.hidden = (h0, c0)

    def forward(self, x: torch.Tensor, reset_hidden: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, window_size, n_features)
            reset_hidden: Whether to reset the hidden state

        Returns:
            Tuple of (action_logits, values)
        """
        batch_size = x.size(0)

        if self.hidden is None or reset_hidden or batch_size != self.hidden[0].size(1):
            self.reset_hidden_state(batch_size)

        # LSTM forward pass
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        # Use only the last output for prediction
        last_output = lstm_out[:, -1]

        # Actor output (action logits)
        action_logits = self.actor(last_output)

        # Critic output (state value)
        values = self.critic(last_output)

        return action_logits, values


class ActorCriticTransformer(nn.Module):
    """
    Actor-Critic model with Transformer architecture for processing financial time series.

    The model uses transformer encoder layers to process time series data, followed by
    separate fully connected layers for the actor (policy) and critic (value).
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, features)
        action_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        activation_fn: nn.Module = nn.GELU(),
        device: str = 'auto'
    ):
        """
        Initialize the Actor-Critic Transformer model.

        Args:
            input_shape: Shape of input data (window_size, features)
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layers (must be divisible by n_heads)
            n_heads: Number of attention heads
            n_layers: Number of transformer encoder layers
            dropout: Dropout probability
            activation_fn: Activation function to use
            device: Device to run the model on ('cpu', 'cuda', 'mps', or 'auto')
        """
        super(ActorCriticTransformer, self).__init__()

        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        window_size, n_features = input_shape

        # Ensure hidden_dim is divisible by n_heads
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        # Input embedding
        self.input_embedding = nn.Linear(n_features, hidden_dim)

        # Positional encoding
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, window_size, hidden_dim),
            requires_grad=True
        )

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=n_layers
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Actor (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, 1)
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
        batch_size, seq_len = x.size(0), x.size(1)

        # Input embedding
        x = self.input_embedding(x)

        # Add positional encoding
        x = x + self.pos_encoder

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Layer normalization
        x = self.layer_norm(x)

        # Use the last output for prediction
        last_output = x[:, -1]

        # Actor output (action logits)
        action_logits = self.actor(last_output)

        # Critic output (state value)
        values = self.critic(last_output)

        return action_logits, values
