"""
Neural network models for reinforcement learning agents.

This module contains the actor and critic network architectures used by RL agents.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        dropout: float = 0.2,
        device: str = "auto",
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
            dropout: Dropout probability for regularization
            device: Device to run the model on ('cpu', 'cuda', 'mps', or 'auto')
        """
        super(ActorCriticCNN, self).__init__()

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

        # Feature extraction layers - 1D CNNs
        self.cnn_layers = nn.ModuleList()

        # First CNN layer
        self.cnn_layers.append(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=n_filters[0],
                kernel_size=kernel_sizes[0],
                stride=1,
            )
        )
        self.cnn_layers.append(activation_fn)

        # Calculate output size after first convolution
        conv_output_size = window_size - kernel_sizes[0] + 1

        # Additional CNN layers
        for i in range(1, len(n_filters)):
            self.cnn_layers.append(
                nn.Conv1d(
                    in_channels=n_filters[i - 1],
                    out_channels=n_filters[i],
                    kernel_size=(
                        kernel_sizes[i] if i < len(kernel_sizes) else kernel_sizes[-1]
                    ),
                    stride=1,
                )
            )
            self.cnn_layers.append(activation_fn)

            # Update output size
            conv_output_size = (
                conv_output_size
                - (kernel_sizes[i] if i < len(kernel_sizes) else kernel_sizes[-1])
                + 1
            )

        # Flatten layer
        self.flatten = nn.Flatten()

        # Calculate flattened size
        flattened_size = n_filters[-1] * conv_output_size

        # Actor (policy network)
        self.actor = nn.Sequential(
            nn.Linear(flattened_size, 128),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(256, 57),
            activation_fn,
            nn.Linear(57, action_dim),
        )

        # Critic (value network)
        self.critic = nn.Sequential(
            nn.Linear(flattened_size, 128),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(256, 57),
            activation_fn,
            nn.Linear(57, 1),
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
        dropout: float = 0.2,
        activation_fn: nn.Module = nn.GELU(),
        device: str = "auto",
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

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,  # (batch, seq, feature)
        )

        # Actor (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(256, 57),
            activation_fn,
            nn.Linear(57, action_dim),
        )

        # Critic (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(256, 57),
            activation_fn,
            nn.Linear(57, 1),
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
            self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=self.device
        )
        c0 = torch.zeros(
            self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=self.device
        )

        self.hidden = (h0, c0)

    def forward(
        self, x: torch.Tensor, reset_hidden: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, window_size, n_features)
            reset_hidden: Whether to reset the hidden state

        Returns:
            Tuple of (action_logits, values)
        """
        batch_size = x.size(0)

        # Check for NaN values in input
        if torch.isnan(x).any():
            # Replace NaN with zeros
            x = torch.nan_to_num(x, nan=0.0)

        # Clip extreme values
        x = torch.clamp(x, min=-10.0, max=10.0)

        if self.hidden is None or reset_hidden or batch_size != self.hidden[0].size(1):
            self.reset_hidden_state(batch_size)

        # LSTM forward pass
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        # Check for NaN in LSTM output
        if torch.isnan(lstm_out).any():
            # This is a serious issue, reset hidden state and try with zeros
            self.reset_hidden_state(batch_size)
            lstm_out = torch.zeros_like(lstm_out)

        # Use only the last output for prediction
        last_output = lstm_out[:, -1]

        # Actor output (action logits)
        action_logits = self.actor(last_output)

        # Clip action logits to prevent extreme values
        action_logits = torch.clamp(action_logits, min=-50.0, max=50.0)

        # Critic output (state value)
        values = self.critic(last_output)

        # Clip value outputs
        values = torch.clamp(values, min=-100.0, max=100.0)

        return action_logits, values


class PyramidalAttention(nn.Module):
    """
    Pyramidal Attention module for time series data.

    This implements log-sparse attention patterns at multiple time scales
    for efficient processing of long sequences with O(n log n) complexity.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        """
        Initialize the Pyramidal Attention module.

        Args:
            hidden_dim: Size of the hidden dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_len: Maximum sequence length supported
        """
        super(PyramidalAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Ensure hidden_dim is divisible by n_heads
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        # Multi-scale attention layers with different dilation rates
        # 2^0, 2^1, 2^2, etc.
        self.dilations = [2**i for i in range(min(5, n_heads))]

        # Projections for Q, K, V for each attention head
        self.query_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, self.head_dim) for _ in range(n_heads)]
        )
        self.key_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, self.head_dim) for _ in range(n_heads)]
        )
        self.value_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, self.head_dim) for _ in range(n_heads)]
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Pre-compute log-sparse attention masks for different dilations
        self.register_buffer("masks", self._generate_masks(max_seq_len))

    def _generate_masks(self, max_seq_len: int) -> torch.Tensor:
        """
        Generate log-sparse attention masks for different dilation rates.

        Args:
            max_seq_len: Maximum sequence length

        Returns:
            Tensor of shape (n_heads, max_seq_len, max_seq_len) with masks
        """
        masks = torch.zeros(len(self.dilations), max_seq_len, max_seq_len)

        for h, dilation in enumerate(self.dilations):
            for i in range(max_seq_len):
                # For each position, attend to:
                # - immediate neighbors based on dilation
                # - log-sparse positions (powers of 2 distance away)
                for j in range(max_seq_len):
                    # Immediate neighborhood attention based on dilation
                    if abs(i - j) <= dilation:
                        masks[h, i, j] = 1.0
                    # Log-sparse attention (powers of 2)
                    elif abs(i - j) % (2 ** (1 + h)) == 0:
                        masks[h, i, j] = 1.0

        return masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pyramidal attention module.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Attended tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Use pre-computed masks or dynamically generate for longer sequences
        if seq_len > self.masks.shape[1]:
            attention_masks = self._generate_masks(seq_len).to(device)
        else:
            attention_masks = self.masks[:, :seq_len, :seq_len]

        # Multi-head attention with different dilations
        head_outputs = []

        for h in range(self.n_heads):
            q = self.query_projections[h](x)
            k = self.key_projections[h](x)
            v = self.value_projections[h](x)

            # Compute attention scores
            attention_scores = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim**0.5)

            # Apply log-sparse mask
            mask = attention_masks[min(h, len(self.dilations) - 1)]
            attention_scores = attention_scores.masked_fill(
                mask.expand(batch_size, seq_len, seq_len) == 0, float("-inf")
            )

            # Apply softmax and dropout
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)

            # Apply attention to values
            head_output = torch.bmm(attention_probs, v)
            head_outputs.append(head_output)

        # Concatenate heads and project
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.output_projection(concatenated)

        return output


class PyraFormerBlock(nn.Module):
    """
    PyraFormer block with feed-forward network and layer normalization.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        """
        Initialize the PyraFormer block.

        Args:
            hidden_dim: Size of the hidden dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_len: Maximum sequence length supported
        """
        super(PyraFormerBlock, self).__init__()

        # Pyramidal attention
        self.attention = PyramidalAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PyraFormer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Multi-head attention with residual connection and layer norm
        attended = self.attention(self.norm1(x))
        x = x + self.dropout(attended)

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x


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
    Actor-Critic model with PyraFormer architecture for processing financial time series.

    This model implements a PyraFormer, which uses a pyramidal attention mechanism
    with log-sparse patterns that efficiently captures multi-scale temporal dependencies
    in time series data.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, features)
        action_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.2,
        activation_fn: nn.Module = nn.GELU(),
        device: str = "auto",
        force_features: int = None,  # Force specific feature count
    ):
        """
        Initialize the Actor-Critic PyraFormer model.

        Args:
            input_shape: Shape of input data (window_size, features)
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layers (must be divisible by n_heads)
            n_heads: Number of attention heads
            n_layers: Number of PyraFormer blocks
            dropout: Dropout probability
            activation_fn: Activation function to use
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

        # PyraFormer blocks
        self.pyraformer_blocks = nn.ModuleList(
            [
                PyraFormerBlock(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    max_seq_len=window_size,
                )
                for _ in range(n_layers)
            ]
        )

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
            nn.Linear(256, 57),
            activation_fn,
            nn.Linear(57, action_dim),
        )

        # Critic (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(256, 57),
            activation_fn,
            nn.Linear(57, 1),
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

        # Apply PyraFormer blocks
        for block in self.pyraformer_blocks:
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
