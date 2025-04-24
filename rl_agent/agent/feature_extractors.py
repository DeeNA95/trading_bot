"""
Feature extraction modules for reinforcement learning agents.

This module contains various feature extraction architectures for processing
time series data in RL agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any


class BasicConvExtractor(nn.Module):
    """
    Basic convolutional feature extractor with optional normalization.

    A simple stack of 1D convolutional layers with activation and normalization.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_instance_norm: bool = False,
    ):
        """
        Initialize the BasicConvExtractor.

        Args:
            in_channels: Number of input channels (features)
            hidden_dim: Hidden dimension size
            out_channels: Output channels (embedding dimension)
            num_layers: Number of convolutional layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
            use_layer_norm: Whether to use LayerNorm instead of BatchNorm
            use_instance_norm: Whether to use InstanceNorm
        """
        super().__init__()

        assert num_layers >= 1, "Must have at least one layer"

        self.layers = nn.ModuleList()

        # First layer: in_channels -> hidden_dim
        self.layers.append(
            self._make_conv_block(
                in_channels,
                hidden_dim,
                kernel_size,
                dropout,
                use_layer_norm,
                use_instance_norm
            )
        )

        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(
                self._make_conv_block(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    dropout,
                    use_layer_norm,
                    use_instance_norm
                )
            )

        # Last layer: hidden_dim -> out_channels (if more than 1 layer)
        if num_layers > 1:
            self.layers.append(
                self._make_conv_block(
                    hidden_dim,
                    out_channels,
                    kernel_size,
                    dropout,
                    use_layer_norm,
                    use_instance_norm
                )
            )

    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
        use_layer_norm: bool,
        use_instance_norm: bool,
    ) -> nn.Sequential:
        """Create a convolutional block with normalization and activation."""
        padding = kernel_size // 2  # Same padding

        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GELU(),
        ]

        # Add normalization
        if use_layer_norm:
            # Custom layer norm for Conv1d (normalizes over last 2 dims)
            class Conv1dLayerNorm(nn.Module):
                def __init__(self, channels):
                    super().__init__()
                    self.norm = nn.LayerNorm([channels, 1])

                def forward(self, x):
                    # x shape: [batch, channels, seq_len]
                    # Transpose to [batch, seq_len, channels] for LayerNorm
                    x = x.transpose(1, 2)
                    # Apply LayerNorm over the channel dimension
                    x = nn.functional.layer_norm(x, [x.size(-1)])
                    # Transpose back to [batch, channels, seq_len]
                    x = x.transpose(1, 2)
                    return x

            layers.append(Conv1dLayerNorm(out_channels))
        elif use_instance_norm:
            layers.append(nn.InstanceNorm1d(out_channels))
        else:
            layers.append(nn.BatchNorm1d(out_channels))

        # Add dropout if specified
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)

        Returns:
            Tensor of shape (batch_size, out_channels, seq_len)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block with skip connections.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_instance_norm: bool = False,
    ):
        """
        Initialize the ResidualConvBlock.

        Args:
            channels: Number of channels (same for input and output)
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
            use_layer_norm: Whether to use LayerNorm instead of BatchNorm
            use_instance_norm: Whether to use InstanceNorm
        """
        super().__init__()

        padding = kernel_size // 2  # Same padding

        # First conv layer
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)

        # Custom layer norm for Conv1d
        class Conv1dLayerNorm(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.norm = nn.LayerNorm([channels, 1])

            def forward(self, x):
                # x shape: [batch, channels, seq_len]
                # Transpose to [batch, seq_len, channels] for LayerNorm
                x = x.transpose(1, 2)
                # Apply LayerNorm over the channel dimension
                x = nn.functional.layer_norm(x, [x.size(-1)])
                # Transpose back to [batch, channels, seq_len]
                x = x.transpose(1, 2)
                return x

        # Normalization layer 1
        if use_layer_norm:
            self.norm1 = Conv1dLayerNorm(channels)
        elif use_instance_norm:
            self.norm1 = nn.InstanceNorm1d(channels)
        else:
            self.norm1 = nn.BatchNorm1d(channels)

        # Second conv layer
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)

        # Normalization layer 2
        if use_layer_norm:
            self.norm2 = Conv1dLayerNorm(channels)
        elif use_instance_norm:
            self.norm2 = nn.InstanceNorm1d(channels)
        else:
            self.norm2 = nn.BatchNorm1d(channels)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x: Input tensor of shape (batch_size, channels, seq_len)

        Returns:
            Tensor of shape (batch_size, channels, seq_len)
        """
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)

        # Skip connection
        out += identity
        out = self.activation(out)

        return out


class ResNetExtractor(nn.Module):
    """
    ResNet-style feature extractor with residual connections.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_instance_norm: bool = False,
    ):
        """
        Initialize the ResNetExtractor.

        Args:
            in_channels: Number of input channels (features)
            hidden_dim: Hidden dimension size
            out_channels: Output channels (embedding dimension)
            num_layers: Number of residual blocks
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
            use_layer_norm: Whether to use LayerNorm instead of BatchNorm
            use_instance_norm: Whether to use InstanceNorm
        """
        super().__init__()

        # Initial projection to hidden_dim
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualConvBlock(
                hidden_dim,
                kernel_size,
                dropout,
                use_layer_norm,
                use_instance_norm
            )
            for _ in range(num_layers)
        ])

        # Final projection to out_channels
        self.output_proj = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)

        Returns:
            Tensor of shape (batch_size, out_channels, seq_len)
        """
        x = self.input_proj(x)

        for block in self.res_blocks:
            x = block(x)

        x = self.output_proj(x)

        return x


class InceptionBlock(nn.Module):
    """
    Inception-style block with multiple kernel sizes in parallel.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [1, 3, 5],
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_instance_norm: bool = False,
    ):
        """
        Initialize the InceptionBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels per branch
            kernel_sizes: List of kernel sizes for parallel branches
            dropout: Dropout rate
            use_layer_norm: Whether to use LayerNorm instead of BatchNorm
            use_instance_norm: Whether to use InstanceNorm
        """
        super().__init__()

        self.branches = nn.ModuleList()

        # Create parallel branches with different kernel sizes
        for k_size in kernel_sizes:
            padding = k_size // 2  # Same padding

            branch = [
                nn.Conv1d(in_channels, out_channels, k_size, padding=padding),
                nn.GELU(),
            ]

            # Custom layer norm for Conv1d
            class Conv1dLayerNorm(nn.Module):
                def __init__(self, channels):
                    super().__init__()
                    self.norm = nn.LayerNorm([channels, 1])

                def forward(self, x):
                    # x shape: [batch, channels, seq_len]
                    # Transpose to [batch, seq_len, channels] for LayerNorm
                    x = x.transpose(1, 2)
                    # Apply LayerNorm over the channel dimension
                    x = nn.functional.layer_norm(x, [x.size(-1)])
                    # Transpose back to [batch, channels, seq_len]
                    x = x.transpose(1, 2)
                    return x

            # Add normalization
            if use_layer_norm:
                branch.append(Conv1dLayerNorm(out_channels))
            elif use_instance_norm:
                branch.append(nn.InstanceNorm1d(out_channels))
            else:
                branch.append(nn.BatchNorm1d(out_channels))

            # Add dropout if specified
            if dropout > 0:
                branch.append(nn.Dropout(dropout))

            self.branches.append(nn.Sequential(*branch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the inception block.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)

        Returns:
            Tensor of shape (batch_size, out_channels * len(kernel_sizes), seq_len)
        """
        # Process each branch
        branch_outputs = [branch(x) for branch in self.branches]

        # Concatenate along channel dimension
        return torch.cat(branch_outputs, dim=1)


class InceptionExtractor(nn.Module):
    """
    Inception-style feature extractor with multi-scale processing.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int = 2,
        kernel_sizes: List[int] = [1, 3, 5],
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_instance_norm: bool = False,
    ):
        """
        Initialize the InceptionExtractor.

        Args:
            in_channels: Number of input channels (features)
            hidden_dim: Hidden dimension size per branch
            out_channels: Output channels (embedding dimension)
            num_layers: Number of inception blocks
            kernel_sizes: List of kernel sizes for parallel branches
            dropout: Dropout rate
            use_layer_norm: Whether to use LayerNorm instead of BatchNorm
            use_instance_norm: Whether to use InstanceNorm
        """
        super().__init__()

        # Calculate dimensions
        branch_dim = hidden_dim // len(kernel_sizes)
        inception_out_dim = branch_dim * len(kernel_sizes)

        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
        )

        # Inception blocks
        self.blocks = nn.ModuleList()

        # First block: hidden_dim -> inception_out_dim
        self.blocks.append(
            InceptionBlock(
                hidden_dim,
                branch_dim,
                kernel_sizes,
                dropout,
                use_layer_norm,
                use_instance_norm
            )
        )

        # Middle blocks: inception_out_dim -> inception_out_dim
        for _ in range(num_layers - 1):
            self.blocks.append(
                InceptionBlock(
                    inception_out_dim,
                    branch_dim,
                    kernel_sizes,
                    dropout,
                    use_layer_norm,
                    use_instance_norm
                )
            )

        # Final projection to out_channels
        self.output_proj = nn.Conv1d(inception_out_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)

        Returns:
            Tensor of shape (batch_size, out_channels, seq_len)
        """
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_proj(x)

        return x


def create_feature_extractor(
    extractor_type: str,
    in_channels: int,
    hidden_dim: int,
    out_channels: int,
    num_layers: int = 2,
    use_skip_connections: bool = False,
    use_layer_norm: bool = False,
    use_instance_norm: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Factory function to create a feature extractor based on the specified type.

    Args:
        extractor_type: Type of feature extractor ('basic', 'resnet', 'inception')
        in_channels: Number of input channels (features)
        hidden_dim: Hidden dimension size
        out_channels: Output channels (embedding dimension)
        num_layers: Number of layers/blocks
        use_skip_connections: Whether to use skip connections (forces 'resnet' type)
        use_layer_norm: Whether to use LayerNorm instead of BatchNorm
        use_instance_norm: Whether to use InstanceNorm
        dropout: Dropout rate

    Returns:
        Feature extractor module
    """
    # Force ResNet if skip connections are requested
    if use_skip_connections and extractor_type != 'resnet':
        extractor_type = 'resnet'

    if extractor_type == 'basic':
        return BasicConvExtractor(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_instance_norm=use_instance_norm,
        )
    elif extractor_type == 'resnet':
        return ResNetExtractor(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_instance_norm=use_instance_norm,
        )
    elif extractor_type == 'inception':
        return InceptionExtractor(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_instance_norm=use_instance_norm,
        )
    else:
        raise ValueError(f"Unknown feature extractor type: {extractor_type}")
