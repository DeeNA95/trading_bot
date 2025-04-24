"""
Test script for the enhanced feature extractors.
"""

import torch
import torch.nn as nn
from rl_agent.agent.feature_extractors import (
    BasicConvExtractor,
    ResNetExtractor,
    InceptionExtractor,
    create_feature_extractor
)

def test_feature_extractors():
    """
    Test the different feature extractor implementations.
    """
    # Test parameters
    batch_size = 8
    seq_len = 60
    in_channels = 59
    hidden_dim = 128
    out_channels = 256
    
    # Create a random input tensor
    x = torch.randn(batch_size, in_channels, seq_len)
    
    # Test BasicConvExtractor
    print("\nTesting BasicConvExtractor...")
    basic_extractor = BasicConvExtractor(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        num_layers=2,
        dropout=0.1
    )
    basic_output = basic_extractor(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {basic_output.shape}")
    print(f"Expected output shape: (batch_size={batch_size}, out_channels={out_channels}, seq_len={seq_len})")
    
    # Test ResNetExtractor
    print("\nTesting ResNetExtractor...")
    resnet_extractor = ResNetExtractor(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        num_layers=2,
        dropout=0.1
    )
    resnet_output = resnet_extractor(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {resnet_output.shape}")
    print(f"Expected output shape: (batch_size={batch_size}, out_channels={out_channels}, seq_len={seq_len})")
    
    # Test InceptionExtractor
    print("\nTesting InceptionExtractor...")
    inception_extractor = InceptionExtractor(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        num_layers=2,
        dropout=0.1
    )
    inception_output = inception_extractor(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {inception_output.shape}")
    print(f"Expected output shape: (batch_size={batch_size}, out_channels={out_channels}, seq_len={seq_len})")
    
    # Test factory function
    print("\nTesting create_feature_extractor factory function...")
    for extractor_type in ["basic", "resnet", "inception"]:
        print(f"\nCreating {extractor_type} extractor...")
        extractor = create_feature_extractor(
            extractor_type=extractor_type,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            num_layers=2,
            use_skip_connections=(extractor_type == "resnet"),
            use_layer_norm=True,
            dropout=0.1
        )
        output = extractor(x)
        print(f"Output shape: {output.shape}")
    
    print("\nAll feature extractors tested successfully!")

if __name__ == "__main__":
    test_feature_extractors()
