from typing import Dict, Any, Optional, Type, Union
import torch
import torch.nn as nn
from dataclasses import dataclass
from rl_agent.agent.embeddings import TimeEmbedding # Added
from rl_agent.agent.blocks.encoder_block import EncoderBlock
from rl_agent.agent.blocks.decoder_block import DecoderBlock
from rl_agent.agent.attention.multi_head_attention import MultiHeadAttention
from rl_agent.agent.attention.multi_latent_attention import MultiLatentAttention
from rl_agent.agent.attention.pyramidal_attention import PyramidalAttention
from rl_agent.agent.attention.multi_query_attention import MultiQueryAttention
from rl_agent.agent.attention.grouped_query_attention import GroupedQueryAttention
from rl_agent.agent.feedforward import FeedForward, MixtureOfExperts
from rl_agent.agent.models import ActorCriticWrapper
from rl_agent.agent.lstm_core import LSTMCore # Added for LSTM support

@dataclass
class ModelConfig:
    # Core model selection
    core_model_type: str = "transformer" # "transformer" or "lstm"

    # Architecture type (for Transformer)
    architecture: str = "encoder_only"  # encoder_only, encoder_decoder, decoder_only

    # Core Transformer model dimensions
    embedding_dim: int = 256
    n_encoder_layers: int = 4 # Used for encoder_only, encoder_decoder
    n_decoder_layers: int = 4 # Used for encoder_decoder, decoder_only
    n_heads: int = 8
    window_size: int = 100
    dropout: float = 0.1

    # Attention configuration
    attention_type: str = "mha"  # mha, mla, pyr, mqn, gqa
    n_latents: Optional[int] = None  # For MLA
    n_groups: Optional[int] = None   # For GQA

    # Feed-forward configuration
    ffn_type: str = "standard"  # standard, moe
    ffn_dim: Optional[int] = None
    n_experts: Optional[int] = None  # For MoE
    top_k: Optional[int] = None      # For MoE

    # Normalization
    norm_type: str = "layer_norm"

    # Residual connection configuration
    residual_scale: float = 1.0      # Scaling factor for residual connections
    use_gated_residual: bool = False # Whether to use learnable gates for residual connections
    use_final_norm: bool = False     # Whether to apply a final layer normalization

    # Feature Extraction Configuration
    feature_extractor_type: str = "basic"  # basic, resnet, inception
    feature_extractor_dim: int = 128       # Hidden dimension for feature extractor
    feature_extractor_layers: int = 2      # Number of layers in feature extractor
    use_skip_connections: bool = False     # Whether to use skip connections in feature extractor
    use_layer_norm: bool = False           # Whether to use layer normalization instead of batch norm
    use_instance_norm: bool = False        # Whether to use instance normalization
    feature_dropout: float = 0.0           # Dropout rate for feature extractor

    # Actor-Critic Head Configuration
    head_hidden_dim: int = 128             # Hidden dimension for actor-critic heads
    head_n_layers: int = 2                 # Number of layers in actor-critic heads
    head_use_layer_norm: bool = False      # Whether to use layer normalization in actor-critic heads
    head_use_residual: bool = False        # Whether to use residual connections in actor-critic heads
    head_dropout: Optional[float] = None   # Dropout rate for actor-critic heads (None = use model dropout)

    # ActorCritic specific
    action_dim: int = 3  # HOLD, LONG, SHORT
    n_features: int = 59 # Number of input features from data
    temperature: float = 0.5  # Temperature for action selection (lower = more deterministic)

    # LSTM Core Configuration (if core_model_type is "lstm")
    lstm_hidden_dim: Optional[int] = 128
    lstm_num_layers: Optional[int] = 2
    lstm_dropout: Optional[float] = 0.0


# --- Dynamically Assembled Transformer Core ---
class DynamicTransformerCore(nn.Module):
    """
    A transformer core dynamically assembled from blocks based on config.
    Handles encoder-only, decoder-only, and encoder-decoder architectures.
    """
    def __init__(self,
                 architecture: str,
                 # Encoder components (Optional)
                 encoder_time_embedding: Optional[nn.Module],
                 encoder_blocks: Optional[nn.ModuleList],
                 encoder_norm: Optional[nn.Module],
                 # Decoder components (Optional)
                 decoder_time_embedding: Optional[nn.Module],
                 decoder_blocks: Optional[nn.ModuleList],
                 decoder_norm: Optional[nn.Module],
                 # Shared
                 dropout_emb: nn.Module,
                 window_size: int,
                 use_causal_mask_encoder: bool = False,
                 use_causal_mask_decoder: bool = True):
        super().__init__()
        self.architecture = architecture
        self.encoder_time_embedding = encoder_time_embedding
        self.encoder_blocks = encoder_blocks
        self.encoder_norm = encoder_norm
        self.decoder_time_embedding = decoder_time_embedding
        self.decoder_blocks = decoder_blocks
        self.decoder_norm = decoder_norm
        self.dropout_emb = dropout_emb
        self.window_size = window_size
        self.use_causal_mask_encoder = use_causal_mask_encoder
        self.use_causal_mask_decoder = use_causal_mask_decoder # Primarily for decoder self-attn

        # Validation
        if architecture == "encoder_only" and (encoder_blocks is None or encoder_norm is None or encoder_time_embedding is None):
             raise ValueError("Encoder components must be provided for encoder_only architecture.")
        if architecture == "decoder_only" and (decoder_blocks is None or decoder_norm is None or decoder_time_embedding is None):
             raise ValueError("Decoder components must be provided for decoder_only architecture.")
        if architecture == "encoder_decoder" and (encoder_blocks is None or encoder_norm is None or encoder_time_embedding is None or
                                                  decoder_blocks is None or decoder_norm is None or decoder_time_embedding is None):
             raise ValueError("Both Encoder and Decoder components must be provided for encoder-decoder architecture.")


    def _generate_causal_mask(self, size: int, device) -> torch.Tensor:
         mask = torch.tril(torch.ones(size, size, device=device)).bool()
         return mask

    def forward(self, src: Optional[torch.Tensor] = None, tgt: Optional[torch.Tensor] = None, src_padding_mask: Optional[torch.Tensor] = None, tgt_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.architecture == "encoder_only":
            if src is None: raise ValueError("Input `src` must be provided for encoder_only.")
            # --- Encoder Processing ---
            _, seq_len_src, _ = src.shape  # We don't need batch_size_src
            assert seq_len_src <= self.window_size, f"Encoder input sequence {seq_len_src} longer than window size {self.window_size}"
            time_indices_src = torch.arange(seq_len_src, dtype=torch.long, device=src.device)
            memory = self.encoder_time_embedding(src, time_indices_src)
            memory = self.dropout_emb(memory)
            encoder_self_mask = None
            if self.use_causal_mask_encoder: encoder_self_mask = self._generate_causal_mask(seq_len_src, src.device)
            elif src_padding_mask is not None: encoder_self_mask = src_padding_mask
            for block in self.encoder_blocks:
                memory = block(memory, src_mask=encoder_self_mask)
            memory = self.encoder_norm(memory)
            return memory

        elif self.architecture == "decoder_only":
            # For decoder-only models, use src as tgt if tgt is not provided
            input_tensor = tgt if tgt is not None else src
            if input_tensor is None:
                raise ValueError("Either `src` or `tgt` must be provided for decoder_only.")

            # --- Decoder Processing ---
            _, seq_len_input, _ = input_tensor.shape
            assert seq_len_input <= self.window_size, f"Decoder input sequence {seq_len_input} longer than window size {self.window_size}"
            time_indices_input = torch.arange(seq_len_input, dtype=torch.long, device=input_tensor.device)
            output = self.decoder_time_embedding(input_tensor, time_indices_input)
            output = self.dropout_emb(output)
            decoder_self_mask = None
            if self.use_causal_mask_decoder: decoder_self_mask = self._generate_causal_mask(seq_len_input, input_tensor.device)
            # TODO: Combine causal and padding if needed
            # if tgt_padding_mask is not None: ...

            # Process through decoder blocks
            for block in self.decoder_blocks:
                output = block(
                    input_tensor=output,  # Use input_tensor instead of tgt
                    memory=None,
                    tgt_mask=decoder_self_mask,
                    memory_mask=None
                )

            output = self.decoder_norm(output)
            return output

        elif self.architecture == "encoder_decoder":
            if src is None or tgt is None: raise ValueError("Inputs `src` and `tgt` must be provided for encoder_decoder.")
            # --- Encoder Processing ---
            _, seq_len_src, _ = src.shape  # We don't need batch_size_src
            assert seq_len_src <= self.window_size, f"Encoder input sequence {seq_len_src} longer than window size {self.window_size}"
            time_indices_src = torch.arange(seq_len_src, dtype=torch.long, device=src.device)
            memory = self.encoder_time_embedding(src, time_indices_src)
            memory = self.dropout_emb(memory)
            encoder_self_mask = None # Encoder usually doesn't use causal mask here
            if src_padding_mask is not None: encoder_self_mask = src_padding_mask
            for block in self.encoder_blocks:
                memory = block(memory, src_mask=encoder_self_mask)
            memory = self.encoder_norm(memory)

            # --- Decoder Processing ---
            _, seq_len_tgt, _ = tgt.shape  # We don't need batch_size_tgt
            assert seq_len_tgt <= self.window_size, f"Decoder input sequence {seq_len_tgt} longer than window size {self.window_size}"
            time_indices_tgt = torch.arange(seq_len_tgt, dtype=torch.long, device=tgt.device)
            output = self.decoder_time_embedding(tgt, time_indices_tgt)
            output = self.dropout_emb(output)
            decoder_self_mask = None
            if self.use_causal_mask_decoder: decoder_self_mask = self._generate_causal_mask(seq_len_tgt, tgt.device)
            # TODO: Combine causal and padding if needed
            # if tgt_padding_mask is not None: ...
            decoder_memory_mask = src_padding_mask # Cross-attention uses encoder padding mask
            for block in self.decoder_blocks:
                output = block(
                    input_tensor=output,  # Use input_tensor instead of tgt
                    memory=memory,
                    tgt_mask=decoder_self_mask,      # Causal mask for self-attention
                    memory_mask=decoder_memory_mask  # Encoder padding mask for cross-attention
                )
            output = self.decoder_norm(output)
            return output
        else:
            raise ValueError(f"Unknown architecture in DynamicTransformerCore: {self.architecture}")


# --- Model Factory Function ---
def create_model(config: ModelConfig, device: str = "auto") -> ActorCriticWrapper:
    """Creates a core model (Transformer or LSTM) and wraps it in ActorCriticWrapper."""

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        elif hasattr(torch, "xla") and torch.xla.is_available():
            device = "xla"  # TPU
        else:
            device = "cpu"

    # Set up attention mechanism
    attention_mapping = {
        "mha": MultiHeadAttention,
        "mla": MultiLatentAttention,
        "pyr": PyramidalAttention,
        "mqn": MultiQueryAttention,
        "gqa": GroupedQueryAttention
    }
    attention_class = attention_mapping[config.attention_type]

    # Configure attention arguments - USE embedding_dim CONSISTENTLY
    attention_args = {
        "embedding_dim": config.embedding_dim, # Changed from n_embd
        "n_heads": config.n_heads,
        "dropout": config.dropout
    }

    # Add specific attention arguments
    if config.attention_type == "mla" and config.n_latents:
        attention_args["n_latents"] = config.n_latents
    elif config.attention_type == "gqa" and config.n_groups:
        attention_args["n_groups"] = config.n_groups
    elif config.attention_type == "pyr":
        # PyramidalAttention also needs max_seq_len
        attention_args["max_seq_len"] = config.window_size

    # Set up feed-forward network
    ffn_mapping = {
        "standard": FeedForward,
        "moe": MixtureOfExperts
    }
    ffn_class = ffn_mapping[config.ffn_type]

    # Configure FFN arguments
    ffn_args = {"dropout": config.dropout}
    if config.ffn_dim:
        ffn_args["dim_feedforward"] = config.ffn_dim
    if config.ffn_type == "moe":
        if config.n_experts:
            ffn_args["num_experts"] = config.n_experts
        if config.top_k:
            ffn_args["top_k"] = config.top_k

    # Set up normalization
    norm_mapping = {
        "layer_norm": nn.LayerNorm,
        "batch_norm": nn.BatchNorm1d,
        "instance_norm": nn.InstanceNorm1d,
        "group_norm": nn.GroupNorm
    }
    
    if config.norm_type not in norm_mapping:
        raise ValueError(f"Unsupported norm_type: {config.norm_type}. Supported types: {list(norm_mapping.keys())}")
    
    norm_class = norm_mapping[config.norm_type]

    # --- Assemble DynamicTransformerCore ---

    # Validate architecture config
    if config.architecture not in ["encoder_only", "encoder_decoder", "decoder_only"]:
         raise ValueError(f"Unsupported architecture: {config.architecture}")

    # Shared components
    dropout_emb = nn.Dropout(config.dropout)
    norm_args = {} # Add if specific norm args are needed from config

    # Initialize component holders
    encoder_time_embedding = None
    encoder_blocks = None
    encoder_norm = None
    decoder_time_embedding = None
    decoder_blocks = None
    decoder_norm = None

    # Build Encoder components if needed
    if config.architecture in ["encoder_only", "encoder_decoder"]:
        if config.n_encoder_layers <= 0: raise ValueError("n_encoder_layers must be > 0 for encoder architectures.")
        encoder_time_embedding = TimeEmbedding(config.embedding_dim, max_len=config.window_size)
        encoder_blocks = nn.ModuleList()
        # Prepare args *before* the loop
        current_attention_args = attention_args.copy()
        # FFN args might need embedding_dim too, ensure it's added if needed by FFN class
        current_ffn_args = ffn_args.copy() if ffn_args else {}
        if 'embedding_dim' not in current_ffn_args:
            current_ffn_args['embedding_dim'] = config.embedding_dim

        for _ in range(config.n_encoder_layers):
            block = EncoderBlock(
                embedding_dim=config.embedding_dim, # Pass embedding_dim directly
                attention_class=attention_class,
                attention_args=current_attention_args, # Pass prepared args
                ffn_class=ffn_class,
                ffn_args=current_ffn_args, # Pass prepared FFN args
                # ffn_args=ffn_args.copy() if ffn_args else None,
                norm_class=norm_class,
                norm_args=norm_args,
                dropout=config.dropout,
                residual_scale=config.residual_scale,
                use_gated_residual=config.use_gated_residual,
                use_final_norm=config.use_final_norm
            )
            encoder_blocks.append(block)
        encoder_norm = norm_class(config.embedding_dim, **norm_args)

    # Build Decoder components if needed
    if config.architecture in ["decoder_only", "encoder_decoder"]:
        if config.n_decoder_layers <= 0: raise ValueError("n_decoder_layers must be > 0 for decoder architectures.")
        decoder_time_embedding = TimeEmbedding(config.embedding_dim, max_len=config.window_size) # Separate instance
        decoder_blocks = nn.ModuleList()
        # Prepare args *before* the loop
        current_attention_args_self = attention_args.copy()
        current_attention_args_cross = attention_args.copy()
        # FFN args might need embedding_dim too, ensure it's added if needed by FFN class
        current_ffn_args = ffn_args.copy() if ffn_args else {}
        if 'embedding_dim' not in current_ffn_args:
            current_ffn_args['embedding_dim'] = config.embedding_dim

        for _ in range(config.n_decoder_layers):
             block = DecoderBlock(
                 embedding_dim=config.embedding_dim, # Pass embedding_dim directly
                 self_attention_class=attention_class,
                 self_attention_args=current_attention_args_self, # Pass prepared args
                 # Cross attention only relevant for encoder_decoder
                 cross_attention_class=attention_class if config.architecture == "encoder_decoder" else None,
                 cross_attention_args=current_attention_args_cross if config.architecture == "encoder_decoder" else None, # Pass prepared args
                 ffn_class=ffn_class,
                 ffn_args=current_ffn_args, # Pass prepared FFN args
                 norm_class=norm_class,
                 norm_args=norm_args,
                 dropout=config.dropout,
                 residual_scale=config.residual_scale,
                 use_gated_residual=config.use_gated_residual,
                 use_final_norm=config.use_final_norm
             )
             decoder_blocks.append(block)
        decoder_norm = norm_class(config.embedding_dim, **norm_args)

    # Instantiate the dynamic core based on core_model_type
    core_model: nn.Module

    if config.core_model_type == "transformer":
        # Instantiate the dynamic transformer core
        core_model = DynamicTransformerCore(
            architecture=config.architecture,
            encoder_time_embedding=encoder_time_embedding,
            encoder_blocks=encoder_blocks,
            encoder_norm=encoder_norm,
            decoder_time_embedding=decoder_time_embedding,
            decoder_blocks=decoder_blocks,
            decoder_norm=decoder_norm,
            dropout_emb=dropout_emb,
            window_size=config.window_size,
            # Causal mask defaults - adjust if needed via config
            use_causal_mask_encoder=(config.architecture == "encoder_only"), # Often True for enc-only if causal
            use_causal_mask_decoder=True # Always True for dec self-attn
        )
    elif config.core_model_type == "lstm":
        if config.lstm_hidden_dim is None or config.lstm_num_layers is None or config.lstm_dropout is None:
            raise ValueError("LSTM parameters (hidden_dim, num_layers, dropout) must be specified in ModelConfig for LSTM core.")
        core_model = LSTMCore(
            input_dim=config.n_features, # LSTM input_dim is n_features directly from data
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
            device=device # Pass the determined device
        )
    else:
        raise ValueError(f"Unsupported core_model_type: {config.core_model_type}")


    # Create ActorCriticWrapper with the chosen core model
    model = ActorCriticWrapper(
        input_shape=(config.window_size, config.n_features), # Use actual n_features
        action_dim=config.action_dim,
        core_model=core_model, # Pass the generic core_model
        feature_extractor_hidden_dim=config.feature_extractor_dim,
        # embedding_dim is specific to Transformer's TimeEmbedding,
        # LSTM directly uses n_features as input_dim.
        # ActorCriticWrapper might need to know the output dim of the core_model.
        embedding_dim=config.embedding_dim if config.core_model_type == "transformer" else config.lstm_hidden_dim, # This might need adjustment based on ActorCriticWrapper
        dropout=config.dropout,
        temperature=config.temperature,  # Pass the temperature parameter
        device=device,
        # Feature extractor parameters
        feature_extractor_type=config.feature_extractor_type,
        feature_extractor_layers=config.feature_extractor_layers,
        use_skip_connections=config.use_skip_connections,
        use_layer_norm=config.use_layer_norm,
        use_instance_norm=config.use_instance_norm,
        feature_dropout=config.feature_dropout,
        # Actor-Critic head parameters
        head_hidden_dim=config.head_hidden_dim,
        head_n_layers=config.head_n_layers,
        head_use_layer_norm=config.head_use_layer_norm,
        head_use_residual=config.head_use_residual,
        head_dropout=config.head_dropout
    )

    return model
