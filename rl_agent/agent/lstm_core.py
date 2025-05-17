import torch
import torch.nn as nn

class LSTMCore(nn.Module):
    """
    A core module using LSTM for sequence processing.
    This module can be used as a replacement for a Transformer core
    in the ActorCriticWrapper.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0, device: str = 'cpu'):
        """
        Initialize the LSTMCore.

        Args:
            input_dim: The number of expected features in the input x.
            hidden_dim: The number of features in the hidden state h.
            num_layers: Number of recurrent layers.
            dropout: If non-zero, introduces a Dropout layer on the outputs of each
                     LSTM layer except the last layer, with dropout probability equal to dropout.
            device: The device to run the model on ('cpu', 'cuda', etc.)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Input/output tensors are (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0  # Dropout only if multiple layers
        )
        self.to(device) # Ensure module is on the correct device

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the LSTMCore.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).
            src_key_padding_mask: Optional mask for input sequences. LSTM doesn't directly use
                                   a padding mask in the same way as Transformers, but it's included
                                   for API compatibility with the TransformerCore.
                                   If you need to handle packed sequences for variable lengths,
                                   that would require torch.nn.utils.rnn.pack_padded_sequence.
                                   For now, we assume fixed length sequences or that padding
                                   doesn't significantly impact LSTM if zeros are used for padding.

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        # LSTM expects input of shape (batch, seq_len, input_dim) when batch_first=True
        # Initialize hidden and cell states
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)
        # For LSTM, num_directions is 1 unless bidirectional=True (which we are not using here)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_len, hidden_dim) containing the output features (h_t) from the last layer of the LSTM, for each t.
        # hn: tensor of shape (num_layers * num_directions, batch, hidden_size) containing the final hidden state for each element in the batch.
        # cn: tensor of shape (num_layers * num_directions, batch, hidden_size) containing the final cell state for each element in the batch.
        out, (hn, cn) = self.lstm(x, (h0, c0))

        return out

    def get_output_dim(self) -> int:
        """
        Returns the output dimension of this core module.
        """
        return self.hidden_dim
