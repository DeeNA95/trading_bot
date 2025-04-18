import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FeedForward(nn.Module):
    """ Standard FeedForward network (using nn.Sequential) """
    def __init__(self, n_embd: int, dim_feedforward: Optional[int] = None, dropout: float = 0.1, activation: str = "relu", bias: bool = True):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = n_embd * 4

        activation_module = nn.ReLU if activation == "relu" else nn.GELU

        self.network = nn.Sequential(
            nn.Linear(n_embd, dim_feedforward, bias=bias),
            activation_module(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_embd, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class MixtureOfExperts(nn.Module):
    """ Mixture of Experts FeedForward layer """
    def __init__(self, n_embd: int, num_experts: int, top_k: int = 2, dim_feedforward: Optional[int] = None, dropout: float = 0.1, activation: str = "relu", bias: bool = True):
        super().__init__()
        assert top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_embd = n_embd

        # Create expert networks (using the standard FeedForward)
        self.experts = nn.ModuleList([
            FeedForward(n_embd, dim_feedforward, dropout, activation, bias)
            for _ in range(num_experts)
        ])

        # Gating network: projects input to scores for each expert
        self.gate = nn.Linear(n_embd, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim) # Flatten batch and sequence dimensions

        # Get gating scores: (B*T, C) -> (B*T, num_experts)
        gate_logits = self.gate(x_flat)

        # Select top-k experts based on scores
        # weights: (B*T, top_k), indices: (B*T, top_k)
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(x.dtype) # Normalize weights for top-k

        # Initialize final output tensor
        final_output_flat = torch.zeros_like(x_flat)

        # Map indices to a flat range for efficient gathering
        flat_indices = indices.view(-1)
        flat_x = x_flat.repeat_interleave(self.top_k, dim=0) # Repeat input for each chosen expert

        # Create expert index mapping for batch processing
        expert_indices = torch.arange(self.num_experts, device=x.device).repeat(batch_size * seq_len)
        batch_indices = torch.arange(batch_size * seq_len, device=x.device).repeat_interleave(self.num_experts)

        # Process inputs through their selected experts
        # This part is tricky to fully vectorize efficiently without scatter operations
        # or loops. A common approach involves routing and then combining.
        # Simplified (potentially less efficient) loop approach for clarity:
        expert_outputs = []
        for i in range(self.num_experts):
             # Find which inputs selected this expert
             idx, = torch.where(indices == i)
             if idx.numel() > 0:
                 # Process inputs for this expert
                 expert_output = self.experts[i](x_flat[idx])
                 # Store output and original index
                 expert_outputs.append((idx, expert_output))

        # Combine outputs weighted by gating scores
        # This requires careful indexing to place results back correctly
        # Using the weights and indices from topk:
        results = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = indices[:, i] # Expert chosen for this rank
            w = weights[:, i].unsqueeze(-1) # Weight for this expert

            # Gather outputs from the correct expert for each input token
            # This is complex. A simpler, but potentially less optimal way:
            current_expert_outputs = torch.zeros_like(x_flat)
            for exp_id in range(self.num_experts):
                mask = (expert_idx == exp_id)
                if mask.any():
                    # Get inputs routed to this expert at this rank
                    inputs_for_expert = x_flat[mask]
                    # Calculate output
                    output_for_expert = self.experts[exp_id](inputs_for_expert)
                    # Place output back
                    current_expert_outputs[mask] = output_for_expert

            results += current_expert_outputs * w


        # Reshape back to original shape
        final_output = results.view(batch_size, seq_len, dim)
        return final_output
