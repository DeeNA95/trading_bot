import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FeedForward(nn.Module):
    """ Standard FeedForward network (using nn.Sequential) """
    def __init__(self, embedding_dim: int, dim_feedforward: Optional[int] = None, dropout: float = 0.1, activation: str = "gelu", bias: bool = True):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = embedding_dim * 4

        activation_module = nn.ReLU if activation == "relu" else nn.GELU

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward, bias=bias),
            activation_module(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embedding_dim, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class MixtureOfExperts(nn.Module):
    """ Mixture of Experts FeedForward layer """
    def __init__(self, embedding_dim: int, num_experts: int, top_k: int = 2, dim_feedforward: Optional[int] = None, dropout: float = 0.1, activation: str = "gelu", bias: bool = True, noisy_gating_std: Optional[float] = None):
        super().__init__()
        assert top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.embedding_dim = embedding_dim
        self.noisy_gating_std = noisy_gating_std

        # Create expert networks (using the standard FeedForward)
        self.experts = nn.ModuleList([
            FeedForward(embedding_dim, dim_feedforward, dropout, activation, bias)
            for _ in range(num_experts)
        ])

        # Gating network: projects input to scores for each expert
        self.gate = nn.Linear(embedding_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        gate_logits = self.gate(x_flat)

        if self.training and self.noisy_gating_std is not None and self.noisy_gating_std > 0:
            noise = torch.randn_like(gate_logits) * self.noisy_gating_std
            effective_gate_logits = gate_logits + noise
        else:
            effective_gate_logits = gate_logits

        weights, indices = torch.topk(effective_gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(x.dtype)

        final_output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            # Get the expert indices and weights for this k
            expert_idx = indices[:, k]
            expert_weights = weights[:, k].unsqueeze(-1)

            # Process each expert's assigned inputs in parallel
            for expert_id in range(self.num_experts):
                expert_mask = (expert_idx == expert_id)
                if expert_mask.any():
                    expert_input = x_flat[expert_mask]
                    expert_output = self.experts[expert_id](expert_input)
                    final_output[expert_mask] += expert_output * expert_weights[expert_mask]

        return final_output.view(batch_size, seq_len, dim)
