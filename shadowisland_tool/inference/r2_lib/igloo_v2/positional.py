"""Positional bias helpers for IGLOO v2 attention logits."""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionalBias(nn.Module):
    """Learnable absolute bias over pooled sequence bins."""

    def __init__(self, pooled_len: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(pooled_len))
        nn.init.normal_(self.bias, std=0.02)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # [B, L']
        return self.bias.unsqueeze(0).expand(batch_size, -1)


def alibi_bias(
    pooled_len: int,
    slope: float,
    device: torch.device,
    ref: float | None = None,
) -> torch.Tensor:
    """
    1D ALiBi-style bias for a single query attending to L' keys.

    Penalises distance from a reference bin (default: sequence centre).
    Returns [1, L'] to broadcast over batch.
    """
    if ref is None:
        ref = (pooled_len - 1) / 2.0
    positions = torch.arange(pooled_len, device=device, dtype=torch.float32)
    return (-slope * (positions - ref).abs()).unsqueeze(0)
