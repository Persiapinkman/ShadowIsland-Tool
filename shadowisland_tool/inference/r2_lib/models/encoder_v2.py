"""Released sequence encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from igloo_v2.config import IglooV2Config
from igloo_v2.igloo_kernel_v2 import IGLOO1D_BlockV2

WORD_SIZE = 4
FRAGMENT_LENGTH = 6000
INPUT_SHAPE = FRAGMENT_LENGTH - WORD_SIZE + 1


def reverse_complement_tokens(x: torch.Tensor) -> torch.Tensor:
    """Reverse-complement 4-mer token ids generated in A/C/G/T lexicographic order."""
    ids = x.long()
    out = torch.zeros_like(ids)
    mask = ids > 0
    z = ids.clamp(min=1) - 1
    rc = torch.zeros_like(z)
    for _ in range(WORD_SIZE):
        digit = z % 4
        z = z // 4
        rc = rc * 4 + (3 - digit)
    out[mask] = rc[mask] + 1
    return torch.flip(out, dims=[1])


class EncoderV2(nn.Module):
    def __init__(
        self,
        cfg: IglooV2Config,
        nb_stacks: int = 3,
        nb_filters: int = 128,
        dropout: float = 0.2,
        output_dim: int = 512,
    ):
        super().__init__()
        self.depth = WORD_SIZE**4 + 1
        self.output_dim = output_dim
        self.cfg = cfg

        if cfg.input_embedding_dim > 0:
            self.token_embedding = nn.Embedding(
                self.depth, cfg.input_embedding_dim, padding_idx=0
            )
            in_channels = cfg.input_embedding_dim
        else:
            self.token_embedding = None
            in_channels = self.depth

        self.igloo = IGLOO1D_BlockV2(
            in_channels=in_channels,
            input_length=INPUT_SHAPE,
            cfg=cfg,
            nb_filters_conv1d=nb_filters,
            nb_stacks=nb_stacks,
            conv1d_kernel=6,
            dropout_rate=dropout,
            l2_reg=1e-3,
            transformer_style=True,
        )
        igloo_out = nb_filters * nb_stacks
        self.fc = nn.Sequential(
            nn.Linear(igloo_out, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        if self.cfg.rc_mode == "invariant":
            h1 = self._forward_once(x)
            h2 = self._forward_once(reverse_complement_tokens(x))
            return (h1 + h2) / 2.0
        return self._forward_once(x)

    def _forward_once(self, x):
        if self.token_embedding is not None:
            x = self.token_embedding(x.long()).permute(0, 2, 1)
        else:
            x = F.one_hot(x.long(), num_classes=self.depth).float()
            x = x.permute(0, 2, 1)
        x = self.igloo(x)
        return self.fc(x)


class FullModelV2(nn.Module):
    def __init__(self, cfg: IglooV2Config, proj_dim: int = 128, **encoder_kw):
        super().__init__()
        self.encoder = EncoderV2(cfg, **encoder_kw)
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.output_dim, proj_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.projector(h)
