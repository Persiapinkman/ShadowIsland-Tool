"""Attention variants for IGLOO v2 transformer-style branch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IglooV2Config
from .positional import PositionalBias, alibi_bias


class IglooAttention(nn.Module):
    """
    Compute attention weights alpha [B, L'] and weighted value summary [B, C].

    Modes
    -----
    static        : softmax(mpi @ W_qk)
    static_bias   : softmax(mpi @ W_qk + pos_bias)
    content       : softmax(mpi @ W_qk + content(mpi, y_proj) + pos_bias)
    content_alibi : softmax(content(mpi, y_proj) + alibi)
    multihead     : H heads, concat/split output channels
    lowrank       : softmax(mpi @ W1 @ W2 [+ pos_bias])
    """

    def __init__(
        self,
        cfg: IglooV2Config,
        nb_patches: int,
        pooled_len: int,
        channels: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.nb_patches = nb_patches
        self.pooled_len = pooled_len
        self.channels = channels
        self.mode = cfg.attn_mode

        if self.mode == "lowrank":
            rank = cfg.w_qk_rank
            if rank <= 0:
                raise ValueError("lowrank mode requires w_qk_rank > 0")
            self.w_qk_a = nn.Parameter(torch.empty(nb_patches, rank))
            self.w_qk_b = nn.Parameter(torch.empty(rank, pooled_len))
            nn.init.xavier_uniform_(self.w_qk_a)
            nn.init.xavier_uniform_(self.w_qk_b)
        elif self.mode != "multihead":
            self.w_qk = nn.Parameter(torch.empty(nb_patches, pooled_len))
            nn.init.xavier_uniform_(self.w_qk)

        self.w_v = nn.Parameter(torch.empty(channels, channels))
        nn.init.xavier_uniform_(self.w_v)

        needs_content = self.mode in ("content", "content_alibi")
        if needs_content:
            d_k = cfg.d_k
            self.w_q = nn.Parameter(torch.empty(nb_patches, d_k))
            self.w_k = nn.Parameter(torch.empty(channels, d_k))
            nn.init.xavier_uniform_(self.w_q)
            nn.init.xavier_uniform_(self.w_k)

        if self.mode == "multihead":
            if channels % cfg.n_heads != 0:
                raise ValueError(
                    f"channels={channels} must be divisible by n_heads={cfg.n_heads}"
                )
            self.head_dim = channels // cfg.n_heads
            self.w_q_heads = nn.Parameter(
                torch.empty(cfg.n_heads, nb_patches, cfg.d_k)
            )
            self.w_k_heads = nn.Parameter(
                torch.empty(cfg.n_heads, channels, cfg.d_k)
            )
            nn.init.xavier_uniform_(self.w_q_heads)
            nn.init.xavier_uniform_(self.w_k_heads)

        self.pos_bias = (
            PositionalBias(pooled_len) if cfg.use_pos_bias else None
        )

    def project_values(self, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(y, self.w_v)

    def _static_logits(self, mpi: torch.Tensor) -> torch.Tensor:
        if self.mode == "lowrank":
            return torch.matmul(torch.matmul(mpi, self.w_qk_a), self.w_qk_b)
        return torch.matmul(mpi, self.w_qk)

    def _content_logits(self, mpi: torch.Tensor, y_proj: torch.Tensor) -> torch.Tensor:
        d_k = self.cfg.d_k
        q = torch.matmul(mpi, self.w_q)  # [B, d_k]
        k = torch.matmul(y_proj, self.w_k)  # [B, L', d_k]
        logits = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1)
        return logits / math.sqrt(d_k)

    def _position_term(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor | None:
        if self.cfg.use_alibi:
            return alibi_bias(
                self.pooled_len, self.cfg.alibi_slope, device
            ).expand(batch_size, -1)
        if self.pos_bias is not None:
            return self.pos_bias(batch_size, device)
        return None

    def forward(
        self,
        mpi: torch.Tensor,
        y_proj: torch.Tensor,
        return_alpha: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        mpi     : [B, P] patch scores
        y_proj  : [B, L', C] pooled value features

        Returns
        -------
        out : [B, C]
        alpha (optional) : [B, L']
        """
        b, device = mpi.shape[0], mpi.device

        if self.mode == "multihead":
            return self._forward_multihead(mpi, y_proj, return_alpha)

        logits = torch.zeros(b, self.pooled_len, device=device)

        if self.mode in ("static", "static_bias", "lowrank"):
            logits = self._static_logits(mpi)
        elif self.mode == "content":
            logits = self._content_logits(mpi, y_proj) + self._static_logits(mpi)
        elif self.mode == "content_alibi":
            logits = self._content_logits(mpi, y_proj)
        else:
            raise ValueError(f"Unknown attn_mode={self.mode!r}")

        pos = self._position_term(b, device)
        if pos is not None:
            logits = logits + pos

        alpha = F.softmax(logits, dim=-1)
        out = torch.matmul(alpha.unsqueeze(1), y_proj).squeeze(1)

        if return_alpha:
            return out, alpha
        return out

    def _forward_multihead(
        self,
        mpi: torch.Tensor,
        y_proj: torch.Tensor,
        return_alpha: bool,
    ):
        b, _, c = y_proj.shape
        h = self.cfg.n_heads
        d_k = self.cfg.d_k
        head_outs = []
        alphas = []

        for hi in range(h):
            q = torch.matmul(mpi, self.w_q_heads[hi])  # [B, d_k]
            k = torch.matmul(y_proj, self.w_k_heads[hi])  # [B, L', d_k]
            logits = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1)
            logits = logits / math.sqrt(d_k)

            if self.pos_bias is not None:
                logits = logits + self.pos_bias.bias.unsqueeze(0)

            alpha = F.softmax(logits, dim=-1)
            head_val = torch.matmul(alpha.unsqueeze(1), y_proj).squeeze(1)
            head_outs.append(head_val)
            alphas.append(alpha)

        # average head outputs to keep [B, C] (channels unchanged vs baseline)
        out = torch.stack(head_outs, dim=0).mean(dim=0)
        if return_alpha:
            alpha_mean = torch.stack(alphas, dim=0).mean(dim=0)
            return out, alpha_mean
        return out
