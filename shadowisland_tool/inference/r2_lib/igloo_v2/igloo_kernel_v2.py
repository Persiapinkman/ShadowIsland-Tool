"""IGLOO1D v2 — drop-in Block replacement with configurable patch/attention/PE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import IglooAttention
from .config import IglooV2Config
from .patch_generators import generate_patches


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=0, dilation=dilation
        )

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class SameLengthConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.total_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=0, dilation=dilation
        )

    def forward(self, x):
        left = self.total_padding // 2
        right = self.total_padding - left
        x = F.pad(x, (left, right))
        return self.conv(x)


class IGLOO1D_KernelV2(nn.Module):
    def __init__(
        self,
        input_channels: int,
        vector_size: int,
        cfg: IglooV2Config,
        dropout_rate: float = 0.01,
        l2_reg: float = 1e-6,
        transformer_style: bool = True,
        incoming_proj: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.nb_patches = cfg.nb_patches
        self.pooling_size = cfg.pooling_size
        self.transformer_style = transformer_style
        self.num_channels_input = input_channels
        self.vector_size = vector_size
        self.incoming_proj_dim = incoming_proj

        if incoming_proj > 0:
            self.w_incoming = nn.Parameter(
                torch.empty(input_channels, incoming_proj)
            )
            nn.init.xavier_uniform_(self.w_incoming)
            current_channels = incoming_proj
        else:
            current_channels = input_channels

        patches_np = generate_patches(
            cfg.patch_mode,
            cfg.patch_size,
            cfg.nb_patches,
            vector_size,
            seed=cfg.patch_seed,
        )
        self.register_buffer("patches", torch.as_tensor(patches_np, dtype=torch.long))

        self.w_mult = nn.Parameter(
            torch.empty(1, cfg.nb_patches, cfg.patch_size, current_channels)
        )
        self.w_summer = nn.Parameter(
            torch.empty(1, cfg.patch_size * current_channels, 1)
        )
        self.w_bias = nn.Parameter(torch.empty(1, cfg.nb_patches))
        nn.init.xavier_uniform_(self.w_mult)
        nn.init.xavier_uniform_(self.w_summer)
        nn.init.xavier_uniform_(self.w_bias)

        self.pooled_len = vector_size // cfg.pooling_size
        if transformer_style:
            self.attn = IglooAttention(
                cfg, cfg.nb_patches, self.pooled_len, input_channels
            )

    def _pool(self, y_proj: torch.Tensor) -> torch.Tensor:
        if self.pooling_size <= 1:
            return y_proj
        y_proj = y_proj.permute(0, 2, 1)
        if self.cfg.pool_mode == "avg":
            y_proj = F.avg_pool1d(y_proj, kernel_size=self.pooling_size)
        else:
            y_proj = F.max_pool1d(y_proj, kernel_size=self.pooling_size)
        return y_proj.permute(0, 2, 1)

    def forward(self, x, return_alpha: bool = False):
        # x: [B, C, L]
        y = x.permute(0, 2, 1)

        if self.incoming_proj_dim > 0:
            y_next = torch.matmul(y, self.w_incoming)
            current_channels = self.incoming_proj_dim
        else:
            y_next = y
            current_channels = self.num_channels_input

        flat_indices = self.patches.view(-1)
        mpi = torch.index_select(y_next, 1, flat_indices)
        mpi = mpi.view(-1, self.nb_patches, self.patch_size, current_channels)
        mpi = mpi * self.w_mult
        mpi = mpi.reshape(-1, self.nb_patches, self.patch_size * current_channels)
        mpi = torch.matmul(mpi, self.w_summer.squeeze(0)).squeeze(-1)
        mpi = mpi + self.w_bias

        if self.training and self.cfg.patch_dropout > 0:
            keep = torch.rand_like(mpi) >= self.cfg.patch_dropout
            mpi = mpi.masked_fill(~keep, 0.0)

        if self.transformer_style:
            y_proj = self.attn.project_values(y)
            y_proj = self._pool(y_proj)
            if self.cfg.readout_mode == "attn_meanmax":
                attn_out = self.attn(mpi, y_proj)
                mean_out = y_proj.mean(dim=1)
                max_out = y_proj.max(dim=1).values
                return (attn_out + mean_out + max_out) / 3.0
            if return_alpha:
                out, alpha = self.attn(mpi, y_proj, return_alpha=True)
                return out, alpha
            return self.attn(mpi, y_proj)

        return F.leaky_relu(mpi, negative_slope=0.1)


class IGLOO1D_BlockV2(nn.Module):
    """API-compatible with igloo_pytorch.IGLOO1D_Block."""

    def __init__(
        self,
        in_channels: int,
        input_length: int,
        cfg: IglooV2Config,
        nb_filters_conv1d: int = 128,
        padding_style: str = "causal",
        nb_stacks: int = 3,
        conv1d_kernel: int = 6,
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-3,
        transformer_style: bool = True,
        spatial_dropout: bool = True,
        incoming_proj: int = 0,
    ):
        super().__init__()
        self.nb_stacks = nb_stacks
        self.conv_stacks = nn.ModuleList()
        self.igloo_stacks = nn.ModuleList()

        for i in range(nb_stacks):
            current_in = in_channels if i == 0 else nb_filters_conv1d
            dilation = cfg.conv_dilations[i] if i < len(cfg.conv_dilations) else cfg.conv_dilations[-1]
            kernel_size = cfg.conv_kernel if cfg.conv_kernel else conv1d_kernel
            padding = cfg.conv_padding or padding_style
            if padding == "causal":
                conv = CausalConv1d(current_in, nb_filters_conv1d, kernel_size, dilation=dilation)
            elif padding == "same":
                conv = SameLengthConv1d(current_in, nb_filters_conv1d, kernel_size, dilation=dilation)
            else:
                conv = nn.Conv1d(
                    current_in,
                    nb_filters_conv1d,
                    kernel_size,
                    padding=(kernel_size // 2) * dilation,
                    dilation=dilation,
                )

            drop = (
                nn.Dropout1d(p=dropout_rate)
                if spatial_dropout
                else nn.Dropout(p=dropout_rate)
            )
            self.conv_stacks.append(
                nn.Sequential(conv, nn.LeakyReLU(0.1), drop)
            )
            self.igloo_stacks.append(
                IGLOO1D_KernelV2(
                    input_channels=nb_filters_conv1d,
                    vector_size=input_length,
                    cfg=cfg,
                    dropout_rate=dropout_rate,
                    l2_reg=l2_reg,
                    transformer_style=transformer_style,
                    incoming_proj=incoming_proj,
                )
            )

    def forward(self, x):
        outputs = []
        curr = x
        for i in range(self.nb_stacks):
            curr = self.conv_stacks[i](curr)
            outputs.append(self.igloo_stacks[i](curr))
        return torch.cat(outputs, dim=1) if self.nb_stacks > 1 else outputs[0]
