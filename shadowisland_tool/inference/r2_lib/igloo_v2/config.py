"""Released model configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass
class IglooV2Config:
    patch_mode: str = "random"
    patch_seed: int = 42
    attn_mode: str = "static"
    n_heads: int = 4
    d_k: int = 32
    w_qk_rank: int = 0
    use_pos_bias: bool = False
    use_alibi: bool = False
    alibi_slope: float = 0.1
    patch_size: int = 4
    nb_patches: int = 2100
    pooling_size: int = 8
    pool_mode: str = "max"
    conv_padding: str = "causal"
    conv_kernel: int = 6
    conv_dilations: Tuple[int, ...] = (1, 1, 1)
    patch_dropout: float = 0.0
    input_embedding_dim: int = 0
    readout_mode: str = "attn"
    rc_mode: str = "invariant"

    def tag(self) -> str:
        return "paper"

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: Dict[str, IglooV2Config] = {
    "paper": IglooV2Config(),
}


def get_model_config(name: str = "paper") -> IglooV2Config:
    key = name.strip().lower()
    if key not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown model preset {name!r}. Choose from: {known}")
    return PRESETS[key]
