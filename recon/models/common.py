# Building blocks shared by SiT (sit.py) and MMDiT (mmdit.py): the REPA
# projector, the semantic-token pooler, adaLN helpers and the fixed sin-cos position
# embedding (https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py).

import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):
    """
    The final layer: adaLN-modulated LayerNorm and a linear map to patch pixels.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x


class SemanticTokenPooler(nn.Module):
    """Pool teacher features into K semantic tokens and map them to the model width.

    The features (N, T, z_dim) are L2-normalized and mean-pooled over a sqrt(K) x sqrt(K) grid of the
    patches. proj_type "linear" maps the tokens with a zero-initialized Linear, and "zeropad" pads
    them with zeros to hidden_size (no parameters).
    """
    def __init__(self, hidden_size, z_dim, num_tokens, proj_type="linear"):
        super().__init__()
        assert proj_type in ("linear", "zeropad"), f"unknown sem proj_type: {proj_type}"
        self.out_side = int(round(math.sqrt(num_tokens)))
        assert self.out_side * self.out_side == num_tokens, \
            f"num_semantic_tokens must be a perfect square, got {num_tokens}"
        self.num_tokens = num_tokens
        self.proj_type = proj_type
        if proj_type == "linear":
            self.proj = nn.Linear(z_dim, hidden_size, bias=True)
            self.pad = 0
        else:
            assert z_dim <= hidden_size, \
                f"zeropad needs z_dim <= hidden_size, got z_dim={z_dim} hidden_size={hidden_size}"
            self.proj = nn.Identity()
            self.pad = hidden_size - z_dim

    def project(self, pooled):
        """Map a pooled feature (N, K, z_dim) to the model width (N, K, hidden_size)."""
        if self.proj_type == "zeropad":
            return F.pad(pooled, (0, self.pad))
        return self.proj(pooled)

    def forward(self, z, extrapolation=None):
        # z: (N, T, z_dim), T must be a perfect square (square grid of teacher patches)
        N, T, C = z.shape
        z = F.normalize(z, dim=-1)                                                # (N, T, z_dim)
        in_side = int(round(math.sqrt(T)))
        assert in_side * in_side == T, f"teacher token count must be a perfect square, got {T}"
        assert in_side % self.out_side == 0, \
            f"teacher grid {in_side} not divisible by pooled grid {self.out_side}"
        block = in_side // self.out_side
        # (N, out_side, block, out_side, block, C) -> mean over the two block dims
        z = z.reshape(N, self.out_side, block, self.out_side, block, C)
        z = z.mean(dim=(2, 4))                                                    # (N, out_side, out_side, C)
        z = z.reshape(N, self.num_tokens, C)                                      # (N, K, z_dim)
        if extrapolation is not None:  # Semantic Token Extrapolation of the pooled token (sampling only)
            z = extrapolation(z)
        return self.project(z)                                                    # (N, K, hidden_size)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
