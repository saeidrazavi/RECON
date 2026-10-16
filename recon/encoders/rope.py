# 2D rotary position embedding used by the PE teachers (pe.py).
# Reduced from lucidrains' rotary-embedding-torch to the code path PE uses:
# the default "lang" frequencies, applied to the leading feature dims.

import torch
from einops import rearrange, repeat
from torch import einsum, nn
from torch.amp import autocast


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t):
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    t, t_right = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_right), dim=-1).type(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.register_buffer("freqs", freqs, persistent=False)

    @autocast("cuda", enabled=False)
    def forward(self, t):
        freqs = einsum("..., f -> ... f", t.type(self.freqs.dtype), self.freqs)
        return repeat(freqs, "... n -> ... (n r)", r=2)


class Rope2D:
    """ Helper class to apply RoPE2D as well as interpolate on the fly. """

    def __init__(self, dim, use_cls_token=False):
        self.dim = dim
        self.use_cls_token = use_cls_token
        self.grid_size = None
        self.freq = None

    def init_tensors(self):
        self.rope = RotaryEmbedding(self.dim // 2)

    def update_grid(self, device, grid_h, grid_w):
        if self.grid_size != (grid_h, grid_w):
            self.grid_size = (grid_h, grid_w)

            self.rope = self.rope.to(device)

            if self.use_cls_token:
                # +1 to leave space for the cls token to be (0, 0)
                grid_y_range = torch.arange(grid_h, device=device) + 1
                grid_x_range = torch.arange(grid_w, device=device) + 1
            else:
                grid_y_range = torch.arange(grid_h, device=device)
                grid_x_range = torch.arange(grid_w, device=device)

            freqs_y = self.rope(grid_y_range)[:, None].expand(grid_h, grid_w, -1)
            freqs_x = self.rope(grid_x_range)[None, :].expand(grid_h, grid_w, -1)
            freq = torch.cat([freqs_x, freqs_y], dim=-1).reshape(grid_h * grid_w, -1)

            if self.use_cls_token:
                freq = torch.cat(
                    [torch.zeros(1, freq.shape[-1], device=device), freq], dim=0
                )

            self.freq = freq[None, ...]

        self.freq = self.freq.to(device)

    def __call__(self, q, k):
        # batch, heads, seq, dim = q.shape
        q = apply_rotary_emb(self.freq[:, None, :, :], q)
        k = apply_rotary_emb(self.freq[:, None, :, :], k)

        return q, k
