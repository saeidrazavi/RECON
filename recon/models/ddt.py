"""DDT (decoupled diffusion transformer) for text-to-image, with REPA projectors and semantic tokens.

Based on DiTwDDTHead in DiffusionBench (src/stage2/models/DDT.py), with text conditioning only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed

from recon.models.common import SemanticTokenPooler


def modulate(x, shift, scale):
    """adaLN modulation; shift and scale have the shape of x (one per token)."""
    return x * (1 + scale) + shift


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class RoPE(nn.Module):
    """2D rotary embedding over the patch grid. The cond_len tokens after the patches (time, text and
    semantic tokens) get angle 0, i.e. no rotation."""
    def __init__(self, dim, vis_len, cond_len=0, theta=10000.):
        super().__init__()
        d, T = dim // 2, int(vis_len ** 0.5)
        vis_freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))  # [D//4]
        vis_base_angles = torch.outer(torch.arange(T).float(), vis_freqs)  # [T, D//4]
        vis_angles = torch.cat([
            vis_base_angles[:, None].expand(-1, T, -1),
            vis_base_angles[None, :].expand(T, -1, -1)
        ], dim=-1).reshape(vis_len, d)  # [T, T, D//2] -> [L', D//2]
        cond_angles = torch.zeros(cond_len, dim // 2)
        angles = torch.cat([vis_angles, cond_angles], dim=0).repeat_interleave(2, dim=-1)  # [L, D]
        self.register_buffer("freqs_cos", angles.cos(), persistent=False)
        self.register_buffer("freqs_sin", angles.sin(), persistent=False)

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_f = x.float()
        return (x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class NormAttention(nn.Module):
    """Multi-head attention with RMSNorm on queries and keys and RoPE."""
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads, self.dim, self.head_dim = num_heads, dim, dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope, attn_mask=None):
        B, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = rope(self.q_norm(q)), rope(self.k_norm(k))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out.permute(0, 2, 1, 3).reshape(B, N, self.dim))


class GaussianFourierEmbedding(nn.Module):
    """Time embedding: fixed random Fourier features of t and an MLP, added to n_tokens learnable tokens."""
    def __init__(self, hidden_size, n_tokens=4, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.normal(0, scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.learnable_tokens = nn.Parameter(torch.normal(0, 1 / hidden_size**0.5, (n_tokens, hidden_size)))

    def forward(self, t):
        """Returns the base embedding (N, 1, D) and the time tokens (N, n_tokens, D)."""
        t = t[:, None] * self.W[None, :] * 2 * torch.pi
        t_embed = self.mlp(torch.cat([torch.sin(t), torch.cos(t)], dim=-1)).unsqueeze(1)
        return t_embed, self.learnable_tokens + t_embed


class TextEmbedder(nn.Module):
    """RMSNorm and a Linear from the text encoder width to the model width."""
    def __init__(self, hidden_size, context_dim):
        super().__init__()
        self.norm = RMSNorm(context_dim)
        self.proj = nn.Linear(context_dim, hidden_size)

    def forward(self, context):
        return self.proj(self.norm(context))


class DDTEncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = NormAttention(hidden_size, num_heads)
        self.mlp = SwiGLUFFN(hidden_size, int(2/3 * hidden_size * mlp_ratio))

    def forward(self, x, rope, attn_mask=None):
        x = x + self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class DDTDecoderBlock(DDTEncoderBlock):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__(hidden_size, num_heads, mlp_ratio)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, c, rope, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=rope, attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaln_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DDT(nn.Module):
    """hidden_size, depth and num_heads are (encoder, decoder) pairs. The t2i configuration used for
    RECON is the default: a 28-block, 1152-wide encoder and a 2-block, 2048-wide decoder on 4x32x32
    SD-VAE latents with patch size 2, 4 time tokens and 256 Qwen3-0.6B text tokens (width 1024)."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=(1152, 2048),
        depth=(28, 2),
        num_heads=(16, 16),
        mlp_ratio=4.0,
        context_dim=1024,
        num_text_tokens=256,
        num_t_tokens=4,
        encoder_depth=8,
        z_dims=(768,),
        num_semantic_tokens=4,
        sem_inject_max_t=1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        enc_hidden_size, dec_hidden_size = hidden_size
        self.num_enc_blocks, num_dec_blocks = depth
        enc_num_heads, dec_num_heads = num_heads
        self.encoder_depth = encoder_depth
        # Semantic tokens are only visible for t <= sem_inject_max_t during training; at
        # inference they are always injected.
        self.sem_inject_max_t = sem_inject_max_t

        self.s_embedder = PatchEmbed(input_size, patch_size, in_channels, enc_hidden_size)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, dec_hidden_size)
        self.s_projector = (nn.Linear(enc_hidden_size, dec_hidden_size)
                            if enc_hidden_size != dec_hidden_size else nn.Identity())
        self.t_embedder = GaussianFourierEmbedding(enc_hidden_size, num_t_tokens)
        self.context_embedder = TextEmbedder(enc_hidden_size, context_dim)
        self.blocks = nn.ModuleList(
            [DDTEncoderBlock(enc_hidden_size, enc_num_heads, mlp_ratio) for _ in range(self.num_enc_blocks)]
            + [DDTDecoderBlock(dec_hidden_size, dec_num_heads, mlp_ratio) for _ in range(num_dec_blocks)]
        )
        self.final_layer = DDTFinalLayer(dec_hidden_size, patch_size, in_channels)
        num_patches, num_cond_tokens = self.s_embedder.num_patches, num_t_tokens + num_text_tokens
        self.enc_rope = RoPE(enc_hidden_size // enc_num_heads, num_patches, num_cond_tokens)
        self.dec_rope = RoPE(dec_hidden_size // dec_num_heads, self.x_embedder.num_patches)

        # REPA: one Linear projector per teacher
        self.projectors = nn.ModuleList([nn.Linear(enc_hidden_size, z_dim) for z_dim in z_dims])
        # Semantic tokens (none when num_semantic_tokens == 0: plain REPA). Once they are appended,
        # the encoder RoPE treats them as condition tokens.
        self.sem_poolers = nn.ModuleList([
            SemanticTokenPooler(enc_hidden_size, z_dim, num_semantic_tokens) for z_dim in z_dims
        ] if num_semantic_tokens > 0 else [])
        self.enc_rope_sem = RoPE(enc_hidden_size // enc_num_heads, num_patches,
                                 num_cond_tokens + num_semantic_tokens * len(self.sem_poolers))
        self.initialize_weights()

    def initialize_weights(self):
        # Patch embedders like nn.Linear
        for embedder in (self.x_embedder, self.s_embedder):
            w = embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(embedder.proj.bias, 0)

        # Zero-out the decoder blocks' adaLN modulation
        for block in self.blocks:
            if hasattr(block, "adaln_modulation"):
                nn.init.constant_(block.adaln_modulation[-1].weight, 0)
                nn.init.constant_(block.adaln_modulation[-1].bias, 0)

        # Time embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Semantic tokens start as zeros
        for pooler in self.sem_poolers:
            nn.init.constant_(pooler.proj.weight, 0)
            nn.init.constant_(pooler.proj.bias, 0)

        # Zero-out the output layer
        nn.init.constant_(self.final_layer.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """(N, T, patch_size**2 * C) -> (N, C, H, W)"""
        h, p, c = int(x.shape[1] ** 0.5), self.patch_size, self.in_channels
        return x.reshape(x.shape[0], h, h, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, h * p, h * p)

    def _inject_semantic_tokens(self, seq, attn_mask, zs, teacher_zs, t):
        """Append each teacher's K semantic tokens, pooled from the teacher features (training) or the
        predicted ones (inference), and K key columns to the additive mask. During training the
        columns are masked for samples with t > sem_inject_max_t, so no token attends to them."""
        sem = torch.cat([pooler(z) for z, pooler in zip(teacher_zs if teacher_zs is not None else zs,
                                                         self.sem_poolers)], dim=1)
        sem_cols = attn_mask.new_zeros(seq.shape[0], 1, 1, sem.shape[1])  # 0 = attend
        if self.training and self.sem_inject_max_t < 1.0:
            drop = (t > self.sem_inject_max_t).view(-1, 1, 1, 1)
            sem_cols = sem_cols.masked_fill(drop, torch.finfo(attn_mask.dtype).min)
        return torch.cat([seq, sem], dim=1), torch.cat([attn_mask, sem_cols], dim=-1)

    def forward(self, x, t, context, attn_mask=None, teacher_zs=None):
        """
        x: (N, C, H, W) latents; t: (N,) diffusion times (t = 1 is noise)
        context: (N, L, context_dim) text tokens; attn_mask: (N, L), 1 for tokens and 0 for padding
        teacher_zs: list of (N, T, z_dim) teacher features for the semantic tokens during training;
                    None at inference (the tokens are pooled from the predicted features)
        Returns the predicted velocity (N, C, H, W) and the list of predicted teacher features.
        """
        n = self.s_embedder.num_patches
        t_base, t_tokens = self.t_embedder(t)
        seq = torch.cat([self.s_embedder(x), t_tokens, self.context_embedder(context)], dim=1)
        # additive key mask over the sequence: 0 = attend, finfo.min = text padding
        keep = torch.ones(seq.shape[:2], device=seq.device)
        if attn_mask is not None:
            keep[:, -attn_mask.shape[1]:] = attn_mask
        mask = (1.0 - keep[:, None, None, :]) * torch.finfo(seq.dtype).min

        rope, zs = self.enc_rope, []
        for i in range(self.num_enc_blocks):
            seq = self.blocks[i](seq, rope, mask)
            if (i + 1) == self.encoder_depth:
                zs = [projector(seq[:, :n]) for projector in self.projectors]
                if self.sem_poolers:
                    seq, mask = self._inject_semantic_tokens(seq, mask, zs, teacher_zs, t)
                    rope = self.enc_rope_sem
        cond = self.s_projector(F.silu(t_base + seq[:, :n]))

        x = self.x_embedder(x)
        for block in self.blocks[self.num_enc_blocks:]:
            x = block(x, cond, self.dec_rope)
        return self.unpatchify(self.final_layer(x, cond)), zs
