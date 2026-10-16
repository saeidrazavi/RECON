### MM-DiT, the core model of SD3, reduced to the configuration used for text-to-image
### training here, plus REPA projectors and semantic-token pooling.

import math
from typing import Optional
import torch
import torch.nn as nn

from recon.models.common import FinalLayer, SemanticTokenPooler, build_mlp, get_2d_sincos_pos_embed, modulate


#################################################################################################
### Core/Utility
#################################################################################################


def attention(q, k, v, heads, mask=None):
    """Convenience wrapper around a basic attention operation"""
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.img_size = (img_size, img_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # NCHW -> NLC


#################################################################################
#                        Embedding Layer for Timesteps                          #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        if torch.is_floating_point(t):
            embedding = embedding.to(dtype=t.dtype)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


def split_qkv(qkv, head_dim):
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, head_dim).movedim(2, 0)
    return qkv[0], qkv[1], qkv[2]


class SelfAttention(nn.Module):
    """QKV projection and output projection around the joint attention in block_mixing."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, pre_only=False, qk_norm=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if not pre_only:
            self.proj = nn.Linear(dim, dim)
        self.pre_only = pre_only

        if qk_norm == "ln":
            self.ln_q = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
            self.ln_k = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)

    def pre_attention(self, x: torch.Tensor):
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.head_dim)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        return self.proj(x)


class DismantledBlock(nn.Module):
    """A DiT block with gated adaptive layer norm (adaLN) conditioning. A pre_only block
    (the context stream of the last joint block) only produces q, k, v."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=False, pre_only=False, qk_norm=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, pre_only=pre_only, qk_norm=qk_norm)
        if not pre_only:
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=nn.GELU(approximate="tanh"))
        n_mods = 6 if not pre_only else 2
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size, bias=True))
        self.pre_only = pre_only

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor):
        if not self.pre_only:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
        qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
        return qkv, None

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


def block_mixing(context, x, context_block, x_block, c, attn_mask=None):
    assert context is not None, "block_mixing called with None context"
    context_qkv, context_intermediates = context_block.pre_attention(context, c)

    x_qkv, x_intermediates = x_block.pre_attention(x, c)

    q, k, v = (torch.cat((c_t, x_t), dim=1) for c_t, x_t in zip(context_qkv, x_qkv))

    attn = attention(q, k, v, x_block.attn.num_heads, mask=attn_mask)
    context_attn, x_attn = (attn[:, : context_qkv[0].shape[1]], attn[:, context_qkv[0].shape[1] :])

    if not context_block.pre_only:
        context = context_block.post_attention(context_attn, *context_intermediates)
    else:
        context = None
    x = x_block.post_attention(x_attn, *x_intermediates)
    return context, x


class JointBlock(nn.Module):
    """A context block and an image block that attend jointly over [context, x]."""

    def __init__(self, *args, pre_only=False, **kwargs):
        super().__init__()
        self.context_block = DismantledBlock(*args, pre_only=pre_only, **kwargs)
        self.x_block = DismantledBlock(*args, pre_only=False, **kwargs)

    def forward(self, *args, **kwargs):
        return block_mixing(*args, context_block=self.context_block, x_block=self.x_block, **kwargs)


class MMDiT(nn.Module):
    """MM-DiT conditioned on the diffusion time and on text tokens (context), with REPA
    projectors and semantic tokens appended to the image stream at encoder_depth."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        depth: int = 24,
        mlp_ratio: float = 4.0,
        context_dim: int = 768,
        qk_norm: Optional[str] = None,
        qkv_bias: bool = True,
        encoder_depth=8,
        z_dims=[768],
        projector_dim=2048,
        num_semantic_tokens=4,
        sem_inject_max_t=1.0,
        sem_proj_type="linear",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size

        # width and head count both scale with depth (head dim 32)
        hidden_size = 32 * depth
        num_heads = depth
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.context_embedder = nn.Linear(context_dim, hidden_size)

        # Fixed sin-cos position embedding over the patch grid (filled in below)
        self.register_buffer("pos_embed", torch.zeros(1, self.x_embedder.num_patches, hidden_size))

        self.joint_blocks = nn.ModuleList([
            JointBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                       pre_only=i == depth - 1, qk_norm=qk_norm)
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # REPA
        self.encoder_depth = encoder_depth
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            ])

        # Semantic-token poolers. Semantic tokens are
        # only injected for timesteps t <= sem_inject_max_t during training; at
        # eval they are always injected.
        self.num_semantic_tokens = num_semantic_tokens
        self.sem_inject_max_t = sem_inject_max_t
        self.sem_proj_type = sem_proj_type
        self.sem_poolers = nn.ModuleList([
            SemanticTokenPooler(hidden_size, z_dim, num_semantic_tokens, proj_type=sem_proj_type)
            for z_dim in z_dims
            ] if num_semantic_tokens > 0 else [])  # 0 semantic tokens: plain REPA
        # Zero-init the linear pooler head so semantic tokens start as zeros at
        # init (as in sit.py). "zeropad" has no parameters.
        for pooler in self.sem_poolers:
            if isinstance(pooler.proj, nn.Linear):
                nn.init.constant_(pooler.proj.weight, 0)
                nn.init.constant_(pooler.proj.bias, 0)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def unpatchify(self, x, hw):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h, w = hw[0] // p, hw[1] // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_core_with_concat(self, x, c_mod, context, t=None, teacher_zs=None):
        # context is B, L', D
        # x is B, L, D (L patch tokens; K semantic tokens are appended at encoder_depth)
        B, L, D = x.shape
        attn_mask = None  # built once semantic tokens are injected (see below)
        for i, block in enumerate(self.joint_blocks):
            context, x = block(context, x, c=c_mod, attn_mask=attn_mask)

            if (i + 1) == self.encoder_depth:
                # REPA projections (z_tilde), used by the alignment loss
                zs = [projector(x.reshape(-1, D)).reshape(B, L, -1) for projector in self.projectors]
                # Pooler input: clean teacher features at training, predicted z_tilde at inference
                zs_cond = teacher_zs if teacher_zs is not None else zs
                for z_cond, pooler in zip(zs_cond, self.sem_poolers):
                    sem_tokens = pooler(z_cond)            # (B, K, D)
                    x = torch.cat([x, sem_tokens], dim=1)  # (B, L + K, D)
                # During training, samples with t > sem_inject_max_t cannot attend to the
                # semantic tokens (mask over [context, x]: True = attend)
                if self.sem_poolers and self.training and self.sem_inject_max_t < 1.0:
                    assert t is not None, "t is required for semantic-token masking"
                    ctx_len = context.shape[1] if context is not None else 0
                    Lj = ctx_len + x.shape[1]
                    keep = x.new_ones(B, 1, 1, Lj, dtype=torch.bool)
                    keep[..., ctx_len + L:] = (t <= self.sem_inject_max_t).view(B, 1, 1, 1)
                    attn_mask = keep

        x = x[:, :L]                    # discard semantic tokens
        x = self.final_layer(x, c_mod)  # (N, T, patch_size ** 2 * out_channels)
        return x, zs

    def forward(self, x, t, context, teacher_zs=None):
        """
        x: (N, C, H, W) tensor of spatial inputs (latent representations of images)
        t: (N,) tensor of diffusion timesteps
        context: (N, L, context_dim) text token embeddings
        teacher_zs: list of (N, T, z_dim) clean encoder features for pooler conditioning.
                    Provided during training; None at inference (falls back to predicted z_tilde).
        """
        hw = x.shape[-2:]
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t, dtype=x.dtype)  # (N, D)
        context = self.context_embedder(context)

        x, zs = self.forward_core_with_concat(x, c, context, t=t, teacher_zs=teacher_zs)

        x = self.unpatchify(x, hw=hw)  # (N, out_channels, H, W)
        return x, zs
