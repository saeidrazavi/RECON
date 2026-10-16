# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field
from timm.models.vision_transformer import PatchEmbed, Mlp, Attention
from typing import List

from recon.models.common import FinalLayer, SemanticTokenPooler, build_mlp, get_2d_sincos_pos_embed, modulate


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.positional_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_norm=False, fused_attn=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm, norm_layer=nn.LayerNorm,
        )
        self.attn.fused_attn = fused_attn
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        z_dims=[768],
        projector_dim=2048,
        fused_attn=True,
        qk_norm=False,
        num_semantic_tokens=4,
        sem_inject_max_t=1.0,
        sem_proj_type="linear",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth
        self.fused_attn = fused_attn
        self.num_semantic_tokens = num_semantic_tokens
        self.sem_proj_type = sem_proj_type
        # Semantic tokens are only injected for timesteps t <= sem_inject_max_t.
        # For t above the threshold (high noise) the predicted features are
        # unreliable, so patch tokens are masked from attending to them.
        self.sem_inject_max_t = sem_inject_max_t
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm, fused_attn=fused_attn) for _ in range(depth)
        ])
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
        ])
        self.sem_poolers = nn.ModuleList([
            SemanticTokenPooler(hidden_size, z_dim, num_semantic_tokens, proj_type=sem_proj_type)
            for z_dim in z_dims
        ] if num_semantic_tokens > 0 else [])  # 0 semantic tokens: plain REPA
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out proj in all sem_poolers so semantic tokens start as zeros at init.
        # Only applies to the "linear" head; "zeropad" has no parameters, so its
        # semantic tokens carry the pooled feature from the very first step.
        for pooler in self.sem_poolers:
            if isinstance(pooler.proj, nn.Linear):
                nn.init.constant_(pooler.proj.weight, 0)
                nn.init.constant_(pooler.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def forward(self, x, t, y, teacher_zs=None, sem_extrapolation=None):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        teacher_zs: list of (N, T, z_dim) clean encoder features for pooler conditioning.
                    Provided during training; None at inference (falls back to predicted z_tilde).
        sem_extrapolation: optional SemanticTokenExtrapolation applied to the pooled semantic token (sampling only).
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape

        # timestep and class embedding
        t_embed = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y                                # (N, D)

        attn_mask = None  # built once semantic tokens are injected (see below)
        for i, block in enumerate(self.blocks):
            x = block(x, c, attn_mask=attn_mask)
            if (i + 1) == self.encoder_depth:
                # REPA projections (z_tilde), used by the alignment loss
                zs = [projector(x[:, :T].reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
                # Pooler input: clean teacher features at training, predicted z_tilde at inference
                zs_cond = teacher_zs if teacher_zs is not None else zs
                for z_cond, pooler in zip(zs_cond, self.sem_poolers):
                    sem_tokens = pooler(z_cond, extrapolation=sem_extrapolation)  # (N, K, D)
                    x = torch.cat([x, sem_tokens], dim=1)  # (N, T + K, D)
                # During training, samples with t > sem_inject_max_t cannot attend to the
                # semantic tokens (mask: True = attend)
                if self.sem_poolers and self.training and self.sem_inject_max_t < 1.0:
                    L = x.shape[1]
                    keep = x.new_ones(N, 1, 1, L, dtype=torch.bool)
                    keep[..., T:] = (t <= self.sem_inject_max_t).view(N, 1, 1, 1)
                    attn_mask = keep

        x = x[:, :T]                              # discard semantic tokens
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                    # (N, out_channels, H, W)

        return x, zs


#################################################################################
#                                   SiT Configs                                 #
#################################################################################
@dataclass
class SiTConfig:
    input_size: int = 16
    patch_size: int = 1
    in_channels: int = 32
    hidden_size: int = 768
    decoder_hidden_size: int = 768
    encoder_depth: int = 4
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    z_dims: List[int] = field(default_factory=lambda: [768])
    projector_dim: int = 2048
    fused_attn: bool = True
    qk_norm: bool = False
    num_semantic_tokens: int = 4
    sem_inject_max_t: float = 1.0
    sem_proj_type: str = "linear"


MODEL_SIZES = {
    "B":  {"depth": 12, "hidden_size": 768,  "decoder_hidden_size": 768,  "num_heads": 12},
    "L":  {"depth": 24, "hidden_size": 1024, "decoder_hidden_size": 1024, "num_heads": 16, "encoder_depth": 8},
    "XL": {"depth": 28, "hidden_size": 1152, "decoder_hidden_size": 1152, "num_heads": 16, "encoder_depth": 8},
}


def build_sit(**overrides: dict):
    return SiT(**SiTConfig(**overrides).__dict__)
