import torch
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Dict, Optional, List
from pathlib import Path
from torchvision.transforms import Normalize
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

# pretrained_models/ is a directory in the repository root (this file is <repo>/recon/encoders/)
REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_DIR = REPO_ROOT / "pretrained_models"


def fix_mocov3_state_dict(state_dict):
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder."):]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            if 'head' not in new_k and new_k.split('.')[0] != 'fc':
                state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if 'pos_embed' in state_dict.keys():
        state_dict['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
            state_dict['pos_embed'], [16, 16],
        )
    return state_dict


class VisionEncoder(ABC):
    """Base class for all vision encoders"""
    
    def __init__(self, encoder_type: str, architecture: str, model_config: str, 
                 device: torch.device, resolution: int = 256, accelerator=None):
        self.encoder_type = encoder_type
        self.architecture = architecture
        self.model_config = model_config
        self.device = device
        self.resolution = resolution
        self.accelerator = accelerator
        self._embed_dim = None
        self.model = None
        
    @abstractmethod
    def load_model(self):
        """Load and initialize the encoder model"""
        pass
        
    @abstractmethod
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess raw images
        Args:
            x: Raw images tensor (B, C, H, W) in range [0, 255]
        Returns:
            Preprocessed tensor ready for encoder
        """
        pass
        
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass through encoder
        Args:
            x: Preprocessed images
        Returns:
            Dictionary with:
                - 'x_norm_clstoken': (B, D) CLS token or None if not available
                - 'x_norm_patchtokens': (B, T, D) patch tokens
        """
        # Default implementation - subclasses should override if needed
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            return out
        else:
            # Assume it's just patch tokens
            return {
                'x_norm_clstoken': None,
                'x_norm_patchtokens': out
            }
    
    @property
    def embed_dim(self) -> int:
        return self._embed_dim
    
    def eval(self):
        """Set model to eval mode"""
        if self.model is not None:
            self.model.eval()
        return self
    
    def to(self, device):
        """Move model to device"""
        if self.model is not None:
            self.model = self.model.to(device)
        self.device = device
        return self


class DINOEncoder(VisionEncoder):
    """DINO encoder implementation"""
    
    def load_model(self):
        # Load model from torch hub (rank 0 downloads first)
        model_name = f'dino_vit{self.model_config}16'
        with self.accelerator.main_process_first() if self.accelerator else nullcontext():
            self.model = torch.hub.load('facebookresearch/dino:main', model_name)
        
        # Remove head
        del self.model.head
        self.model.head = torch.nn.Identity()
        
        # Resample position embeddings if needed
        patch_resolution = 16 * (self.resolution // 256)
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [patch_resolution, patch_resolution],
        )
        
        # Set embed dim
        self._embed_dim = self.model.embed_dim
        
        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Apply ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        # Interpolate if needed
        x = torch.nn.functional.interpolate(x, self.resolution, mode='bicubic')
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # DINO returns a dictionary with cls and patch tokens
        out = self.model.get_intermediate_layers(x)[0]
        return {
            'x_norm_clstoken': out[:, 0],
            'x_norm_patchtokens': out[:, 1:]
        }


class DINOv2Encoder(VisionEncoder):
    """DINOv2 encoder implementation. A "noaffine" config suffix (e.g. dinov2reg-vit-bnoaffine)
    replaces the final LayerNorm with one without affine parameters, as in DiffusionBench's DINOv2 encoder."""
    
    def load_model(self):
        # Determine if using register tokens
        use_reg = 'reg' in self.encoder_type
        no_affine = self.model_config.endswith("noaffine")

        # Load model from torch hub (rank 0 downloads first)
        model_name = f'dinov2_vit{self.model_config.removesuffix("noaffine")}14{"_reg" if use_reg else ""}'
        with self.accelerator.main_process_first() if self.accelerator else nullcontext():
            self.model = torch.hub.load('facebookresearch/dinov2:main', model_name)
        
        # Remove head
        del self.model.head
        self.model.head = torch.nn.Identity()
        
        # Resample position embeddings if needed
        patch_resolution = 16 * (self.resolution // 256)
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [patch_resolution, patch_resolution],
        )
        
        # Set embed dim
        self._embed_dim = self.model.embed_dim
        if no_affine:
            self.model.norm = torch.nn.LayerNorm(self._embed_dim, elementwise_affine=False)
        
        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Apply ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        # Interpolate if needed
        x = torch.nn.functional.interpolate(x, 224 * (self.resolution // 256), mode='bicubic')
        return x


class DINOv2MixedEncoder(DINOv2Encoder):
    """DINOv2 encoder with mixed CLS and patch tokens"""
    
    def __init__(self, *args, alpha: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward_features(x)
        cls_token = out['x_norm_clstoken']
        patch_tokens = out['x_norm_patchtokens']
        
        # Mix CLS token into patch tokens
        mixed_patch_tokens = cls_token[:, None, :] * self.alpha + patch_tokens * (1 - self.alpha)
        
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': mixed_patch_tokens
        }


class DINOv3Encoder(VisionEncoder):
    """DINOv3 encoder implementation"""
    
    def load_model(self):
        from recon.encoders.dinov3_loader import load_dinov3, make_dinov3_transform

        self.model = load_dinov3(f"dinov3_vit{self.model_config}")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = make_dinov3_transform(resize_size=self.resolution)

        # Set embed dim
        self._embed_dim = self.model.embed_dim

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class SigLIPEncoder(VisionEncoder):
    """SigLIP encoder implementation"""
    
    def load_model(self):
        from transformers import SiglipVisionModel

        assert self.resolution == 256, "SigLIP2 only supports 256 resolution"

        # Map model config to full model name
        model_map = {
            'b': 'google/siglip-base-patch16-256',
            'l': 'google/siglip-large-patch16-256',
            'so400m': 'google/siglip-so400m-patch14-224',
        }
        
        if self.model_config not in model_map:
            raise ValueError(f"Unknown SigLIP model config: {self.model_config}")

        self.patch_size = 14 if self.model_config == "so400m" else 16

        self.model = SiglipVisionModel.from_pretrained(model_map[self.model_config])
        self.model.to(self.device)
        self.model.eval()
        
        self._embed_dim = self.model.config.hidden_size
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Apply ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, self.patch_size * (self.resolution // 16), mode='bicubic')
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model(x).last_hidden_state
        return {
            'x_norm_clstoken': None,  # SigLIP has no CLS token
            'x_norm_patchtokens': out
        }


class SigLIP2Encoder(VisionEncoder):
    """SigLIP2 encoder implementation"""
    
    def load_model(self):
        from transformers import SiglipVisionModel
        
        assert self.resolution == 256, "SigLIP2 only supports 256 resolution"
        
        # Map model config to full model name
        model_map = {
            'b': 'google/siglip2-base-patch16-256',
            'l': 'google/siglip2-large-patch16-256',
            'so400m': 'google/siglip2-so400m-patch16-256',
            'g': 'google/siglip2-giant-opt-patch16-256'
        }
        
        if self.model_config not in model_map:
            raise ValueError(f"Unknown SigLIP2 model config: {self.model_config}")

        self.model = SiglipVisionModel.from_pretrained(model_map[self.model_config])
        self.model.to(self.device)
        self.model.eval()
        
        self._embed_dim = self.model.config.hidden_size
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Apply ImageNet normalization
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, self.resolution, mode='bicubic')
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model(x).last_hidden_state
        return {
            'x_norm_clstoken': None,  # SigLIP has no CLS token
            'x_norm_patchtokens': out
        }


class CLIPEncoder(VisionEncoder):
    """CLIP encoder implementation"""
    
    def load_model(self):
        import clip
        from recon.encoders.clip_vit import UpdatedVisionTransformer

        encoder_ = clip.load(f"ViT-{self.model_config}/14", device='cpu')[0].visual
        self.model = UpdatedVisionTransformer(encoder_).to(self.device)

        patch_resolution = 16 * (self.resolution // 256)
        self.model.model.positional_embedding.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.model.positional_embedding.data.unsqueeze(0), [patch_resolution, patch_resolution],
        ).squeeze(0)
        self._embed_dim = self.model.model.transformer.width
        self.model.eval()
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        x = x / 255.
        # Interpolate for CLIP
        resolution = x.shape[-1]
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        # Apply CLIP normalization
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # UpdatedVisionTransformer already drops the CLS token
        return {
            'x_norm_clstoken': None,
            'x_norm_patchtokens': self.model(x)
        }


class MoCoV3Encoder(VisionEncoder):
    """MoCoV3 encoder implementation"""
    
    def load_model(self):
        from recon.encoders import mocov3_vit

        # Create model based on config
        if self.model_config == 's':
            self.model = mocov3_vit.vit_small()
        elif self.model_config == 'b':
            self.model = mocov3_vit.vit_base()
        elif self.model_config == 'l':
            self.model = mocov3_vit.vit_large()
        else:
            raise ValueError(f"Unknown MoCoV3 model config: {self.model_config}")
        
        # Load checkpoint
        ckpt = torch.load(PRETRAINED_DIR / f'mocov3_vit{self.model_config}.pth', weights_only=False)
        state_dict = fix_mocov3_state_dict(ckpt['state_dict'])
        del self.model.head
        self.model.load_state_dict(state_dict, strict=True)
        self.model.head = torch.nn.Identity()
        
        self.model = self.model.to(self.device)
        self.model.eval()

        patch_resolution = 16 * (self.resolution // 256)
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [patch_resolution, patch_resolution],
        )

        # Set embed dim
        self._embed_dim = self.model.embed_dim
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # MoCoV3 returns CLS + patch tokens, we skip CLS (index 0)
        out = self.model.forward_features(x)  # By default MoCov3 uses CLS token
        cls_token = out[:, 0]
        patch_tokens = out[:, 1:]
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens,
        }


class MAEEncoder(VisionEncoder):
    """MAE encoder implementation"""
    
    def load_model(self):
        from recon.encoders.mae_vit import vit_large_patch16

        assert self.resolution == 256, "MAE only supports 256 resolution"

        kwargs = dict(img_size=256)
        self.model = vit_large_patch16(**kwargs).to(self.device)
        
        with open(PRETRAINED_DIR / f"mae_vit{self.model_config}.pth", "rb") as f:
            state_dict = torch.load(f, weights_only=False)
        
        if 'pos_embed' in state_dict["model"].keys():
            state_dict["model"]['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                state_dict["model"]['pos_embed'], [16, 16],
            )
        
        self.model.load_state_dict(state_dict["model"])
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [16, 16],
        )
        
        self._embed_dim = self.model.embed_dim
        self.model.eval()
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        return x


class JEPAEncoder(VisionEncoder):
    """I-JEPA encoder implementation"""
    
    def load_model(self):
        from recon.encoders.jepa import vit_huge

        kwargs = dict(img_size=[224, 224], patch_size=14)
        self.model = vit_huge(**kwargs).to(self.device)
        
        with open(PRETRAINED_DIR / f"ijepa_vit{self.model_config}.pth", "rb") as f:
            state_dict = torch.load(f, map_location=self.device, weights_only=False)
        
        new_state_dict = dict()
        for key, value in state_dict['encoder'].items():
            new_state_dict[key[7:]] = value
        
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        patch_resolution = 16 * (self.resolution // 256)
        self.model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.model.pos_embed.data, [patch_resolution, patch_resolution], num_prefix_tokens=0
        )

        self._embed_dim = self.model.embed_dim
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (self.resolution // 256), mode='bicubic')
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward(x)  # JEPA doesn't have CLS token
        return {
            'x_norm_clstoken': None,
            'x_norm_patchtokens': out
        }


class WebSSLEncoder(VisionEncoder):
    """WebSSL encoder implementation"""
    
    def load_model(self):
        from transformers import Dinov2Model

        model_name = f"facebook/webssl-{self.model_config.replace('_', '-')}"
        self.model = Dinov2Model.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self._embed_dim = self.model.config.hidden_size

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (self.resolution // 256), mode='bicubic')
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # Skip CLS token (index 0)
        out = self.model.forward(x).last_hidden_state
        cls_token = out[:, 0]
        patch_tokens = out[:, 1:]
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens
        }


class PEEncoder(VisionEncoder):
    """PE (Perceptual Encoder) implementation"""
    
    def load_model(self):
        import recon.encoders.pe as pe
        
        # Check if using normalization
        self.use_norm = self.model_config.endswith("norm")
        if self.use_norm:
            config_name = self.model_config[:-4]
        else:
            config_name = self.model_config
        
        # Map config to model name
        if self.encoder_type == "pe":
            config_map = {
                "t": "PE-Core-T16-384",
                "s": "PE-Core-S16-384",
                "b": "PE-Core-B16-224",
                "l": "PE-Core-L14-336",
                "g": "PE-Core-G14-448"
            }
        elif self.encoder_type == "spatialpe":
            config_map = {
                "b": "PE-Spatial-B16-512",
                "l": "PE-Spatial-L14-448",
                "g": "PE-Spatial-G14-448"
            }
        elif self.encoder_type == "langpe":
            config_map = {
                "l": "PE-Lang-L14-448",
                "g": "PE-Lang-G14-448"
            }
        else:
            raise ValueError(f"Unknown PE encoder type: {self.encoder_type}")
        
        if config_name not in config_map:
            raise ValueError(f"Unknown PE model config: {config_name}")
        
        self.model = pe.VisionTransformer.from_config(config_map[config_name], pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self._embed_dim = self.model.width
        
        # Get patch size for preprocessing
        if config_name in {"t", "s", "b", "tnorm", "snorm", "bnorm"}:
            self.patch_size = 16
        elif config_name in {"l", "g", "lnorm", "gnorm"}:
            self.patch_size = 14
        else:
            raise NotImplementedError()
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.
        x = torch.nn.functional.interpolate(
            x, self.patch_size * (self.resolution // 16), mode='bilinear'
        )
        x = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # PE returns patch tokens without CLS
        out = self.model.forward_features(x, norm=self.use_norm, strip_cls_token=False)
        if self.model.use_cls_token:
            cls_token = out[:, 0]
            patch_tokens = out[:, 1:]
        else:
            cls_token = None
            patch_tokens = out
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens
        }


# Registry mapping encoder types to classes
ENCODER_REGISTRY = {
    'dino': DINOEncoder,
    # dinov2 and dinov3 encoders
    'dinov2': DINOv2Encoder,
    'dinov2reg': DINOv2Encoder,
    'dinov2mixed': DINOv2MixedEncoder,
    'dinov2mixedreg': DINOv2MixedEncoder,
    'dinov3': DINOv3Encoder,
    # older encoders
    'siglip': SigLIPEncoder,
    'siglip2': SigLIP2Encoder,
    'clip': CLIPEncoder,
    'mocov3': MoCoV3Encoder,
    'mae': MAEEncoder,
    'jepa': JEPAEncoder,
    # webssl encoder
    'webssl': WebSSLEncoder,
    # PE encoders
    'pe': PEEncoder,
    'spatialpe': PEEncoder,
    'langpe': PEEncoder,
}


def create_encoder(encoder_string: str, device: torch.device, 
                   resolution: int = 256, accelerator=None) -> VisionEncoder:
    """
    Factory function to create encoder from string specification
    
    Args:
        encoder_string: Format "encoder_type-architecture-model_config"
        device: torch device
        resolution: Input image resolution
        accelerator: Optional accelerator for distributed training
    
    Returns:
        VisionEncoder instance
    """
    parts = encoder_string.split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid encoder string format: {encoder_string}. "
                        f"Expected format: encoder_type-architecture-model_config")
    
    encoder_type, architecture, model_config = parts
    
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Available types: {list(ENCODER_REGISTRY.keys())}")
    
    encoder_class = ENCODER_REGISTRY[encoder_type]
    encoder = encoder_class(encoder_type, architecture, model_config, 
                            device, resolution, accelerator)
    encoder.load_model()
    
    return encoder


def load_encoders(enc_type: str, device: torch.device, resolution: int = 256,
                  accelerator=None) -> List[VisionEncoder]:
    """Load the comma-separated teacher encoders in `enc_type`, e.g. "dinov2-vit-b"."""
    return [create_encoder(name, device, resolution, accelerator).eval() for name in enc_type.split(',')]
