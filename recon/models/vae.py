# --vae-name keys. recon/data/imagenet.py reads the latents from <data-dir>/imagenet-latents-<key>,
# and preprocessing/README.md lists the hosted latents. The REPA-E VAEs use their "-hf" repos,
# which AutoencoderKL.from_pretrained can read.
HF_VAE_MAPPER = {
    "sdvae-ft-ema-f8d4": "stabilityai/sd-vae-ft-ema",
    "sdvae-ft-mse-f8d4": "stabilityai/sd-vae-ft-mse",
    "invae-f16d32": "REPA-E/invae-hf",
    "vavae-f16d32": "REPA-E/vavae-hf",
    "e2e-vavae-f16d32": "REPA-E/e2e-vavae-hf",
    "e2e-sdvae-f8d4": "REPA-E/e2e-sdvae-hf",
}

HF_VAE_DS_FACTOR = {
    "sdvae-ft-ema-f8d4": 8,
    "sdvae-ft-mse-f8d4": 8,
    "invae-f16d32": 16,
    "vavae-f16d32": 16,
    "e2e-vavae-f16d32": 16,
    "e2e-sdvae-f8d4": 8,
}


def load_vae(name, device):
    """Load the VAE `name` together with the statistics that normalize its latents
    for the diffusion model: z_model = (z - latents_bias) * latents_scale.

    The REPA-E VAEs store per-channel latents_mean / latents_std in their config;
    SD-VAE only has the scalar scaling_factor (0.18215) and no shift.
    """
    import torch
    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(HF_VAE_MAPPER[name]).to(device)
    config = vae.config
    if getattr(config, "latents_std", None) is not None:
        latents_scale = 1 / torch.tensor(config.latents_std).view(1, -1, 1, 1).to(device)
        latents_bias = torch.tensor(config.latents_mean).view(1, -1, 1, 1).to(device)
    else:
        latents_scale = config.scaling_factor
        latents_bias = 0.0
    return vae, latents_scale, latents_bias
