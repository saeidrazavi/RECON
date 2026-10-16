"""Text-to-image evaluation with the EMA model: FID and CLIP score.

evaluate (MMDiT) follows REPA's MS-COCO protocol, and evaluate_ddt (DDT) the BLIP3o protocol of
DiffusionBench.
"""

import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Normalize
from tqdm import tqdm

from recon.diffusion.samplers import euler_sampler, euler_maruyama_sampler
from recon.diffusion.samplers_ddt import euler_sampler as ddt_euler_sampler
from recon.eval.c2i import compute_metrics
from recon.eval.fid_inception import InceptionV3


@torch.no_grad()
def compute_inception_stats(images, device, batch_size=128, verbose=False):
    """mu/sigma of pool3 features, pytorch-fid convention.

    images: uint8 numpy array (N, H, W, 3).
    """
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    feats = []
    it = range(0, len(images), batch_size)
    if verbose:
        from tqdm import tqdm
        it = tqdm(it, desc="inception")
    for i in it:
        x = torch.from_numpy(images[i:i + batch_size]).to(device)
        x = x.permute(0, 3, 1, 2).float() / 255.0  # pytorch-fid: float in [0, 1]
        f = model(x)[0].squeeze(-1).squeeze(-1)
        feats.append(f.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    feats = np.concatenate(feats).astype(np.float64)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance (pytorch-fid, verbatim)."""
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipSimilarity:
    """Mean CLIP cosine similarity between images and their captions."""

    def __init__(self, device, model_name="openai/clip-vit-large-patch14"):
        from transformers import CLIPModel, CLIPTokenizer
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.device = device
        self.normalize = Normalize(CLIP_MEAN, CLIP_STD)

    @torch.no_grad()
    def __call__(self, images, captions):
        """images: float tensor (B, 3, H, W) in [0, 1]; captions: list of B strings.
        Returns per-sample cosine similarities, shape (B,)."""
        x = F.interpolate(images.float(), size=224, mode="bicubic", align_corners=False)
        x = self.normalize(x.clamp(0, 1)).to(self.device)
        tokens = self.tokenizer(
            list(captions), truncation=True, max_length=77,
            padding="max_length", return_tensors="pt",
        )["input_ids"].to(self.device)
        out = self.model(input_ids=tokens, pixel_values=x)
        img_emb = F.normalize(out.image_embeds, dim=-1)
        txt_emb = F.normalize(out.text_embeds, dim=-1)
        return (img_emb * txt_emb).sum(dim=-1)


@torch.no_grad()
def evaluate(model, vae, latents_scale, latents_bias, accelerator, val_dataset, empty_context, ref_stats,
             num_samples, batch_size, mode="ode", heun=True, num_steps=50, cfg_scale=2.0,
             guidance_low=0., guidance_high=1., clip_model="openai/clip-vit-large-patch14",
             sample_dir=None, save_npz=False, fid_batch_size=128):
    """Generate one image for each of the first num_samples captions of val_dataset, then compute
    FID against ref_stats (a pytorch-fid mu/sigma npz) and the mean CLIP similarity (skipped when
    clip_model is None). With sample_dir, the images are also saved there as PNGs, plus one npz
    with save_npz. Returns (fid, clip_sim) on the main process and (None, None) elsewhere."""
    device = accelerator.device
    rank, world_size = accelerator.process_index, accelerator.num_processes
    global_batch_size = batch_size * world_size
    assert num_samples % global_batch_size == 0, \
        f"num_samples={num_samples} must be divisible by batch*world={global_batch_size}"
    loader = accelerator.prepare(DataLoader(
        Subset(val_dataset, range(num_samples)), batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True, drop_last=False,
    ))
    y_null = torch.from_numpy(empty_context).to(device).unsqueeze(0)
    latent_size = model.x_embedder.img_size[0]
    sample_fn = euler_maruyama_sampler if mode == "sde" else euler_sampler
    if sample_dir is not None and accelerator.is_main_process:
        os.makedirs(sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    clip_fn = ClipSimilarity(device=device, model_name=clip_model) if clip_model else None

    images, clip_sum, clip_count, total = [], 0., 0, 0
    pbar = tqdm(total=num_samples // global_batch_size, disable=not accelerator.is_main_process)
    for _, _, context, captions in loader:
        z = torch.randn(context.shape[0], model.in_channels, latent_size, latent_size, device=device)
        samples = sample_fn(
            model, z, {"context": context}, {"context": y_null.repeat(context.shape[0], 1, 1)},
            num_steps=num_steps, heun=heun, cfg_scale=cfg_scale,
            guidance_low=guidance_low, guidance_high=guidance_high,
        ).to(torch.float32)
        samples = (vae.decode(samples / latents_scale + latents_bias).sample + 1) / 2.
        # contiguous NHWC, as images read back from PNG are (Inception's convolutions then run
        # in channels-last, as they always have for this FID)
        batch = torch.clamp(255. * samples, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
        if sample_dir is not None:
            for i, image in enumerate(batch.cpu().numpy()):
                Image.fromarray(image).save(f"{sample_dir}/{i * world_size + rank + total:06d}.png")
        # gathered rank-major; reorder to the PNG index order (i * world_size + rank)
        batch = accelerator.gather(batch)
        if accelerator.is_main_process:
            images.append(batch.unflatten(0, (world_size, -1)).transpose(0, 1).flatten(0, 1).cpu().numpy())
        if clip_fn is not None:
            sims = accelerator.gather(clip_fn(torch.clamp(samples, 0, 1), captions))
            clip_sum += sims.sum().item()
            clip_count += sims.numel()
            pbar.set_postfix(clip_sim=clip_sum / clip_count)
        total += global_batch_size
        pbar.update(1)
    pbar.close()
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return None, None

    images = np.concatenate(images)
    if sample_dir is not None and save_npz:
        np.savez(f"{sample_dir}.npz", arr_0=images)
        print(f"Saved .npz file to {sample_dir}.npz [shape={images.shape}].")
    ref = np.load(ref_stats)
    print("Computing Inception statistics of samples...")
    mu, sigma = compute_inception_stats(images, device=device, batch_size=fid_batch_size, verbose=True)
    fid = calculate_frechet_distance(mu, sigma, ref["mu"], ref["sigma"])
    clip_sim = (clip_sum / clip_count) if clip_count else float("nan")
    return fid, clip_sim


@torch.no_grad()
def evaluate_ddt(model, vae, latents_scale, latents_bias, accelerator, text_encoder, prompts, ref_stats,
                 shard_dir, batch_size=64, num_steps=50, cfg_scale=6.0, clip_model="openai/clip-vit-large-patch14",
                 npz_path=None, fid_batch_size=128):
    """DiffusionBench's DDT eval: one image per prompt (contiguous chunks of `prompts` per rank),
    sampled with the Euler ODE and CFG and decoded under bf16 autocast, then FID against ref_stats
    (mu / sigma of torch-fidelity Inception features) and the mean CLIP similarity (skipped when
    clip_model is None). The images of each rank go through an .npz shard in shard_dir; with npz_path
    they are also saved there. Returns (fid, clip_sim) on the main process and (None, None) elsewhere."""
    device, rank, world_size = accelerator.device, accelerator.process_index, accelerator.num_processes
    chunk = len(prompts) // world_size
    prompts = prompts[rank * chunk: len(prompts) if rank == world_size - 1 else (rank + 1) * chunk]
    latent_size = model.x_embedder.img_size[0]
    clip_fn = ClipSimilarity(device=device, model_name=clip_model) if clip_model else None
    if accelerator.is_main_process:
        os.makedirs(shard_dir, exist_ok=True)

    images, clip_sum, clip_count = [], 0., 0
    for i in tqdm(range(0, len(prompts), batch_size), disable=not accelerator.is_main_process):
        captions = prompts[i:i + batch_size]
        z = torch.randn(len(captions), model.in_channels, latent_size, latent_size, device=device)
        context, attn_mask = text_encoder(captions)
        null_context, null_mask = text_encoder([""] * len(captions))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            samples = ddt_euler_sampler(
                model, z, {"context": context, "attn_mask": attn_mask},
                {"context": null_context, "attn_mask": null_mask}, num_steps=num_steps, cfg_scale=cfg_scale,
            )
            samples = ((vae.decode(samples / latents_scale + latents_bias).sample + 1) / 2).clamp(0, 1)
        # uint8 by truncation, as in DiffusionBench; contiguous NHWC, as the FID input always is
        images.append(samples.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).contiguous().numpy())
        if clip_fn is not None:
            clip_sum += clip_fn(samples.float(), captions).sum().item()
            clip_count += len(captions)
    del clip_fn
    shard = os.path.join(shard_dir, f"ddt_eval_shard_{rank:02d}.npz")
    accelerator.wait_for_everyone()
    np.savez(shard, arr_0=np.concatenate(images))
    clip_totals = accelerator.reduce(torch.tensor([clip_sum, clip_count], dtype=torch.float64, device=device), "sum")
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return None, None

    shards = [os.path.join(shard_dir, f"ddt_eval_shard_{r:02d}.npz") for r in range(world_size)]
    images = np.concatenate([np.load(f)["arr_0"] for f in shards])
    for f in shards:
        os.remove(f)
    if npz_path is not None:
        np.savez(npz_path, arr_0=images)
        print(f"Saved .npz file to {npz_path} [shape={images.shape}].")
    fid = compute_metrics(images, np.load(ref_stats), batch_size=fid_batch_size, device=str(device))[0]
    clip_sim = (clip_totals[0] / clip_totals[1]).item() if clip_totals[1] > 0 else float("nan")
    return fid, clip_sim


def append_eval_csv(csv_path, step, fid, clip_sim, **settings):
    """Append one row to csv_path (writing the header for a new file): the step, the sampling
    settings and the metrics."""
    row = {"step": step, **settings, "fid": f"{fid:.4f}", "clip_sim": f"{clip_sim:.4f}"}
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
